#!/usr/bin/env python3
r"""Run SPARC-RAG inference over a dataset.

Loads a jsonnet config, builds the participants and the ModelController, and streams
predictions for every question in --eval_path into predictions/<trial_name>/<run_id>/:
  prediction__<eval-file-stem>.json       qid -> predicted answer
  prediction__<eval-file-stem>_all.json   qid -> prediction, support_docs (paragraphs retrieved on
                                          all branches), paths (inference steps of the selected
                                          path) and token_count (completion tokens of the LLM
                                          calls on the selected path, see
                                          StructuredDataInstance.get_total_token_count)
  log_graph__<eval-file-stem>.json        every step and branch of every question
  config__<eval-file-stem>.jsonnet, all_prompts.json, process.log
While the run is in progress, the two prediction files are written with a _temp suffix; a run with
the same --run_id resumes from them (or from the final files) and skips the questions already in
them.

Usually started through exp_helper.py, which materialises the config first. Direct use:
    python predict.py --config_path configs/qwen2.5_7b_dpo_musique.jsonnet \
        --eval_path processed_data/musique/test_subsampled.jsonl \
        --trial_name my_experiment --run_id my_run_001
"""

import argparse
import json
import logging
import os
import time
import uuid

import _jsonnet

from inference.dataset_readers import MultiParaRCReader
from inference.model_control import ModelController
from inference.model_map import MODEL_NAME_CLASS
from lib import get_llm_server_address
from logger import RunLogger

CONFIG_KEYS = {"start_state", "end_state", "controller", "models"}
CONTROLLER_KEYS = {"fixed_width", "fixed_length", "max_length"}


def load_config(config_file: str) -> dict:
    """Load a json or jsonnet config."""
    if config_file.endswith(".jsonnet"):
        return json.loads(_jsonnet.evaluate_file(config_file))
    with open(config_file, "r") as fp:
        return json.load(fp)


def _gen_run_id():
    return time.strftime("%Y%m%d-%H%M%S") + "_" + uuid.uuid4().hex[:6]


def get_controller(config_map, logger=None):
    """Build the participants and the controller described by a config.

    ``controller.fixed_width`` (W >= 1) is the number of branches per round.
    ``controller.fixed_length`` > -1 runs exactly that many rounds: it sets
    ``controller.max_length`` and makes the answer evaluator reject every answer. With
    ``fixed_length`` = -1 the answer evaluator decides when to stop, within ``max_length`` rounds.
    """
    unknown = set(config_map) - CONFIG_KEYS
    if unknown:
        raise ValueError(f"Unknown config key(s): {sorted(unknown)}")
    controller_config = config_map.get("controller", {})
    unknown = set(controller_config) - CONTROLLER_KEYS
    if unknown:
        raise ValueError(f"Unknown controller key(s): {sorted(unknown)}")
    models = config_map["models"]

    fixed_length = controller_config.get("fixed_length", -1)
    if fixed_length > -1:
        controller_config["max_length"] = fixed_length
        if "answer_eval" in models:
            models["answer_eval"]["always_reject"] = True

    width = controller_config.get("fixed_width")
    if not isinstance(width, int) or width < 1:
        raise ValueError(f"controller.fixed_width must be an integer >= 1, got {width!r}.")
    rewrite = models.get("rewrite", {})
    if width > 1 and not rewrite.get("dispatch_prompt_file"):
        raise ValueError("controller.fixed_width > 1 needs models.rewrite.dispatch_prompt_file.")
    if width == 1 and not rewrite.get("prompt_file"):
        raise ValueError("controller.fixed_width = 1 needs models.rewrite.prompt_file.")

    participants = {}
    for key, value in models.items():
        value = dict(value)
        class_name = value.pop("name")
        if class_name not in MODEL_NAME_CLASS:
            raise ValueError(f"No class mapped to model name: {class_name}")
        participants[key] = MODEL_NAME_CLASS[class_name](**value)
    return ModelController(
        participants,
        start_state=config_map["start_state"],
        end_state=config_map["end_state"],
        width=width,
        max_length=controller_config.get("max_length", -1),
        logger=logger,
    )


def _load_previous(logger: RunLogger, *paths: str) -> dict:
    """Results of an earlier run with the same run id: the first existing file of ``paths``."""
    for path in paths:
        if os.path.exists(path):
            with open(path, "r") as fp:
                results = json.load(fp)
            logger.log(f"[resume] loaded {path}")
            return results
    return {}


def run_inference(args, reader, controller, logger: RunLogger):
    logger.log("Running inference on examples")
    prediction_filename = args.output
    temp_all_path = logger.path_temp_all(prediction_filename)
    temp_pred_path = logger.path_temp_predictions(prediction_filename)

    temp_all = _load_previous(logger, temp_all_path, logger.path_final_all(prediction_filename))
    temp_preds = _load_previous(logger, temp_pred_path, logger.path_final_predictions(prediction_filename))

    processed_qids = set(temp_all.keys())
    start = time.time()
    for example in reader.read_examples(args.input):
        qid = example["qid"]
        if args.test_qid:
            if qid != args.test_qid:
                continue
        elif qid in processed_qids:
            continue

        _, ans, final_state, docs = controller.answer(example, debug=args.debug)
        temp_all[qid] = {
            "prediction": ans,
            "support_docs": docs,
            "paths": final_state.data.get_structured_path(),
            "token_count": final_state.data.get_total_token_count(),
        }
        temp_preds[qid] = ans

        with open(temp_all_path, "w") as fp:
            json.dump(temp_all, fp, indent=4)
        with open(temp_pred_path, "w") as fp:
            json.dump(temp_preds, fp, indent=4)

        processed_qids.add(qid)
        if args.cut_at and len(processed_qids) >= args.cut_at:
            break

    duration = int(time.time() - start)
    logger.log(f"Completed inference in {duration}s. Finalizing outputs.")

    logger.finalize_predictions(prediction_filename, temp_all, temp_preds)


def main():
    parser = argparse.ArgumentParser(
        description="Run SPARC-RAG inference on given config and dataset."
    )
    parser.add_argument("--config_path", type=str, required=True, help="config path")
    parser.add_argument("--eval_path", type=str, required=True, help="evaluation file (.jsonl)")
    parser.add_argument(
        "--debug", action="store_true", default=False, help="Debug output"
    )
    parser.add_argument(
        "--cut_at", type=int, default=0, help="stop after N questions (0: run all)"
    )
    parser.add_argument(
        "--test_qid",
        type=str,
        default=None,
        help="If specified, only run inference on this specific question ID.",
    )
    parser.add_argument(
        "--trial_name",
        type=str,
        default="default",
        help="Name of the set for predictions, used to create directories.",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Run ID; if omitted we generate YYYYMMDD-HHMMSS_xxxxxx.",
    )
    parser.add_argument(
        "--predictions_dir",
        type=str,
        default="predictions",
        help="Base predictions directory (default: predictions).",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.ERROR)

    run_id = args.run_id or _gen_run_id()
    prediction_filename = os.path.splitext(os.path.basename(args.eval_path))[0]
    logger = RunLogger(
        trial_name=args.trial_name,
        run_id=run_id,
        prediction_filename=prediction_filename,
        predictions_dir=args.predictions_dir,
    )
    logger.log(f"[run] trial={args.trial_name}  run_id={run_id}")
    logger.log(f"[run_dir] {logger.run_dir}")

    # The LLM client (models/llm_client_generator.py) reads the server address from these.
    llm_server_address = get_llm_server_address()
    os.environ["LLM_SERVER_HOST"] = llm_server_address["host"]
    os.environ["LLM_SERVER_PORT"] = llm_server_address["port"]

    logger.backup_config(args.config_path, prediction_filename)
    config_map = load_config(args.config_path)
    logger.log("Loaded config_map:")
    logger.log(json.dumps(config_map, indent=2))

    # Snapshot of every prompt file the config names (prompt_file, dispatch_prompt_file).
    prompt_files = sorted(
        {
            value
            for model_cfg in config_map.get("models", {}).values()
            for key, value in model_cfg.items()
            if key.endswith("prompt_file") and value
        }
    )
    all_prompts = {}
    for pf in prompt_files:
        with open(pf, "r") as f:
            all_prompts[pf] = f.read()
    logger.save_prompts(all_prompts)

    controller = get_controller(config_map, logger)
    inf_args = argparse.Namespace(
        input=args.eval_path,
        output=prediction_filename,
        cut_at=args.cut_at,
        debug=args.debug,
        test_qid=args.test_qid,
    )
    try:
        run_inference(inf_args, MultiParaRCReader(), controller, logger)
        for name, participant in controller.participants.items():
            logger.log(f"[stats] {name}: {json.dumps(participant.get_stats(), sort_keys=True)}")
    finally:
        logger.close()


if __name__ == "__main__":
    main()
