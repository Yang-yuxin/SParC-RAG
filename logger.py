"""Run directory, artifacts and per-question log graphs of a predict.py run."""

import json
import os
import shutil
import time


class RunLogger:
    """
    Paths, config backup, prompt snapshot, process log and log graphs of the run in
    predictions/<trial_name>/<run_id>/.

    The log graph of a question (log_graph__<eval-file-stem>.json, keyed by qid) has one node per
    inference step, dispatch and branch, and edges for the transitions between them. Unlike the
    predictions' ``paths``, it keeps the steps of discarded branches.
    """

    def __init__(self, trial_name: str, run_id: str, prediction_filename: str = None, predictions_dir: str = "predictions"):
        self.trial_name = trial_name
        self.run_id = run_id
        self.run_dir = os.path.join(predictions_dir, trial_name, run_id)
        os.makedirs(self.run_dir, exist_ok=True)
        self._log_fp = None  # opened lazily
        self.log_graph = {"nodes": [], "edges": [], "qa_metadata": None}  # graph-based logging structure
        self.prediction_filename = prediction_filename

    # ---- artifact paths ----
    def _prediction_base(self, prediction_filename: str) -> str:
        return os.path.join(self.run_dir, f"prediction__{prediction_filename}.json")

    def path_temp_all(self, prediction_filename: str) -> str:
        base = self._prediction_base(prediction_filename)
        i = base.rfind(".")
        return base[:i] + "_temp_all.json"

    def path_temp_predictions(self, prediction_filename: str) -> str:
        """
        predictions-only temp file, parallel to _temp_all.json
        """
        base = self._prediction_base(prediction_filename)
        i = base.rfind(".")
        return base[:i] + "_temp.json"

    def path_final_predictions(self, prediction_filename: str) -> str:
        """
        final predictions-only file
        """
        return self._prediction_base(prediction_filename)

    def path_final_all(self, prediction_filename: str) -> str:
        base = self._prediction_base(prediction_filename)
        i = base.rfind(".")
        return base[:i] + "_all.json"

    def path_config_backup(self, prediction_filename: str) -> str:
        return os.path.join(self.run_dir, f"config__{prediction_filename}.jsonnet")

    def path_prompts(self) -> str:
        return os.path.join(self.run_dir, "all_prompts.json")

    def path_process_log(self) -> str:
        return os.path.join(self.run_dir, "process.log")

    # ---- operations ----
    def backup_config(self, src_config_path: str, prediction_filename: str) -> None:
        dst = self.path_config_backup(prediction_filename)
        # Skip if source and destination are the same file (e.g., when resuming a run)
        if os.path.realpath(src_config_path) == os.path.realpath(dst):
            self.log(f"[backup] skipped (same file): {src_config_path}")
            return
        shutil.copyfile(src_config_path, dst)
        self.log(f"[backup] {src_config_path} -> {dst}")

    def save_prompts(self, prompts_map: dict) -> None:
        out = self.path_prompts()
        with open(out, "w") as f:
            json.dump(prompts_map, f, indent=2)
        self.log(f"[prompts] saved -> {out}")

    def finalize_predictions(self, prediction_filename: str, all_results: dict, predictions: dict) -> None:
        """Write the final prediction files and remove the temp files they replace."""
        for temp_path, final_path, data in (
            (self.path_temp_all(prediction_filename), self.path_final_all(prediction_filename), all_results),
            (self.path_temp_predictions(prediction_filename), self.path_final_predictions(prediction_filename),
             predictions),
        ):
            with open(final_path, "w") as fp:
                json.dump(data, fp, indent=4)
            if os.path.exists(temp_path):
                os.remove(temp_path)
            self.log(f"[finalize] wrote {final_path}")

    def log(self, msg: str) -> None:
        print(msg)
        if self._log_fp is None:
            self._log_fp = open(self.path_process_log(), "a", encoding="utf-8")
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        self._log_fp.write(f"[{ts}] {msg}\n")
        self._log_fp.flush()

    def _extract_qa_metadata(self, state) -> dict:
        """Extract QA task metadata from state"""
        data = state.data
        metadata = {}
        if "qid" in data:
            metadata["qid"] = data["qid"]
        if "question" in data:
            metadata["original_question"] = data["question"]
        if "query" in data:
            metadata["original_query"] = data["query"]
        if "answer" in data:
            metadata["answer"] = data["answer"]
        if "rewritten_question" in data:
            metadata["rewritten_question"] = data["rewritten_question"]
        current_query = data.get_last_query()
        if current_query:
            metadata["current_query"] = current_query
        context = data.get("context", "")
        if context:
            metadata["context"] = context
        titles = data.get("titles", [])
        paras = data.get("paras", [])
        if titles or paras:
            metadata["supporting_docs"] = {"count": len(titles), "titles": titles, "paras": paras}
        if "metadata" in data:
            metadata["metadata"] = data["metadata"]
        return metadata

    def add_graph_node(self, step_data: dict, state=None) -> str:
        """Add a node with ``step_data`` to the question's log graph and return its id.

        The first node added with a ``state`` also stores the question's metadata.
        """
        node_id = str(len(self.log_graph["nodes"]))
        node = {
            "id": node_id,
            "step_type": step_data["step_type"],
            "participant": step_data["participant"],
            "data": step_data,
        }

        # Store QA metadata at instance level (only once, for the first node)
        if state is not None and self.log_graph["qa_metadata"] is None:
            qa_metadata = self._extract_qa_metadata(state)
            if qa_metadata:
                self.log_graph["qa_metadata"] = qa_metadata

        self.log_graph["nodes"].append(node)
        return node_id

    def add_graph_edge(self, from_node: str, to_node: str, action: str) -> str:
        """Add an edge to the log_graph when transitioning between steps. Returns the generated edge ID."""
        edge_id = str(len(self.log_graph["edges"]))
        self.log_graph["edges"].append({"id": edge_id, "from": from_node, "to": to_node, "action": action})
        return edge_id

    def save_log_graph(self, instance_id: str) -> None:
        """Save the log_graph to consolidated JSON file indexed by qid"""
        graph_path = os.path.join(self.run_dir, f"log_graph__{self.prediction_filename}.json")

        # Load existing consolidated graphs or create empty dict
        consolidated_graphs = {}
        if os.path.exists(graph_path):
            with open(graph_path, "r") as f:
                consolidated_graphs = json.load(f)

        # Add current instance graph
        consolidated_graphs[instance_id] = {
            "nodes": self.log_graph["nodes"],
            "edges": self.log_graph["edges"],
            "qa_metadata": self.log_graph.get("qa_metadata")
        }

        # Save consolidated file
        with open(graph_path, "w") as f:
            json.dump(consolidated_graphs, f, indent=2, default=str)
        self.log(f"[graph] saved log graph for {instance_id} -> {graph_path}")

    def reset_log_graph(self) -> None:
        """Reset the log_graph for a new instance"""
        self.log_graph = {"nodes": [], "edges": [], "qa_metadata": None}

    def close(self):
        if self._log_fp:
            self._log_fp.close()
            self._log_fp = None
