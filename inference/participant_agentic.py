"""Participants of the SPARC-RAG controller.

Each participant implements ``query`` and returns the next search state; the Query Rewriter
returns a list of states when it dispatches parallel branches. The controller
(``inference.model_control.ModelController``) calls the participant named by ``state.next``.

Round 1 runs on a single path with the original question:
    document_retriever -> context -> answer_gen -> answer_eval
Every later round starts with one Query Rewriter call that emits W rewritten queries (one per
branch). Each branch then runs retrieval, the Context Manager update, answer generation and
answer evaluation, and the StateCombiner keeps one branch. The search stops when the kept
branch's answer is accepted or when the depth budget is reached.
"""

import json
import random
import re
import sys
import time
from collections import Counter

import requests

from inference.data_instances import (
    AnswerEvaluationStep,
    ContextUpdateStep,
    DispatchCombineStep,
    QuestionAnsweringStep,
    QuestionRewritingStep,
    RetrievalStep,
)
from inference.prompt_readers import read_prompt
from inference.utils import (
    get_history_representation,
    is_para_closely_matching,
    para_to_text,
    remove_wh_words,
)
from models.llm_client_generator import LLMClientGenerator

END_STATE = "[EOQ]"
# Retrieved paragraphs longer than this many words are discarded.
MAX_PARAGRAPH_WORDS = 600
# Seconds to wait for a retriever response before retrying.
RETRIEVER_TIMEOUT = 300


def safe_post_request(url, params, max_retries=10, retry_delay=20):
    """POST ``params`` as JSON, retrying failed requests and non-OK responses.

    Returns the first OK response or the response to the last attempt; raises if the last
    attempt raised.
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            result = requests.post(url, json=params, timeout=RETRIEVER_TIMEOUT)
            if result.ok or attempt == max_retries - 1:
                return result
            print(
                f"WARNING: Post request returned error status {result.status_code}. Will wait "
                f"{retry_delay}s and retry (attempt {attempt + 1}/{max_retries}).",
                file=sys.stderr,
            )
            time.sleep(retry_delay)
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                print(
                    f"WARNING: Post request failed: {e}. Will wait {retry_delay}s and retry "
                    f"(attempt {attempt + 1}/{max_retries}).",
                    file=sys.stderr,
                )
                time.sleep(retry_delay)
    raise Exception(
        f"Post request couldn't succeed after {max_retries} attempts. Last error: {last_error}"
    )


def documents_text(state, max_para_num_words):
    """The state's current paragraphs, formatted for a prompt."""
    return "\n\n".join(
        para_to_text(title, para, max_para_num_words)
        for title, para in zip(state.data["titles"], state.data["paras"])
    )


class ParticipantModel:
    """Base class of the participants.

    ``stats`` counts calls and fallbacks; predict.py logs it at the end of a run.
    """

    def __init__(self):
        self.stats = Counter()

    def get_stats(self) -> dict:
        stats = dict(self.stats)
        generator = getattr(self, "generator", None)
        if generator is not None and generator.num_truncated_prompts:
            stats["truncated_prompts"] = generator.num_truncated_prompts
        return stats


class RetrieveAndResetParagraphsParticipant(ParticipantModel):
    """Retrieve paragraphs for the current query; they replace the state's previous paragraphs.

    The query is the branch's latest rewritten question (the original question before the first
    rewrite), with question words removed. The strategy is the one the Query Rewriter chose for
    the branch; ``default_auto_strategy`` is used before the first rewrite and always when
    ``fixed_retrieval_type`` is true. ``hybrid`` concatenates the BM25 and the dense hits, drops
    duplicates and keeps the first ``retrieval_count``.
    """

    RETRIEVAL_METHODS = {
        "bm25": "retrieve_from_elasticsearch",
        "dense": "vector_based_similarity_search",
    }

    def __init__(
        self,
        retriever_host,
        retriever_port,
        source_corpus_name,
        next_model,
        retrieval_count=6,
        document_type="title_paragraph_text",
        default_auto_strategy="dense",
        fixed_retrieval_type=False,
        global_max_num_paras=15,
    ):
        super().__init__()
        assert default_auto_strategy in (
            "bm25",
            "dense",
            "hybrid",
        ), f"default_auto_strategy {default_auto_strategy} not among the valid choices."
        self.retriever_host = retriever_host
        self.retriever_port = retriever_port
        self.source_corpus_name = source_corpus_name
        self.next_model = next_model
        self.retrieval_count = retrieval_count
        self.document_type = document_type
        self.default_auto_strategy = default_auto_strategy
        self.fixed_retrieval_type = fixed_retrieval_type
        self.global_max_num_paras = global_max_num_paras
        self.retrieval_failures_so_far = 0
        self.retrieval_failures_max = 9

    def reset_failure_counter(self):
        """Reset the retrieval failure counter. Called at the start of each question."""
        self.retrieval_failures_so_far = 0

    def query(self, state):
        query = state.data.get("rewritten_question") or state.data["question"]
        strategy = self.default_auto_strategy
        if not self.fixed_retrieval_type:
            for step in reversed(state.data["inference_seq"]):
                if isinstance(step, QuestionRewritingStep) and step.strategy:
                    strategy = step.strategy.strip().lower()
                    break
        if strategy not in ("bm25", "dense", "hybrid"):
            raise ValueError(
                f"Unknown retrieval strategy {strategy!r}; expected bm25, dense or hybrid."
            )

        clean_query = remove_wh_words(query)
        titles, paras = [], []
        if clean_query.strip():
            if strategy == "hybrid":
                items = self._retrieve_hybrid(clean_query)
            else:
                items = self._retrieve(strategy, clean_query)
            for item in items:
                self._add_item(item, titles, paras)
        if not titles:
            self.stats["empty_retrievals"] += 1
            reason = "" if clean_query.strip() else " (empty after removing question words)"
            print(
                f"WARNING: Retrieval returned no paragraphs for the {strategy} query "
                f"{query!r}{reason}.",
                file=sys.stderr,
            )
        self.stats["calls"] += 1

        new_state = state.copy()
        new_state.data["inference_seq"].append(
            RetrievalStep(titles=titles, paras=paras, participant=state.next, token_size=0)
        )
        new_state.next = self.next_model
        new_state.data["paras"] = paras
        new_state.data["titles"] = titles
        return new_state

    def _url(self, strategy):
        return (
            self.retriever_host[strategy].rstrip("/")
            + ":"
            + str(self.retriever_port[strategy])
            + "/retrieve/"
        )

    def _params(self, strategy, query):
        params = {
            "retrieval_method": self.RETRIEVAL_METHODS[strategy],
            "query_text": query,
            "max_hits_count": self.retrieval_count,
            "corpus_name": self.source_corpus_name,
        }
        if strategy == "bm25":
            params["document_type"] = self.document_type
        return params

    def _hits(self, result):
        return result.json()["retrieval"]

    def _count_failure(self):
        self.retrieval_failures_so_far += 1
        self.stats["failed_requests"] += 1

    def _check_failures(self):
        if self.retrieval_failures_so_far > self.retrieval_failures_max:
            raise Exception(
                f"Retrieval failure exceeded max allowed times ({self.retrieval_failures_so_far} > {self.retrieval_failures_max})"
            )

    def _retrieve(self, strategy, query):
        result = safe_post_request(self._url(strategy), self._params(strategy, query))
        if result.ok:
            return self._hits(result)
        self._count_failure()
        self._check_failures()
        print(
            f"WARNING: Retrieval did not succeed {self.retrieval_failures_so_far} times. Skipping it.",
            file=sys.stderr,
        )
        return []

    def _retrieve_hybrid(self, query):
        responses = {
            strategy: safe_post_request(self._url(strategy), self._params(strategy, query))
            for strategy in ("bm25", "dense")
        }
        merged = []
        for strategy, result in responses.items():
            if result.ok:
                merged.extend(self._hits(result))
            else:
                self._count_failure()
                print(
                    f"WARNING: {'BM25' if strategy == 'bm25' else 'Dense'} branch failed in hybrid.",
                    file=sys.stderr,
                )
        self._check_failures()

        seen = set()
        unique_items = []
        for item in merged:
            key = (item.get("title", "").lower(), item.get("paragraph_text", "")[:200])
            if key not in seen:
                seen.add(key)
                unique_items.append(item)
        return unique_items[: self.retrieval_count]

    def _add_item(self, item, titles, paras):
        if item["corpus_name"] != self.source_corpus_name:
            raise ValueError(
                f"The retrieved corpus name {item['corpus_name']} doesn't match {self.source_corpus_name}."
            )
        paragraph_text = item.get("paragraph_text", "")
        if len(paragraph_text.split(" ")) > MAX_PARAGRAPH_WORDS:
            self.stats["long_paragraphs_discarded"] += 1
            print("WARNING: Discarding a retrieved paragraph as it's excessively long.", file=sys.stderr)
            return
        if is_para_closely_matching(titles, paras, item["title"], paragraph_text):
            return
        if len(paras) >= self.global_max_num_paras:
            return
        titles.append(item["title"])
        paras.append(paragraph_text)


class StateCombinerParticipant(ParticipantModel):
    """Keep one branch once every branch has paused before the Query Rewriter or finished.

    ``select_strategy``:
      "llm"    -- the LLM picks a branch with ``prompt_file`` (default; all main results). An
                  index past the last branch selects the last branch. If the output has no
                  ``<choice>N</choice>``, the first branch that has not finished is kept, else
                  branch 0.
      "first"  -- always branch 0, no LLM call (Context Manager decomposition: update +
                  first-branch selection).
      "random" -- a uniformly random branch, no LLM call (update + random selection). The picks
                  come from one random stream per process (seed 12345), so they depend on the
                  question order.
    The kept branch carries its own note forward; the notes of other branches are dropped.
    """

    SELECT_STRATEGIES = ("llm", "first", "random")

    def __init__(
        self,
        prompt_file="prompts/branch_selector",
        select_strategy="llm",
        stopword=None,
        **kwargs,
    ):
        super().__init__()
        if select_strategy not in self.SELECT_STRATEGIES:
            raise ValueError(
                f"Unknown select_strategy {select_strategy!r}; expected one of {self.SELECT_STRATEGIES}."
            )
        self.prompt = read_prompt(prompt_file) if prompt_file else ""
        self.select_strategy = select_strategy
        self.stopword = stopword
        self._select_rng = random.Random(12345)
        self.generator = LLMClientGenerator(**kwargs)

    def _llm_choose_winner(self, branches):
        """Ask the LLM for the best branch. Returns (index, (reason, token_count, prompt))."""
        question = branches[0].data["question"]
        blocks = []
        for i, branch in enumerate(branches):
            summary = get_history_representation(
                branch,
                use_rewritten_question=True,
                use_generated_answer=True,
                use_answer_evaluation=True,
                use_context=True,
            )
            blocks.append(f"### Branch {i}\n" + summary)
        prompt_filled = self.prompt.format(
            query=question, reasonings="\n\n".join(blocks)
        ).strip()

        output_text_scores = self.generator.generate_text_sequence(prompt_filled, self.stopword)
        self.stats["llm_calls"] += 1
        text = output_text_scores[0][0][0].strip()
        token_count = output_text_scores[0][0][1]

        reasoning = ""
        reason_match = re.search(r"<reason>\s*(.*?)\s*</reason>", text, re.DOTALL)
        if reason_match:
            reasoning = reason_match.group(1).strip()

        choice_match = re.search(r"<choice>\s*(\d+)\s*</choice>", text)
        if choice_match:
            idx = int(choice_match.group(1))
            if idx >= len(branches):
                self.stats["selection_index_clamped"] += 1
                idx = len(branches) - 1
            return idx, (reasoning if reasoning else text, token_count, prompt_filled)
        print(
            "WARNING: No <choice> in the branch selector's output; keeping the fallback branch.",
            file=sys.stderr,
        )

        self.stats["selection_fallbacks"] += 1
        for i, branch in enumerate(branches):
            if branch.next != END_STATE:
                return i, ("[Fallback: non-EOQ branch found]", token_count, prompt_filled)
        return 0, ("[Fallback: all branches at EOQ; keeping branch 0]", token_count, prompt_filled)

    def query(self, states):
        if self.select_strategy == "first":
            idx = 0
            reason = "[select_strategy=first: deterministic branch 0]"
            token_count = 0
            prompt_filled = ""
        elif self.select_strategy == "random":
            idx = self._select_rng.randrange(len(states))
            reason = "[select_strategy=random: uniform pick]"
            token_count = 0
            prompt_filled = ""
        else:
            idx, (reason, token_count, prompt_filled) = self._llm_choose_winner(states)
        self.stats["calls"] += 1

        next_state = states[idx].copy()
        next_state.data["inference_seq"].append(
            DispatchCombineStep(
                choices=[s.data.get_last_answer(reasoning=True) for s in states],
                combined_answer=next_state.data.get_last_answer(),
                reason=reason,
                participant="state_combiner",
                token_size=token_count,
                prompt=prompt_filled,
            )
        )
        return next_state


class AnswerGenerationParticipant(ParticipantModel):
    """Generate an answer to the original question.

    The prompt may use {query} (original question), {note} (the Context Manager's note),
    {refs} (the paragraphs of the latest retrieval) and {history} (earlier rewritten queries and
    answers). If the output contains ``<answer>...</answer>``, the tagged text is the answer;
    otherwise the whole output is.
    """

    def __init__(
        self,
        prompt_file,
        next_model,
        max_para_num_words=300,
        stopword=None,
        **kwargs,
    ):
        super().__init__()
        self.prompt = read_prompt(prompt_file)
        self.next_model = next_model
        self.max_para_num_words = max_para_num_words
        self.stopword = stopword
        self.generator = LLMClientGenerator(**kwargs)

    def query(self, state):
        new_state = state.copy()
        refs = documents_text(state, self.max_para_num_words)
        history = get_history_representation(
            state,
            use_retrieved_documents=False,
            use_rewritten_question=True,
            use_generated_answer=True,
            use_context=True,
        )
        prompt_filled = self.prompt.format(
            query=state.data["question"],
            note=state.data.get("context", refs),
            refs=refs,
            history=history,
        )
        output_text_scores = self.generator.generate_text_sequence(
            prompt_filled, self.stopword
        )
        output = output_text_scores[0][0][0].strip()
        token_count = output_text_scores[0][0][1]
        self.stats["llm_calls"] += 1

        answer_match = re.search(r"<answer>(.*?)(</answer>|$)", output, re.DOTALL)
        if answer_match:
            final_answer = answer_match.group(1).strip()
        else:
            if "<answer>" in self.prompt:
                self.stats["answers_without_tags"] += 1
            final_answer = output
        new_state.data.add_answer(
            QuestionAnsweringStep(
                answer=final_answer,
                reasoning=output,
                participant=state.next,
                token_size=token_count,
                prompt=prompt_filled,
            )
        )
        new_state.next = self.next_model
        return new_state


def parse_verdict(output):
    """Read the verdict of an Answer-Evaluator output: "yes" (STOP) or "no" (CONTINUE).

    The first of the first two lines that starts with yes/accept ("yes") or no/reject ("no")
    decides. If there is none, the verdict is "yes" when the first three characters contain
    "yes" or "acc", else "no". Returns (verdict, whether one of the two lines decided).
    training/gen_dpo_data_answer_eval.py and training/train_dpo_asymmetric.py read verdicts with
    this function too.
    """
    for line in output.split("\n")[:2]:
        line = line.strip().lower()
        if line.startswith(("yes", "accept")):
            return "yes", True
        if line.startswith(("no", "reject")):
            return "no", True
    first_chars = output[:3].lower()
    return ("yes" if ("yes" in first_chars or "acc" in first_chars) else "no"), False


class AnswerEvaluatorParticipant(ParticipantModel):
    """Decide whether to accept the latest answer (STOP) or to run another round (CONTINUE).

    The prompt may use {query} (original question), {answer} (the answer generator's full
    output) and {context} (the Context Manager's note). The verdict is read with
    ``parse_verdict``. With ``always_reject`` (fixed depth) the LLM is not called and every
    answer is rejected; otherwise ``prompt_file`` is required.
    """

    def __init__(
        self,
        next_model,
        prompt_file="",
        end_state=END_STATE,
        stopword=None,
        always_reject=False,
        **kwargs,
    ):
        super().__init__()
        if not always_reject and not prompt_file:
            raise ValueError(
                "answer_eval needs a prompt_file unless always_reject is true. The base "
                "configurations have none: they run at a fixed depth only."
            )
        self.prompt = read_prompt(prompt_file) if prompt_file else ""
        self.next_model = next_model
        self.end_state = end_state
        self.stopword = stopword
        self.always_reject = always_reject
        self.generator = LLMClientGenerator(**kwargs)

    def _parse_verdict(self, output):
        verdict, decided_by_line = parse_verdict(output)
        if not decided_by_line:
            self.stats["unparsed_verdicts"] += 1
            print(
                "WARNING: No yes/no in the first two lines of the answer evaluation; "
                f"read {verdict!r} from its first characters.",
                file=sys.stderr,
            )
        return verdict

    def query(self, state):
        new_state = state.copy()
        generated_answer = state.data.get_last_answer()
        _, answer_with_reasoning = state.data.get_last_answer(reasoning=True)
        context = state.data.get("context", "")
        # Stored in the step even without an LLM call; training/gen_dpo_data_answer_eval.py reads it.
        prompt_filled = self.prompt.format(
            query=state.data["question"], answer=answer_with_reasoning, context=context
        )

        if self.always_reject:
            evaluation = "no"
            reasoning = "Auto-rejected (always_reject=True)"
            token_count = 0
            self.stats["auto_rejected"] += 1
        else:
            output_text_scores = self.generator.generate_text_sequence(
                prompt_filled, self.stopword
            )
            token_count = output_text_scores[0][0][1]
            reasoning = output_text_scores[0][0][0].strip()
            evaluation = self._parse_verdict(reasoning)
            self.stats["llm_calls"] += 1

        new_state.data["inference_seq"].append(
            AnswerEvaluationStep(
                answer=generated_answer,
                evaluation=evaluation,
                reasoning=reasoning,
                participant=state.next,
                token_size=token_count,
                prompt=prompt_filled,
                context=context,
            )
        )
        new_state.next = self.end_state if evaluation == "yes" else self.next_model
        return new_state


def normalize_strategy(text, default_strategy):
    """Map the model's strategy text to bm25, dense or hybrid (else ``default_strategy``)."""
    if not text:
        return default_strategy
    s_norm = text.strip().lower()
    if s_norm in {"bm25", "dense", "hybrid"}:
        return s_norm
    if s_norm in {"sparse", "lexical", "bm-25", "bm 25"}:
        return "bm25"
    if s_norm in {"semantic", "vector", "dense-retrieval", "dense retrieval"}:
        return "dense"
    return default_strategy


def clean_query(text, max_words):
    """Replace control characters and keep at most ``max_words`` words."""
    text = re.sub(r"[\x00-\x1F\x7F]", " ", text)
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).strip()


def parse_dispatch_output(output, width, default_strategy="dense", max_rewrite_words=100):
    """Read the queries of a dispatch output: ``<item rank="i">`` elements inside ``<queries>``.

    Returns (queries, num_items). ``queries`` holds a (strategy, query) pair for each of the
    first ``width`` items that has both a <strategy> and a <query> tag, with the strategy
    normalized by ``normalize_strategy`` and the query cleaned by ``clean_query``.
    ``num_items`` counts all items (0 without ``<queries>``).
    training/gen_dpo_data_query_rewrite.py reads its candidates with this function too.
    """
    queries_match = re.search(r"<queries[^>]*>(.*?)</queries>", output, re.DOTALL | re.IGNORECASE)
    if not queries_match:
        return [], 0
    items = re.findall(
        r'<item[^>]*rank="(\d+)"[^>]*>(.*?)</item>', queries_match.group(1), re.DOTALL
    )
    queries = []
    for _, item_content in items[:width]:
        strategy_match = re.search(r"<strategy>(.*?)</strategy>", item_content, re.DOTALL)
        query_match = re.search(r"<query>(.*?)</query>", item_content, re.DOTALL)
        if strategy_match and query_match:
            queries.append(
                (
                    normalize_strategy(strategy_match.group(1).strip(), default_strategy),
                    clean_query(query_match.group(1).strip(), max_rewrite_words),
                )
            )
    return queries, len(items)


class QueryRewriteParticipant(ParticipantModel):
    """Rewrite the question for the next retrieval and choose bm25, dense or hybrid for it.

    With ``dispatch_size`` W > 1 (the controller's fixed width), one call with
    ``dispatch_prompt_file`` should return W items ``<item rank="i"><strategy>...</strategy>
    <query>...</query></item>`` inside ``<queries>``; each of the first W items that has both
    tags becomes a branch (``parse_dispatch_output``). If no item is usable, the search
    continues on one branch with the original question. With W = 1, ``prompt_file`` produces a
    single rewrite.

    Prompts may use {query} (original question), {current_query} (latest rewrite), {context}
    (the note), {refs} (latest paragraphs), {history} (earlier rewrites, answers and
    evaluations) and, in the dispatch prompt, {N} (= W).
    """

    HISTORY_ARGS = dict(
        use_retrieved_documents=False,
        use_rewritten_question=True,
        use_generated_answer=True,
        use_answer_evaluation=True,
        use_context=True,
        use_last_only=False,
    )

    def __init__(
        self,
        next_model,
        prompt_file="",
        dispatch_prompt_file="",
        max_para_num_words=300,
        stopword=None,
        default_strategy="dense",
        max_rewrite_words=100,
        **kwargs,
    ):
        super().__init__()
        self.prompt = read_prompt(prompt_file) if prompt_file else ""
        self.dispatch_prompt = read_prompt(dispatch_prompt_file) if dispatch_prompt_file else None
        self.next_model = next_model
        self.max_para_num_words = max_para_num_words
        self.stopword = stopword
        self.default_strategy = default_strategy
        self.max_rewrite_words = max_rewrite_words
        self.generator = LLMClientGenerator(**kwargs)

    @staticmethod
    def _extract_last_tag(text, tag):
        """Return the content of the last <tag>...</tag> in ``text``, or None."""
        matches = re.findall(rf"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL | re.IGNORECASE)
        if not matches:
            return None
        return matches[-1].strip()

    def query(self, state, dispatch_size=None):
        new_state = state.copy()
        question = state.data["question"]
        values = dict(
            query=question,
            current_query=state.data.get("rewritten_question") or question,
            refs=documents_text(state, self.max_para_num_words),
            context=state.data.get("context", ""),
            history=get_history_representation(state, **self.HISTORY_ARGS),
        )
        if dispatch_size is not None and dispatch_size > 1:
            return self._rewrite_branches(state, new_state, values, dispatch_size)
        return self._rewrite_single(state, new_state, values)

    def _rewrite_single(self, state, new_state, values):
        question = values["query"]
        prompt_filled = self.prompt.format(**values)
        output_text_scores = self.generator.generate_text_sequence(
            prompt_filled, self.stopword
        )
        token_count = output_text_scores[0][0][1]
        output = output_text_scores[0][0][0].strip()
        self.stats["llm_calls"] += 1

        analysis = self._extract_last_tag(output, "think") or output
        rewritten_query = self._extract_last_tag(output, "query")
        strategy_raw = self._extract_last_tag(output, "strategy")
        if not strategy_raw:
            m = re.search(r"(?im)^\s*(bm25|dense|hybrid)\s*$", output)
            strategy_raw = m.group(1) if m else None
        retrieval_strategy = normalize_strategy(strategy_raw, self.default_strategy)

        if not rewritten_query:
            # Use the last non-empty output line unless it is a bare tag; else the question.
            lines = [ln.strip() for ln in output.splitlines() if ln.strip()]
            heuristic = lines[-1] if lines else ""
            if heuristic.startswith("<") and heuristic.endswith(">"):
                heuristic = ""
            if heuristic:
                rewritten_query = heuristic
                self.stats["query_missing_tags_heuristic_used"] += 1
            else:
                rewritten_query = question
                self.stats["query_missing_tags_fallback_to_original"] += 1
        rewritten_query = clean_query(rewritten_query, self.max_rewrite_words)

        new_state.data["rewritten_question"] = rewritten_query
        new_state.data["inference_seq"].append(
            QuestionRewritingStep(
                input_question=json.dumps(question),
                output_question=rewritten_query,
                reasoning=analysis,
                strategy=retrieval_strategy,
                participant=state.next,
                token_size=token_count,
                prompt=prompt_filled,
            )
        )
        new_state.next = self.next_model
        return new_state

    def _rewrite_branches(self, state, new_state, values, dispatch_size):
        question = values["query"]
        prompt_template = self.dispatch_prompt.replace("{N}", str(dispatch_size))
        prompt_filled = prompt_template.format(**values)
        output_text_scores = self.generator.generate_text_sequence(prompt_filled, self.stopword)
        output = output_text_scores[0][0][0].strip()
        token_count = output_text_scores[0][0][1]
        self.stats["llm_calls"] += 1

        queries, num_items = parse_dispatch_output(
            output, dispatch_size, self.default_strategy, self.max_rewrite_words
        )
        num_skipped = min(num_items, dispatch_size) - len(queries)
        if num_skipped:
            self.stats["rewrite_items_skipped"] += num_skipped
        if queries:
            analysis = self._extract_last_tag(output, "think") or output
            branches = []
            for strategy, rewritten_query in queries:
                branch_state = new_state.copy()
                branch_state.data["rewritten_question"] = rewritten_query
                branch_state.data["inference_seq"].append(
                    QuestionRewritingStep(
                        input_question=json.dumps(question),
                        output_question=rewritten_query,
                        reasoning=analysis,
                        strategy=strategy,
                        participant=state.next,
                        # The call's completion tokens are split among all parsed items.
                        token_size=token_count // num_items,
                        prompt=prompt_filled,
                        dispatch_output=output,
                        dispatch_token_size=token_count,
                    )
                )
                branch_state.next = self.next_model
                branches.append(branch_state)
            if len(branches) < dispatch_size:
                self.stats["rewrite_fewer_branches"] += 1
            return branches
        if num_items:
            print(
                "WARNING: Parsed <queries> but got no valid items, falling back to original query",
                file=sys.stderr,
            )

        print(
            "WARNING: Failed to parse dispatch queries, falling back to original query with dense retrieval",
            file=sys.stderr,
        )
        self.stats["rewrite_fallbacks"] += 1
        new_state.data["rewritten_question"] = question
        new_state.data["inference_seq"].append(
            QuestionRewritingStep(
                input_question=json.dumps(question),
                output_question=question,
                reasoning=output,
                strategy=self.default_strategy,
                participant=state.next,
                token_size=token_count,
                prompt=prompt_filled,
                dispatch_output=output,
            )
        )
        new_state.next = self.next_model
        return new_state


class ContextManagerParticipant(ParticipantModel):
    """Context Manager: rewrite the branch's note with the branch's newest information.

    The prompt receives {query} (original question), {note} (the current note; initially "The
    initial query is: <question>") and {new_context}: the branch's latest rewritten query and
    retrieved paragraphs, and the answer and evaluation of its previous round. The output
    replaces the note.
    """

    def __init__(self, prompt_file, next_model, stopword=None, **kwargs):
        super().__init__()
        self.prompt = read_prompt(prompt_file)
        self.next_model = next_model
        self.stopword = stopword
        self.generator = LLMClientGenerator(**kwargs)

    @staticmethod
    def init_note(initial_query):
        return "The initial query is: " + initial_query + "\n"

    def query(self, state):
        new_state = state.copy()
        question = state.data["question"]
        new_context = get_history_representation(
            state,
            use_retrieved_documents=True,
            use_rewritten_question=True,
            use_generated_answer=True,
            use_answer_evaluation=True,
            use_context=False,
            use_last_only=True,
        )
        prompt = self.prompt.format(
            query=question,
            note=state.data.get("context") or self.init_note(question),
            new_context=new_context,
        )
        output_text_scores = self.generator.generate_text_sequence(prompt, self.stopword)
        self.stats["llm_calls"] += 1
        output = output_text_scores[0][0][0].strip()
        token_count = output_text_scores[0][0][1]

        new_state.data["context"] = output
        new_state.data["inference_seq"].append(
            ContextUpdateStep(
                participant=state.next,
                old_context=state.data.get("context"),
                new_context=output,
                token_size=token_count,
                prompt=prompt,
            )
        )
        new_state.next = self.next_model
        return new_state


class NaiveConcatContextParticipant(ParticipantModel):
    """No-CM ablation: replaces the Context Manager without calling an LLM.

    The note becomes the concatenation of all previous generated answers (each with the answer
    generator's full output) and answer evaluations of the path. Retrieved paragraphs are not
    added. No inference step is recorded.
    """

    def __init__(self, next_model):
        super().__init__()
        self.next_model = next_model

    def query(self, state):
        new_state = state.copy()
        context = get_history_representation(
            state,
            use_generated_answer=True,
            use_answer_evaluation=True,
            use_retrieved_documents=False,
            use_rewritten_question=False,
            use_context=False,
            use_last_only=False,
            wrapped_in=None,
        )
        new_state.data["context"] = context.strip()
        self.stats["calls"] += 1
        new_state.next = self.next_model
        return new_state
