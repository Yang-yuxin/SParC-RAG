// "No CM" ablation of the DPO-tuned Qwen2.5-7B-Instruct on HotpotQA (paper, Context Manager
// decomposition): the Context Manager is replaced by naive_concat_context, which concatenates
// the previous answers and evaluations, and answers are generated with prompts/answer_generator_no_cm.
{
  "start_state": "document_retriever",
  "end_state": "[EOQ]",
  "controller": {
    "fixed_width": 2,
    "fixed_length": -1,
    "max_length": 8
  },
  "models": {
    "document_retriever": {
      "name": "retrieve_and_reset_paragraphs",
      "next_model": "answer_gen",
      "source_corpus_name": "hotpotqa",
      "retriever_host": { "bm25": "http://localhost", "dense": "http://localhost" },
      "retriever_port": { "bm25": 8020, "dense": 8000 },
      "retrieval_count": 6,
      "global_max_num_paras": 15,
      "document_type": "title_paragraph_text",
      "default_auto_strategy": "dense",
      "fixed_retrieval_type": false
    },
    "answer_gen": {
      "name": "answer_gen",
      "next_model": "answer_eval",
      "prompt_file": "prompts/answer_generator_no_cm",
      "max_para_num_words": 300,
      "model_name": "Qwen2.5-7B-Instruct-SPARC-DPO",
      "temperature": 0.5,
      "max_length": 600,
      "model_tokens_limit": 32768,
      "stopword": "[END]",
      "enable_thinking": false
    },
    "answer_eval": {
      "name": "answer_eval",
      "next_model": "naive_concat_context",
      "prompt_file": "prompts/answer_evaluator",
      "always_reject": false,
      "model_name": "Qwen2.5-7B-Instruct-SPARC-DPO",
      "temperature": 0,
      "max_length": 600,
      "model_tokens_limit": 32768,
      "stopword": "[END]",
      "enable_thinking": false
    },
    "naive_concat_context": {
      "name": "naive_concat_context",
      "next_model": "rewrite"
    },
    "rewrite": {
      "name": "rewrite",
      "next_model": "document_retriever",
      "prompt_file": "prompts/query_rewriter",
      "dispatch_prompt_file": "prompts/query_rewriter_dispatch",
      "max_para_num_words": 300,
      "model_name": "Qwen2.5-7B-Instruct-SPARC-DPO",
      "temperature": 0.5,
      "max_length": 600,
      "model_tokens_limit": 32768,
      "stopword": "[END]",
      "enable_thinking": false
    },
    "state_combiner": {
      "name": "state_combiner",
      "prompt_file": "prompts/branch_selector",
      "model_name": "Qwen2.5-7B-Instruct-SPARC-DPO",
      "temperature": 0.5,
      "max_length": 600,
      "model_tokens_limit": 32768,
      "stopword": "[END]",
      "enable_thinking": false
    }
  }
}
