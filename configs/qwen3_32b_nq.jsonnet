// SPARC-RAG with Qwen3-32B on Natural Questions (paper, main results): fixed depth 8, width 2.
{
  "start_state": "document_retriever",
  "end_state": "[EOQ]",
  "controller": {
    "fixed_width": 2,
    "fixed_length": 8
  },
  "models": {
    "document_retriever": {
      "name": "retrieve_and_reset_paragraphs",
      "next_model": "context",
      "source_corpus_name": "wiki",
      "retriever_host": { "bm25": "http://localhost", "dense": "http://localhost" },
      "retriever_port": { "bm25": 8020, "dense": 8000 },
      "retrieval_count": 6,
      "global_max_num_paras": 15,
      "document_type": "title_paragraph_text",
      "default_auto_strategy": "dense",
      "fixed_retrieval_type": false
    },
    "context": {
      "name": "context",
      "next_model": "answer_gen",
      "prompt_file": "prompts/context_manager",
      "model_name": "Qwen/Qwen3-32B",
      "temperature": 0.5,
      "max_length": 600,
      "model_tokens_limit": 2000,
      "stopword": "[END]",
      "enable_thinking": false
    },
    "answer_gen": {
      "name": "answer_gen",
      "next_model": "answer_eval",
      "prompt_file": "prompts/answer_generator",
      "max_para_num_words": 300,
      "model_name": "Qwen/Qwen3-32B",
      "temperature": 0.5,
      "max_length": 600,
      "model_tokens_limit": 32768,
      "stopword": "[END]",
      "enable_thinking": false
    },
    "answer_eval": {
      "name": "answer_eval",
      "next_model": "rewrite",
      "always_reject": true,
      "model_name": "Qwen/Qwen3-32B",
      "temperature": 0,
      "max_length": 600,
      "model_tokens_limit": 32768,
      "stopword": "[END]",
      "enable_thinking": false
    },
    "rewrite": {
      "name": "rewrite",
      "next_model": "document_retriever",
      "prompt_file": "prompts/query_rewriter",
      "dispatch_prompt_file": "prompts/query_rewriter_dispatch",
      "max_para_num_words": 300,
      "model_name": "Qwen/Qwen3-32B",
      "temperature": 0.5,
      "max_length": 600,
      "model_tokens_limit": 32768,
      "stopword": "[END]",
      "enable_thinking": false
    },
    "state_combiner": {
      "name": "state_combiner",
      "prompt_file": "prompts/branch_selector",
      "model_name": "Qwen/Qwen3-32B",
      "temperature": 0.5,
      "max_length": 600,
      "model_tokens_limit": 32768,
      "stopword": "[END]",
      "enable_thinking": false
    }
  }
}
