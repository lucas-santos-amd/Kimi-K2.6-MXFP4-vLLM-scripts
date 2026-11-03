MODEL_PATH="${1:-/data/models/Kimi-K2.6-MXFP4}"
lm_eval \
    --model local-completions \
    --model_args model=${MODEL_PATH},base_url=http://localhost:8002/v1/completions,num_concurrent=64,max_retries=3,tokenized_requests=False,trust_remote_code=True \
    --tasks gsm8k \
    --num_fewshot 5 \
	
	
lm_eval \
    --model local-chat-completions \
    --model_args model=${MODEL_PATH},base_url=http://localhost:8002/v1/chat/completions,num_concurrent=64,max_retries=3,tokenized_requests=False,trust_remote_code=True \
    --tasks gpqa_diamond_cot_zeroshot \
    --apply_chat_template \
    --gen_kwargs max_gen_toks=16384
