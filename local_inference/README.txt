Local development inference: Llama-3.1-8B-Instruct

Built with Llama. Model card and license:
https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

The user explicitly requested local 8B/3B inference for the judge-error study.
This local setup is separate from the staff's model server and official grader.
TABot continues to receive its endpoint and model through config.

On this machine, the complete 8B checkpoint is already cached in Hugging Face's
standard hub cache, revision 0e9e39f249a16976918f6564b8830bc894c89659.
The existing /data/mfp5696/.venvs/vrga-official-judge-pinned/bin/python environment
has vLLM 0.19.1, PyTorch 2.10.0 (CUDA 12.8), and Transformers 5.6.0.
No weights or environment packages are committed to this repository.

Start the model with that environment active:

  A3_MODEL_PATH="$HOME/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    python -m vllm.entrypoints.openai.api_server \
    --model "$A3_MODEL_PATH" \
    --served-model-name meta-llama/Llama-3.1-8B-Instruct \
    --host 127.0.0.1 --port 8000 --dtype bfloat16 \
    --max-model-len 16384 --gpu-memory-utilization 0.5 --max-num-seqs 4 \
    --enforce-eager --seed 42 --generation-config vllm

The current server was started using the cached snapshot's absolute path.
PID, exact command and startup log are in /tmp/ece6256-local-8b/.
It binds only to this machine and uses GPU 0. It does not occupy GPUs 1 and 2.

Check the bot on the real local model:

  python eval_dev.py --endpoint http://127.0.0.1:8000/v1 \
    --model meta-llama/Llama-3.1-8B-Instruct

Capture a local judge experiment:

  python local_inference/run_judge_study.py \
    --endpoint http://127.0.0.1:8000/v1 \
    --model meta-llama/Llama-3.1-8B-Instruct --output /tmp/new-local-judge-study

The completed study is archived in report/bonus/local_judge_run/. To verify its
checksums and regenerate its documented LoRA counterexample:

  python local_inference/summarize_judge_study.py

The repo's ignored .env file sets A3_ENDPOINT and A3_MODEL for convenience:

  source .env
  python eval_dev.py --endpoint "$A3_ENDPOINT" --model "$A3_MODEL"

The study fixes the prompt and public question set before judging. It captures
unaltered bot answers and all HTTP requests/responses, then presents the actual
retrieved evidence to a local self-judge. It never reconstructs the hidden grader
prompt. A factual review is still required to substantiate any claimed mistake;
local judge scores are not official course scores or a guarantee of bonus credit.
