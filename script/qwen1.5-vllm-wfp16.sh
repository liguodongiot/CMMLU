set -x
cd ../src

python qwen1.5-vllm.py \
    --model_name_or_path /workspace/models/Qwen1.5-7B-Chat \
    --save_dir ../results/Qwen1.5-7B-Chat-kvfp16 \
    --num_few_shot 0






