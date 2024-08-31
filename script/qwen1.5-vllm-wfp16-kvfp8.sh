set -x
cd ../src

nohup \
python qwen1.5-vllm.py \
    --save_dir ../results/vllm-Qwen1.5-7B-Chat-kvfp8-0828 \
    --num_few_shot 0 \
> ../results/vllm-Qwen1.5-7B-Chat-kvfp8-0828.log  2>&1  &


