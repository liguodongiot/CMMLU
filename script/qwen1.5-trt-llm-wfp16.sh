set -x
cd ../src


nohup \
python qwen1.5-trt-llm.py \
    --save_dir ../results/trtllm-Qwen1.5-7B-Chat-kvfp16-0828 \
    --num_few_shot 0 \
> ../results/trtllm-Qwen1.5-7B-Chat-kvfp16-0828.log  2>&1  &
