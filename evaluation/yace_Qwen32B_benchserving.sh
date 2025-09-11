MODEL_PATH=$1 

python3 -m sglang.bench_serving --backend sglang  --port 30028 --host 127.0.0.1  \
            --model  $MODEL_PATH  --dataset-path /hetero_infer/lanyi.ht/dev/exp/data/ShareGPT_V3_unfiltered_cleaned_split/ShareGPT_V3_unfiltered_cleaned_split.json \
            --dataset-name random --random-input 4096 --random-output 1536 --random-range-ratio 1.0 \
            --num-prompt 100 