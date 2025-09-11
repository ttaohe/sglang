MODEL_PATH=$1

nohup python -m sglang.launch_server   --model-path  $MODEL_PATH --trust-remote-code \
      --enable-metrics --disable-radix-cache  --host 127.0.0.1 --port 30028 \
      --engine-mode normal \
      --served-model-name  qwen32b --mem-fraction-static 0.9  --tp 4 > "baseline_server.log" 2>&1 &
