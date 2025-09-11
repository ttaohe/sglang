MODEL_PATH=$1

nohup python -m sglang.launch_server   --model-path  $MODEL_PATH --trust-remote-code \
      --disable-radix-cache  --host 127.0.0.1 --port 30028 \
      --engine-mode semipd --decode-log-interval 20 \
      --served-model-name  Qwen32B --mem-fraction-static 0.85  --tp 4 > "semipdgtx_server.log" 2>&1 &