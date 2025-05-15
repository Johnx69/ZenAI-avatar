if [ "$#" -lt 1 ]; then
    echo "Usage: $0 [lora] [db_lora] [ti]"
    exit 1
fi
CUDA_VISIBLE_DEVICES=4 python3 evaluate_metrics.py --methods "$@"