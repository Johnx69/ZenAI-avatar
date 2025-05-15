# -----------------------------------------------------------------------------
# Train textual inversion (TI) on data/itay for token <itay>
# in hyperparameter sweep over num_vectors (1,2,4,8,16),
# saving learned embeddings every 10 epochs.
# -----------------------------------------------------------------------------

# Base settings
MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
TRAIN_DATA_DIR="data/itay"
PLACEHOLDER_TOKEN="<itay>"
INITIALIZER_TOKEN="man"
TRAIN_SCRIPT="diffuser/textual_inversion.py"
OUTPUT_BASE="output_textual_inversion"
NVECTORS=(16)

# Training hyperparams
TOTAL_EPOCHS=300
BLOCK_EPOCHS=10
MIXED_PRECISION="bf16"
RESOLUTION=1024
BATCH_SIZE=4
GRAD_ACC=4
LEARNING_RATE="5.0e-4"

for NV in "${NVECTORS[@]}"; do
  echo "===== NV=${NV} ====="
  OUTPUT_DIR="${OUTPUT_BASE}/nv_${NV}"
  mkdir -p "${OUTPUT_DIR}"

  # no resume on first block
  RESUME_FLAG=""

  for (( START=1; START<=TOTAL_EPOCHS; START+=BLOCK_EPOCHS )); do
    END=$(( START + BLOCK_EPOCHS - 1 ))
    (( END > TOTAL_EPOCHS )) && END=${TOTAL_EPOCHS}
    echo "-- epochs ${START}→${END} --"

    # run training block
    CUDA_VISIBLE_DEVICES=2 accelerate launch "${TRAIN_SCRIPT}" \
      --pretrained_model_name_or_path "${MODEL_NAME}" \
      --train_data_dir           "${TRAIN_DATA_DIR}" \
      --placeholder_token        "${PLACEHOLDER_TOKEN}" \
      --initializer_token        "${INITIALIZER_TOKEN}" \
      --mixed_precision          "${MIXED_PRECISION}" \
      --resolution               "${RESOLUTION}" \
      --train_batch_size         "${BATCH_SIZE}" \
      --gradient_accumulation_steps "${GRAD_ACC}" \
      --learning_rate            "${LEARNING_RATE}" \
      --scale_lr \
      --lr_scheduler             "constant" \
      --lr_warmup_steps          0 \
      --num_vectors              "${NV}" \
      --num_train_epochs         "${BLOCK_EPOCHS}" \
      ${RESUME_FLAG:+--resume_from_checkpoint latest} \
      --output_dir               "${OUTPUT_DIR}" \
      --dataloader_num_workers 4 \
      --enable_xformers_memory_efficient_attention 

    # copy out learned embeddings
    DEST="${OUTPUT_DIR}/epoch_${END}"
    mkdir -p "${DEST}"
    cp "${OUTPUT_DIR}/learned_embeds.safetensors"     "${DEST}/learned_embeds.safetensors"
    cp "${OUTPUT_DIR}/learned_embeds_2.safetensors"   "${DEST}/learned_embeds_2.safetensors"

    # subsequent blocks should resume
    RESUME_FLAG="--resume_from_checkpoint latest"
  done
done

echo "🎉 All TI training complete."
