#!/bin/bash

# Global configuration variables
export TORCH_SHOW_CPP_STACKTRACES=1
export TORCHELASTIC_ERROR_FILE=error.log
export ATTN_LAYER=24
export TARGET_TOKEN_ID=","
export MAX_INFERENCES=2
export BATCH_SIZE=1
export MODEL_PATH="Qwen/Qwen3-VL-8B-Instruct"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Dataset paths
export SCREENSPOT_PRO_BASE_DIR="./data/screenspot-pro"  

# Install dependencies
pip install transformers==4.57.1

# Run ScreenSpot-Pro experiment
echo "Running ScreenSpot-Pro..."
torchrun --nproc_per_node=1 mvp_sspro_qwen3vl.py \
    --attn_layer $ATTN_LAYER \
    --target_token_id "$TARGET_TOKEN_ID" \
    --max_inferences $MAX_INFERENCES \
    --json_file_dir "$SCREENSPOT_PRO_BASE_DIR/annotations" \
    --base_image_dir "$SCREENSPOT_PRO_BASE_DIR/images" \
    --model_path "$MODEL_PATH" \
    --batch_size $BATCH_SIZE

echo "All experiments completed!"