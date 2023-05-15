#!/bin/bash

# This script is used for audio tagging with a pre-trained model.
# Usage: ./inference.sh <audiofile>

CHECKPOINT_PATH="/networks/Cnn14_mAP=0.431.pth"
MODEL_TYPE="Cnn14"
AUDIO_FILE=$1

if [ -z "$AUDIO_FILE" ]
then
      echo "No audio file provided. Usage: ./inference.sh <audiofile>"
      exit 1
fi

export CUDA_VISIBLE_DEVICES=0 

python3 pytorch/inference.py audio_tagging \
    --model_type=$MODEL_TYPE \
    --checkpoint_path=$CHECKPOINT_PATH \
    --audio_path="$AUDIO_FILE" \
    --cuda