#!/bin/bash
set -x

# nza_32_das_stride2_grayscale model
# MODEL_NAME=nza32_gray_pbn_dtd
# CELL=io
# TRANSFORMER_MODEL=nza_32_das_stride2_grayscale


bazel build  -c opt --copt=-mavx --config=cuda \
  magenta/models/image_stylization:any_image_stylization_transform

TRANSFORMER_MODEL=nza_das

CONTENT_IMAGES_PATHS=/usr/local/google/home/golnazg/opensourcing/data/contents/*.png
STYLE_IMAGES_PATHS=/usr/local/google/home/golnazg/opensourcing/data/styles/*.png
OUTPUT_DIR=/usr/local/google/home/golnazg/opensourcing/data/output_dir/
IMAGE_SIZE=256
STYLE_IMAGE_SIZE=256
CHECKPOINTi=/usr/local/google/home/golnazg/opensourcing/data/model/model.ckpt-2695278

bazel-bin/magenta/models/image_stylization/any_image_stylization_transform \
  --checkpoint="$CHECKPOINTi" \
  --output_dir="$OUTPUT_DIR" \
  --style_images_paths="$STYLE_IMAGES_PATHS" \
  --content_images_paths="$CONTENT_IMAGES_PATHS" \
  --image_size="$IMAGE_SIZE" \
  --style_image_size="$STYLE_IMAGE_SIZE" \
  --transformer_model="$TRANSFORMER_MODEL" \
  --logtostderr
