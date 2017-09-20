#!/bin/bash
set -x

bazel build  -c opt --copt=-mavx --config=cuda \
  magenta/models/any_image_stylization:interpolation_with_identity

CONTENT_IMAGES_PATHS=/cns/io-d/home/golnazg/art_embedding_keep/datasets/s_content_images/*.png
STYLE_IMAGES_PATHS=/cns/io-d/home/golnazg/art_embedding_keep/datasets/s_style_images/*.png

CONTENT_IMAGES_PATHS=/usr/local/google/home/golnazg/opensourcing/data/s_content_images/*.png
STYLE_IMAGES_PATHS=/usr/local/google/home/golnazg/opensourcing/data/s_style_images/*.png
OUTPUT_DIR=/usr/local/google/home/golnazg/opensourcing/data/output_dir_interpolation2/
IMAGE_SIZE=256
STYLE_IMAGE_SIZE=256
CHECKPOINT=/usr/local/google/home/golnazg/opensourcing/data/model/model.ckpt-2695278

bazel-bin/magenta/models/any_image_stylization/interpolation_with_identity \
  --checkpoint="$CHECKPOINT" \
  --output_dir="$OUTPUT_DIR" \
  --style_images_paths="$STYLE_IMAGES_PATHS" \
  --content_images_paths="$CONTENT_IMAGES_PATHS" \
  --image_size="$IMAGE_SIZE" \
  --style_image_size="$STYLE_IMAGE_SIZE" \
  --logtostderr
