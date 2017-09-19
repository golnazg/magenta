#!/bin/bash
set -e

bazel build  -c opt --copt=-mavx --config=cuda \
  magenta/models/image_stylization/any_image_stylization_train

# Run training:
train_dir=<TRAIN_DIR>
TRAIN_DIR=/tmp/train_dir
RECORDIO_PATH=/usr/local/google/home/golnazg/style_transfer/dtd/dtd/train_images/dtd_training_paisley.tfrecord

bazel-bin/magenta/models/image_stylization/any_image_stylization_train \
      --batch_size=8 \
      --style_dataset_file=$RECORDIO_PATH \
      --train_dir=$TRAIN_DIR \
      --alsologtostderr
