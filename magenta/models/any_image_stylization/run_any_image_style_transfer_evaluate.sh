#!/bin/bash
set -e

bazel build  -c opt --copt=-mavx --config=cuda \
  magenta/models/any_image_stylization/any_image_stylization_evaluate

bazel build  -c opt --copt=-mavx \
  magenta/models/any_image_stylization/any_image_stylization_evaluate

# Run training:
TRAIN_DIR=<TRAIN_DIR>
TRAIN_DIR=/tmp/train_dir_cobwebbed_a
EVAL_DIR="$TRAIN_DIR"/eval
EVAL_RECORDIO_PATH=/usr/local/google/home/golnazg/style_transfer/dtd/dtd/train_images/dtd_training_cobwebbed.tfrecord

bazel-bin/magenta/models/any_image_stylization/any_image_stylization_evaluate \
      --batch_size=16 \
      --eval_style_dataset_file=$EVAL_RECORDIO_PATH \
      --checkpoint_dir=$TRAIN_DIR \
      --eval_dir=$EVAL_DIR \
      --alsologtostderr
