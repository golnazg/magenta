#!/bin/bash
set -x

# Download PBN dataset
# wget https://github.com/zo7/painter-by-numbers/releases/download/data-v1.0/train.tgz
# wget https://github.com/zo7/painter-by-numbers/releases/download/data-v1.0/test.tgz

# Download DTD dataset
# wget https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz
# tar -xvzf dtd-r1.0.1.tar.gz

PATH_TO_DTD_DATASET=/usr/local/google/home/golnazg/Downloads/dtd/
#STYLE_IMAGES_PATHS=$(awk -v prefix="${PATH_TO_DTD_DATASET}/images/" '{print prefix $0}' "${PATH_TO_DTD_DATASET}/labels/train1.txt")
#STYLE_IMAGES_PATHS=$( echo $STYLE_IMAGES_PATHS | sed 'y/ /,/' )
STYLE_IMAGES_PATHS="${PATH_TO_DTD_DATASET}/images/cobwebbed/*.jpg"

OUTPUT_RECORDIO_PATH=/usr/local/google/home/golnazg/style_transfer/dtd/dtd/train_images/dtd_training_cobwebbed.tfrecord
COMPUTE_GRAM_MATRICES=False

#set vgg path
bazel build  -c opt --copt=-mavx --config=cuda \
  magenta/models/image_stylization/image_stylization_create_dataset

bazel-bin/magenta/models/image_stylization/image_stylization_create_dataset \
    --style_files=$STYLE_IMAGES_PATHS \
    --output_file=$OUTPUT_RECORDIO_PATH \
    --compute_gram_matrices=$COMPUTE_GRAM_MATRICES \
    --logtostderr
