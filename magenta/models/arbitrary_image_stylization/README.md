# Style Transfer

This directory contains the code for "Exploring the structure of a
real-time, arbitrary neural artistic stylization network" paper which
can transfer style of *arbitrary* style image to arbitrary content image in
*real-time*, in contrast to [limited styles](https://arxiv.org/abs/1610.07629).

[Exploring the structure of a real-time, arbitrary neural artistic stylization network](https://arxiv.org/abs/1705.06830). *Golnaz Ghiasi, Honglak Lee, Manjunath Kudlur, Vincent Dumoulin, Jonathon Shlens*.


# Setup
Set up your [Magenta environment](/README.md).

# Stylizing an Image using a pre-trained model
Download our pre-trained model:

* [Pretrained on PNB and DTD training images](TODO: add the link)


In order to stylize an image according to an arbitrary painting, run the
following command.

```bash
$ arbitrary_image_stylization_with_weights \
  --checkpoint=/path/to/model.ckpt \
  --output_dir=/tmp/arbitrary_image_stylization/output \
  --style_images_paths=/path/to/style_images \
  --content_images_paths=/path/to/content_images \
  --image_size=256 \
  --style_image_size=256
```

In order to stylize an image using the "identity interpolation" technique (see
Figure 8 in paper), run the following command where $INTERPOLATION_WEIGHTS
represents the desired weights for interpolation.

```bash
$ INTERPOLATION_WEIGHTS='[0.0 0.2 0.4 0.6 0.8 1.0 1.5]'
$ arbitrary_image_stylization_with_weights \
  --checkpoint=/path/to/model.ckpt \
  --output_dir=/tmp/arbitrary_image_stylization/interpolation_output \
  --style_images_paths=/path/to/style_images \
  --content_images_paths=/path/to/content_images \
  --image_size=256 \
  --style_image_size=256 \
  --interpolation_weights=$INTERPOLATION_WEIGHTS
```

# Training a Model
To train your own model, you need to have the followings:

1. A directory of images to use as styles. We used [Painter by Number dataset
   (PBN)](https://www.kaggle.com/c/painter-by-numbers) and
   [Describable Textures Dataset (DTD)](https://www.robots.ox.ac.uk/~vgg/data/dtd/).
   [PBN training](https://github.com/zo7/painter-by-numbers/releases/download/data-v1.0/train.tgz)
   [PBN testing](https://github.com/zo7/painter-by-numbers/releases/download/data-v1.0/test.tgz)
   [DTD dataset](https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz)
2. The ImageNet dataset. Instructions for downloading the dataset can be found
   [here](https://github.com/tensorflow/models/tree/master/inception#getting-started).
3. A [trained VGG model checkpoint](http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz).
4. A [trained INCEPTION\_V3 model
   checkpoint](http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz).

First step is to prepare the style images and create a tfrecord file.
Following command can be used for that.
To train and evaluate the model on different set of style images, you need
to prepare different tfrecord for each of them. Eg. use the PBN and DTD
training images to create the training dataset and use a subset of PBN
and DTD testing images for testing dataset.

```bash
$ image_stylization_create_dataset \
    --style_files=/path/to/style/images/*.jpg \
    --output_file=/tmp/arbitrary_image_stylization/style_images.tfrecord
    --compute_gram_matrices=False
```

Then, to train a model:

```bash
logdir=/path/to/logdir
$ arbitrary_image_stylization_train \
      --batch_size=8 \
      --imagenet_data_dir=/path/to/imagenet-2012-tfrecord \
      --vgg_checkpoint=/path/to/vgg-checkpoint \
      --inception_v3_checkpoint=/path/to/inception-v3-checkpoint \
      --style_dataset_file=/tmp/arbitrary_image_stylization/style_images.tfrecord
      --train_dir="$logdir"/train_dir \
```

To run an evalution job while training model on the CPU or another GPU:

```bash
$ arbitrary_image_stylization_evaluate \
      --batch_size=16 \
      --imagenet_data_dir=/path/to/imagenet-2012-tfrecord \
      --eval_style_dataset_file=/tmp/arbitrary_image_stylization/evaluation_style_images.tfrecord \
      --checkpoint_dir=/tmp/arbitrary_image_stylization/train_dir \
      --eval_dir="$logdir"/eval_dir \
      --evaluation_device=/device:CPU:0 \
```

To run tensorboard and see the progress of training:

```bash
$ tensorboard --logdir="$logdir" \
```

