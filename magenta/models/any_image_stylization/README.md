# Style Transfer

Style transfer is the task of producing a pastiche image 'p' that shares the
content of a content image 'c' and the style of a style image 's'. This code
implements the paper "A Learned Representation for Artistic Style":

[Exploring the structure of a real-time, arbitrary neural artistic stylization network](https://arxiv.org/abs/1705.06830). *Golnaz Ghiasi, Honglak Lee, Manjunath Kudlur, Vincent Dumoulin, Jonathon Shlens*.

# Setup
Whether you want to stylize an image with one of our pre-trained models or train your own model, you need to set up your [Magenta environment](/README.md).

# Stylizing an Image
First, download one of our pre-trained models:

* [Pretrained on PNB and DTD training images](add the link)

(You can also train your own model, but if you're just getting started we recommend using a pre-trained model first.)

Then, run the following command:

```bash
$ any_image_stylization_transform \
  --checkpoint=/path/to/model.ckpt \
  --output_dir=/tmp/any_image_stylization/output \
  --style_images_paths=/path/to/style_images \
  --content_images_paths=/path/to/content_images \
  --image_size=256 \
  --style_image_size=256 \
```


Intorpolation with identity

```bash
$ interpolation_with_identity \
  --checkpoint=/path/to/model.ckpt \
  --output_dir=/tmp/any_image_stylization/interpolation_output \
  --style_images_paths=/path/to/style_images \
  --content_images_paths=/path/to/content_images \
  --image_size=256 \
  --style_image_size=256 \
```

# Training a Model
To train your own model, you'll need three things:

1. A directory of images to use as styles. We used [Painter by Number Dataset
   (PBN)]() and [(DTD) dataset]().
    [PBN training](https://github.com/zo7/painter-by-numbers/releases/download/data-v1.0/train.tgz)
    [PBN testing](https://github.com/zo7/painter-by-numbers/releases/download/data-v1.0/test.tgz)
    [DTD dataset](https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz)
2. A [trained VGG model checkpoint](http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz).
3. The ImageNet dataset. Instructions for downloading the dataset can be found [here](https://github.com/tensorflow/models/tree/master/inception#getting-started).

First, you need to prepare your style images:
To train your model on Painting By Number training images
mkdir $PATH/dataset
cd $PATH/dataset
wget https://github.com/zo7/painter-by-numbers/releases/download/data-v1.0/train.tgz
tar -xvzf train.tgz

```bash
$ image_stylization_create_dataset \
    --style_files=/path/to/style/images/*.jpg \
    --output_file=/tmp/any_image_stylization/style_images.tfrecord
    --compute_gram_matrices=False \
```

Then, to train a model:

```bash
$ any_image_stylization_train \
      --batch_size=8 \
      --style_dataset_file=/tmp/any_image_stylization/style_images.tfrecord
      --train_dir=/tmp/any_image_stylization/train_dir \
```

To evaluate the model:

```bash
$ eany_image_stylization_evaluate \
      --batch_size=16 \
      --eval_style_dataset_file=/tmp/any_image_stylization/evaluation_style_images.tfrecord \
      --checkpoint_dir=/tmp/any_image_stylization/train_dir \
      --eval_dir=/tmp/any_image_stylization/eval_dir \
```

