"""Generates stylized images given an content and style images.

Given paths to a set of content images and paths to a set of style images, this
script computes stylized images for each pair of the content and style images
and saves them to the given output_dir. It also computes content and style
losses and saves them to .csv and numpy npz files.
See run_transform.sh for example usage.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ast
import os

import tensorflow as tf
import numpy as np
#from google3.pyglib import gfile

from magenta.models.any_image_stylization import any_image_stylization_build_model as build_model
from magenta.models.image_stylization import image_utils

slim = tf.contrib.slim

DEFAULT_CONTENT_WEIGHTS = '{"vgg_16/conv3": 1.0}'
DEFAULT_STYLE_WEIGHTS = ('{"vgg_16/conv1": 1e-4, "vgg_16/conv2": 1e-4,'
                         ' "vgg_16/conv3": 1e-4, "vgg_16/conv4": 1e-4}')

flags = tf.flags
flags.DEFINE_string('content_weights', DEFAULT_CONTENT_WEIGHTS,
                    'Content weights.')
flags.DEFINE_string('style_weights', DEFAULT_STYLE_WEIGHTS, 'Style weights.')
flags.DEFINE_float('total_variation_weight', 0.0, 'Total variation weight')
flags.DEFINE_string('checkpoint', None, 'Path to the model checkpoint.')
flags.DEFINE_string('style_images_paths', None, 'Paths to the style images'
                    'for evaluation.')
flags.DEFINE_string('content_images_paths', None, 'Paths to the content images'
                    'for evaluation.')
flags.DEFINE_string('output_dir', None, 'Output directory.')
flags.DEFINE_integer('image_size', 300, 'Image size.')
flags.DEFINE_integer('style_image_size', 300, 'Style image size.')
flags.DEFINE_string('transformer_model', 'nza_das', 'Type of ransformer model.')
flags.DEFINE_integer('maximum_styles_to_evaluate', 1024, 'Maximum number of'
                     'styles to evaluate.')
flags.DEFINE_integer('maximum_results_to_save', 1024, 'Maximum number of'
                     'stylezed images to save.')
FLAGS = flags.FLAGS


def main(unused_argv=None):
  tf.logging.set_verbosity(tf.logging.INFO)
  if not tf.gfile.Exists(FLAGS.output_dir):
    tf.gfile.MkDir(FLAGS.output_dir)

  if not os.path.exists(FLAGS.output_dir):
    os.makedirs(FLAGS.output_dir)

  with tf.Graph().as_default(), tf.Session() as sess:
    # Defines place holder for the style image.
    style_img_ph = tf.placeholder(tf.float32, shape=[None, None, 3])
    style_img_croped_resized = image_utils.center_crop_resize_image(
        style_img_ph, FLAGS.style_image_size)

    # Defines place holder for the content image.
    content_img_ph = tf.placeholder(tf.float32, shape=[None, None, 3])
    content_img_cropped_resized = image_utils.center_crop_resize_image(
        content_img_ph, FLAGS.image_size)

    content_weights = ast.literal_eval(FLAGS.content_weights)
    style_weights = ast.literal_eval(FLAGS.style_weights)
    # Defines the model.
    stylized_images, _, loss_dict, _ = build_model.build_model(
        content_img_cropped_resized,
        style_img_croped_resized,
        trainable=False,
        is_training=False,
        transformer_model_name=FLAGS.transformer_model,
        inception_end_point='Mixed_6e',
        style_prediction_bottleneck=100,
        adds_losses=True,
        content_weights=content_weights,
        style_weights=style_weights,
        total_variation_weight=FLAGS.total_variation_weight)

    eval_dict = loss_dict
    eval_dict['stylized_images'] = stylized_images

    if tf.gfile.IsDirectory(FLAGS.checkpoint):
      checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint)
    else:
      checkpoint = FLAGS.checkpoint
      tf.logging.info('loading latest checkpoint file: {}'.format(checkpoint))
    init_fn = slim.assign_from_checkpoint_fn(
        checkpoint,
        slim.get_variables_to_restore())
    sess.run([tf.local_variables_initializer()])
    init_fn(sess)

    # Gets the list of the input style images.
    style_img_list = tf.gfile.Glob(FLAGS.style_images_paths)
    if len(style_img_list) > FLAGS.maximum_styles_to_evaluate:
      np.random.seed(1234)
      style_img_list = np.random.permutation(style_img_list)
      style_img_list = style_img_list[:FLAGS.maximum_styles_to_evaluate]

    # Gets list of input content images.
    content_img_list = tf.gfile.Glob(FLAGS.content_images_paths)

    total_content_loss = 0
    total_style_loss = 0
    style_names = []
    content_names = []
    content_scores = []
    style_scores = []
    for content_i, content_img_path in enumerate(content_img_list):
      content_img_np = image_utils.load_np_image_uint8(content_img_path)[:, :, :3]
      content_img_name = os.path.basename(content_img_path)[:-4]

      for style_i, style_img_path in enumerate(style_img_list):
        if style_i > FLAGS.maximum_styles_to_evaluate:
          break
        style_img_name = os.path.basename(style_img_path)[:-4]
        style_image_np = image_utils.load_np_image_uint8(style_img_path)[:, :, :3]

        if style_i % 10 == 0:
          tf.logging.info('Stylizing (%d) %s with (%d) %s' %
                          (content_i, content_img_name, style_i,
                           style_img_name))

        eval_dict_res = sess.run(
            eval_dict,
            feed_dict={
                style_img_ph: style_image_np,
                content_img_ph: content_img_np
            })
        stylized_image_res = eval_dict_res['stylized_images']

        total_content_loss += eval_dict_res['total_content_loss']
        total_style_loss += eval_dict_res['total_style_loss']

        content_scores.append(eval_dict_res['total_content_loss'])
        style_scores.append(eval_dict_res['total_style_loss'])
        style_names.append(style_img_name)
        content_names.append(content_img_name)

        if len(content_scores) < FLAGS.maximum_results_to_save:
          # Saves cropped resized content image.
          inp_img_croped_resized_np = sess.run(
              content_img_cropped_resized,
              feed_dict={content_img_ph: content_img_np})
          image_utils.save_np_image(inp_img_croped_resized_np,
                                    os.path.join(FLAGS.output_dir, '%s.jpg' %
                                                 (content_img_name)), 'jpg')

          # Saves cropped resized style image.
          style_img_croped_resized_np = sess.run(
              style_img_croped_resized,
              feed_dict={style_img_ph: style_image_np})
          image_utils.save_np_image(style_img_croped_resized_np,
                                    os.path.join(FLAGS.output_dir, '%s.jpg' %
                                                 (style_img_name)), 'jpg')

          # Saves stylized image.
          image_utils.save_np_image(
              stylized_image_res,
              os.path.join(FLAGS.output_dir, '%s_stylized_%s.jpg' %
                           (content_img_name, style_img_name)), 'jpg')

          # Saves content and style losses.
          scores_file = tf.gfile.Open(
              os.path.join(FLAGS.output_dir, 'scores_%s.csv' % style_img_name),
              'a')
          scores_file.write(
              '%7s,%20s, content loss:% 10.2f, style loss:% 10.2f\n' %
              (style_img_name, content_img_name,
               eval_dict_res['total_content_loss'],
               eval_dict_res['total_style_loss']))
          scores_file.close()

    # Saves losses of all the stylized images in a numpy npz file.
    np.savez(
        os.path.join(FLAGS.output_dir, 'scores_np.npz'),
        content_scores=content_scores,
        style_scores=style_scores,
        style_names=style_names,
        content_names=content_names,
        count=len(content_scores),
        total_content_loss=total_content_loss,
        total_style_loss=total_style_loss)

    scores_file = tf.gfile.Open(
        os.path.join(FLAGS.output_dir, 'total_scores.csv'), 'w')
    scores_file.write('total_content_loss:%.3f total_style_loss: %.3f' %
                      (total_content_loss / len(content_scores),
                       total_style_loss / len(content_scores)))
    scores_file.close()

    tf.logging.info('list of content image names:')
    content_img_names = ''
    for content_i, img_path in enumerate(content_img_list):
      content_img_name = os.path.basename(img_path)[:-4]
      content_img_names += ',' + content_img_name
    tf.logging.info(content_img_names)

    tf.logging.info('list of style image names:')
    style_image_names = ''
    for style_i, style_img_path in enumerate(style_img_list):
      style_img_name = os.path.basename(style_img_path)[:-4]
      style_image_names += ',' + style_img_name
      if style_i > FLAGS.maximum_styles_to_evaluate:
        break
    tf.logging.info(style_image_names)
    tf.logging.info('total_content_loss:%.3f total_style_loss: %.3f' %
                    (total_content_loss / len(content_scores),
                     total_style_loss / len(content_scores)))


if __name__ == '__main__':
  tf.app.run(main)
