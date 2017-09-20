# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generates stylized images with different strengths of a stylization.

For each pair of the content and style images this script computes stylized
images with different strengths of stylization (interpolates between the
identity transform parameters and the style parameters for the style image) and
saves them to the given output_dir.
See run_interpolation_with_identity.sh for example usage.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf

from magenta.models.any_image_stylization import any_image_stylization_build_model as build_model
from magenta.models.image_stylization import image_utils

slim = tf.contrib.slim

flags = tf.flags
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
flags.DEFINE_integer('num_interpolations', 11, 'Number of the interpolations'
                     'between the parameters of the identity transform and the'
                     'style parameters of the style image.')
FLAGS = flags.FLAGS


def main(unused_argv=None):
  tf.logging.set_verbosity(tf.logging.INFO)
  if not tf.gfile.Exists(FLAGS.output_dir):
    tf.gfile.MkDir(FLAGS.output_dir)

  with tf.Graph().as_default(), tf.Session() as sess:
    # Defines place holder for the style image.
    style_img_ph = tf.placeholder(tf.float32, shape=[None, None, 3])
    style_img_croped_resized = image_utils.center_crop_resize_image(
        style_img_ph, FLAGS.style_image_size)

    # Defines place holder for the content image.
    content_img_ph = tf.placeholder(tf.float32, shape=[None, None, 3])
    content_img_cropped_resized = image_utils.center_crop_resize_image(
        content_img_ph, FLAGS.image_size)

    # Defines the model.
    stylized_images, _, _, bottleneck_feat = build_model.build_model(
        content_img_cropped_resized,
        style_img_croped_resized,
        trainable=False,
        is_training=False,
        inception_end_point='Mixed_6e',
        style_prediction_bottleneck=100,
        adds_losses=False)

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

    for content_i, content_img_path in enumerate(content_img_list):
      content_img_np = image_utils.load_np_image_uint8(content_img_path)[:, :, :3]
      content_img_name = os.path.basename(content_img_path)[:-4]

      # Saves cropped resized content image.
      inp_img_croped_resized_np = sess.run(
          content_img_cropped_resized,
          feed_dict={content_img_ph: content_img_np})
      image_utils.save_np_image(inp_img_croped_resized_np,
                                os.path.join(FLAGS.output_dir, '%s.jpg' %
                                             (content_img_name)))

      # Computes bottleneck features of the style prediction network for the
      # identity transform.
      identity_params = sess.run(
          bottleneck_feat, feed_dict={style_img_ph: content_img_np})

      for style_i, style_img_path in enumerate(style_img_list):
        if style_i > FLAGS.maximum_styles_to_evaluate:
          break
        style_img_name = os.path.basename(style_img_path)[:-4]
        style_image_np = image_utils.load_np_image_uint8(style_img_path)[:, :, :3]

        if style_i % 10 == 0:
          tf.logging.info('Stylizing (%d) %s with (%d) %s' %
                          (content_i, content_img_name, style_i,
                           style_img_name))

        # Saves cropped resized style image.
        style_img_croped_resized_np = sess.run(
            style_img_croped_resized, feed_dict={style_img_ph: style_image_np})
        image_utils.save_np_image(style_img_croped_resized_np,
                                  os.path.join(FLAGS.output_dir, '%s.jpg' %
                                               (style_img_name)))

        # Computes bottleneck features of the style prediction network for the
        # given style image.
        style_params = sess.run(
            bottleneck_feat, feed_dict={style_img_ph: style_image_np})

        # Interpolates between the parameters of the identity transform and
        # style parameters of the given style image.
        for interp_i in range(0, FLAGS.num_interpolations):
          wi = interp_i / (FLAGS.num_interpolations - 1)
          stylized_image_res = sess.run(
              stylized_images,
              feed_dict={
                  bottleneck_feat:
                      identity_params * (1 - wi) + style_params * wi,
                  content_img_ph:
                      content_img_np
              })

          # Saves stylized image.
          image_utils.save_np_image(
              stylized_image_res,
              os.path.join(FLAGS.output_dir, '%s_stylized_%s_%d.jpg' %
                           (content_img_name, style_img_name, interp_i)))

if __name__ == '__main__':
  tf.app.run(main)
