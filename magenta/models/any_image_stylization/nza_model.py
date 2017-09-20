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
"""Style transfer network code.

This model does not apply styles in the encoding
layers. Encoding layers (contract) use batch norm as the normalization function.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.slim.python.slim.nets import inception_v3

slim = tf.contrib.slim


def transform(input_, normalizer_fn=None, normalizer_params=None,
              reuse=False, trainable=True, is_training=True):
  """Maps content images to stylized images.

  Args:
    input_: Tensor. Batch of input images.
    normalizer_fn: normalization layer function for applying style
        normalization.
    normalizer_params: dict of parameters to pass to the style normalization op.
    reuse: bool. Whether to reuse model parameters. Defaults to False.
    trainable: bool. Should the parameters be marked as trainable?
    is_training: bool. Is it training phase or not?

  Returns:
    Tensor. The output of the transformer network.
  """
  with tf.variable_scope('transformer', reuse=reuse):
    with slim.arg_scope(
        [slim.conv2d],
        activation_fn=tf.nn.relu,
        normalizer_fn=normalizer_fn,
        normalizer_params=normalizer_params,
        weights_initializer=tf.random_normal_initializer(0.0, 0.01),
        biases_initializer=tf.constant_initializer(0.0),
        trainable=trainable):
      with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
                          normalizer_params=None,
                          trainable=trainable):
        with slim.arg_scope([slim.batch_norm], is_training=is_training,
                            trainable=trainable):
          with tf.variable_scope('contract'):
            h = conv2d(input_, 9, 1, 32, 'conv1')
            h = conv2d(h, 3, 2, 64, 'conv2')
            h = conv2d(h, 3, 2, 128, 'conv3')
      with tf.variable_scope('residual'):
        h = residual_block(h, 3, 'residual1')
        h = residual_block(h, 3, 'residual2')
        h = residual_block(h, 3, 'residual3')
        h = residual_block(h, 3, 'residual4')
        h = residual_block(h, 3, 'residual5')
      with tf.variable_scope('expand'):
        h = upsampling(h, 3, 2, 64, 'conv1')
        h = upsampling(h, 3, 2, 32, 'conv2')
        return upsampling(h, 9, 1, 3, 'conv3', activation_fn=tf.nn.sigmoid)


def style_normalization_activations(pre_name='transformer',
                                    post_name='StyleNorm'):
  """Returns scope name and depths of the style normalization activations.

  Args:
    pre_name: string. Prepends this name to the scope names.
    post_name: string. Appends this name to the scope names.

  Returns:
    string. Scope names of the activations of the transformer network which are
        used to apply style normalization.
    int[]. Depths of the activations of the transformer network which are used
        to apply style normalization.
  """

  scope_names = ['residual/residual1/conv1',
                 'residual/residual1/conv2',
                 'residual/residual2/conv1',
                 'residual/residual2/conv2',
                 'residual/residual3/conv1',
                 'residual/residual3/conv2',
                 'residual/residual4/conv1',
                 'residual/residual4/conv2',
                 'residual/residual5/conv1',
                 'residual/residual5/conv2',
                 'expand/conv1/conv',
                 'expand/conv2/conv',
                 'expand/conv3/conv']
  scope_names = ['{}/{}/{}'.format(pre_name, name, post_name)
                 for name in scope_names]
  depths = [128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 64, 32, 3]

  return scope_names, depths

def conv2d(input_, kernel_size, stride, num_outputs, scope,
           activation_fn=tf.nn.relu):
  """Same-padded convolution with mirror padding instead of zero-padding.

  This function expects `kernel_size` to be odd.

  Args:
    input_: 4-D Tensor input.
    kernel_size: int (odd-valued) representing the kernel size.
    stride: int representing the strides.
    num_outputs: int. Number of output feature maps.
    scope: str. Scope under which to operate.
    activation_fn: activation function.

  Returns:
    4-D Tensor output.

  Raises:
    ValueError: if `kernel_size` is even.
  """
  if isinstance(kernel_size, int):
    kernel_size_x, kernel_size_y = kernel_size, kernel_size
  else:
    if not isinstance(kernel_size, (tuple, list)):
      raise TypeError('kernel_size is expected to be tuple or a list.')
    if len(kernel_size) != 2:
      raise TypeError('kernel_size is expected to be of length 2.')
    kernel_size_x, kernel_size_y = kernel_size
  if kernel_size_x % 2 == 0 or kernel_size_y % 2 == 0:
    raise ValueError('kernel_size is expected to be odd.')
  padding_x = kernel_size_x // 2
  padding_y = kernel_size_y // 2
  padded_input = tf.pad(
      input_, [[0, 0], [padding_x, padding_y],
               [padding_x, padding_y], [0, 0]], mode='REFLECT')
  return slim.conv2d(
      padded_input,
      padding='VALID',
      kernel_size=kernel_size,
      stride=stride,
      num_outputs=num_outputs,
      activation_fn=activation_fn,
      scope=scope)


def upsampling(input_, kernel_size, stride, num_outputs, scope,
               activation_fn=tf.nn.relu):
  """A smooth replacement of a same-padded transposed convolution.

  This function first computes a nearest-neighbor upsampling of the input by a
  factor of `stride`, then applies a mirror-padded, same-padded convolution.

  It expects `kernel_size` to be odd.

  Args:
    input_: 4-D Tensor input.
    kernel_size: int (odd-valued) representing the kernel size.
    stride: int representing the strides.
    num_outputs: int. Number of output feature maps.
    scope: str. Scope under which to operate.
    activation_fn: activation function.

  Returns:
    4-D Tensor output.

  Raises:
    ValueError: if `kernel_size` is even.
  """
  if kernel_size % 2 == 0:
    raise ValueError('kernel_size is expected to be odd.')
  with tf.variable_scope(scope):
    if input_.get_shape().is_fully_defined():
      _, height, width, _ = [s.value for s in input_.get_shape()]
    else:
      shape = tf.shape(input_)
      height, width = shape[1], shape[2]
    upsampled_input = tf.image.resize_nearest_neighbor(
        input_, [stride * height, stride * width])
    return conv2d(upsampled_input, kernel_size, 1, num_outputs, 'conv',
                  activation_fn=activation_fn)


def residual_block(input_, kernel_size, scope, activation_fn=tf.nn.relu):
  """A residual block made of two mirror-padded, same-padded convolutions.

  This function expects `kernel_size` to be odd.

  Args:
    input_: 4-D Tensor, the input.
    kernel_size: int (odd-valued) representing the kernel size.
    scope: str, scope under which to operate.
    activation_fn: activation function.

  Returns:
    4-D Tensor, the output.

  Raises:
    ValueError: if `kernel_size` is even.
  """
  if kernel_size % 2 == 0:
    raise ValueError('kernel_size is expected to be odd.')
  with tf.variable_scope(scope):
    num_outputs = input_.get_shape()[-1].value
    h_1 = conv2d(input_, kernel_size, 1, num_outputs, 'conv1', activation_fn)
    h_2 = conv2d(h_1, kernel_size, 1, num_outputs, 'conv2', None)
    return input_ + h_2
