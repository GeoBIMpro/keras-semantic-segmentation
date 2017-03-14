"""
Implementation of Fully Convolutional DenseNets for Semantic Segmentation
(aka The One Hundred Layers Tiramisu) https://arxiv.org/abs/1611.09326
"""

from keras.models import Model
from keras.layers import (
    Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Activation,
    Reshape, BatchNormalization, Dropout)
from keras.regularizers import l2
from keras.backend import shape


def make_conv_layer(input_tensor, block_idx, layer_idx, growth_rate=16,
                    drop_prob=0.2, weight_decay=1e-4):
    name = 'batch_norm_{}_{}'.format(block_idx, layer_idx)
    x = BatchNormalization(name=name)(input_tensor)
    x = Activation('relu')(x)
    name = 'conv_{}_{}'.format(block_idx, layer_idx)
    x = Convolution2D(growth_rate, 3, 3, name=name, init='he_uniform',
                      border_mode='same', bias=False,
                      W_regularizer=l2(weight_decay))(x)
    x = Dropout(drop_prob)(x)

    return x


def make_dense_block(input_tensor, block_idx, nb_layers, growth_rate=16,
                     drop_prob=0.2, weight_decay=1e-4):
    layer_outputs = []
    x = input_tensor
    for layer_idx in range(nb_layers):
        layer_output = make_conv_layer(
            x, block_idx, layer_idx, growth_rate, drop_prob, weight_decay)

        x = merge([x, layer_output], mode='concat')
        layer_outputs.append(layer_output)

    block_output = merge(layer_outputs, mode='concat')
    block_size = nb_layers * growth_rate

    return block_output, block_size


def make_trans_down(input_tensor, nb_filters, block_idx, drop_prob=0.2,
                    weight_decay=1e-4):
    name = 'batch_norm_trans_down_{}'.format(block_idx)
    x = BatchNormalization(name=name)(input_tensor)
    x = Activation('relu')(x)
    name = 'conv_trans_down_{}'.format(block_idx)
    x = Convolution2D(nb_filters, 1, 1, name=name, init='he_uniform',
                      border_mode='same', bias=False,
                      W_regularizer=l2(weight_decay))(x)
    x = Dropout(drop_prob)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    return x


def make_trans_up(input_tensor, nb_filters, block_idx, weight_decay=1e-4):
    x = UpSampling2D(size=(2, 2))(input_tensor)
    name = 'conv_trans_up_{}'.format(block_idx)
    x = Convolution2D(nb_filters, 3, 3, name=name, init='he_uniform',
                      border_mode='same', bias=False,
                      W_regularizer=l2(weight_decay))(x)

    return x


def make_densenet_seg(input_shape, nb_labels, growth_rate=16,
                      drop_prob=0.2, weight_decay=1e-4,
                      down_blocks=[4, 5, 7, 10, 12, 15],
                      up_blocks=[12, 10, 7, 5, 4]):
    skips = []

    name = 'conv_initial'
    nb_rows, nb_cols, nb_channels = input_shape
    nb_filters = nb_channels * growth_rate

    input_tensor = Input(input_shape)
    x = Convolution2D(nb_filters, 3, 3, name=name, init='he_uniform',
                      border_mode='same', bias=False,
                      W_regularizer=l2(weight_decay))(input_tensor)

    x_size = nb_filters

    for block_idx, nb_layers in enumerate(down_blocks):
        block_output, block_size = make_dense_block(
            x, block_idx, nb_layers, growth_rate, drop_prob, weight_decay)

        if block_idx < len(down_blocks) - 1:
            x = merge([x, block_output], mode='concat')
            x_size += block_size
            skips.append(x)

            x = make_trans_down(x, x_size, block_idx, drop_prob, weight_decay)
        else:
            x = block_output
            x_size = block_size

    for nb_layers, skip in zip(up_blocks, reversed(skips)):
        block_idx += 1

        x = make_trans_up(x, x_size, block_idx, weight_decay)

        x = merge([x, skip], mode='concat')
        x, block_size = make_dense_block(
            x, block_idx, nb_layers, growth_rate, drop_prob, weight_decay)
        x_size = block_size

    name = 'conv_final'
    x = Convolution2D(nb_labels, 1, 1, name=name, init='he_uniform',
                      border_mode='same', bias=False,
                      W_regularizer=l2(weight_decay))(x)

    x = Reshape((nb_rows * nb_cols, nb_labels))(x)
    x = Activation('softmax')(x)
    x = Reshape((nb_rows, nb_cols, nb_labels))(x)

    model = Model(input=input_tensor, output=x)

    return model
