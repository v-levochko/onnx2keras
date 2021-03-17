from tensorflow import keras
import tensorflow as tf
import numpy as np
import logging

from .custom_layers import Resample


def convert_upsample(node, params, layers, lambda_func, custom_objects, node_name, keras_name):
    """
    Convert upsample.
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras:upsample')

    if "scales" in params:
        # for opset version - 7
        if len(node.input) != 1:
            raise AttributeError('Unsupported number of inputs')
        scale = params['scales'][-2:]
    else:
        # for opset version - 9+
        # Upsample since opset version 9 uses input[1] as 'scales' instead of attributes.
        scale = layers[node.input[1]][-2:]

    mode = params['mode'].decode('utf-8')
    if mode != 'nearest':
        mode = 'bilinear'
        logger.warning(f'Resampling mode: {mode}. Falling back to `bilinear`, which may cause unexpected results')
    
    upsampling = Resample(scale, method=mode, data_format='channels_first', name=keras_name)

    layers[node_name] = upsampling(layers[node.input[0]])
    custom_objects['Resample'] = Resample
