from tensorflow.keras.layers import Layer
import tensorflow as tf


class Resample(Layer):
    """
    Resampling layer; wraps `tf.image.resize()`.
    
    Parameters:
    :param scale: (h, w)-tuple of resampling factors
    :param method: method name string ('nearest', 'bilinear' etc.)
    :param data_format: keras data format string ('channels_first' or 'channels_last')
    :param antialias: antialiasing flag
    :param name: layer name
    """
    
    def __init__(self, scale, method, data_format = "channels_last", antialias = False, name = None):
        super().__init__(name = name)
        self.scale = tuple(scale)
        self.method = method
        self.antialias = antialias
        self.data_format = data_format
        
    def get_config(self):
        return dict(
            scale = self.scale, 
            method = self.method, 
            antialias = self.antialias,
            data_format = self.data_format, 
            name = self.name
        )
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
        
    def build(self, input_shape):
        if self.data_format == 'channels_last':
            _, h, w, _ = input_shape
        else:
            _, _, h, w = input_shape
        self.size = (int(h * self.scale[0]), int(w * self.scale[1]))
        
    def call(self, inputs):
        if self.data_format != 'channels_last':
            inputs = tf.transpose(inputs, [0,2,3,1])
        resampled = tf.image.resize(inputs, self.size, self.method, antialias = self.antialias)
        if self.data_format != 'channels_last':
            resampled = tf.transpose(resampled, [0,3,1,2])
        return resampled
    