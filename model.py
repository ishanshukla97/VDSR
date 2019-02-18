import numpy as np
import tensorflow as tf


def VDSR(input):
    layers = 19
    scale = input.get_shape().as_list()[1:3]
    scale = [x*2 for x in scale]
    
    x_input = tf.image.resize_images(input, tf.convert_to_tensor(scale), method=tf.image.ResizeMethod.AREA, align_corners=True)

    conv = conv_layer(x_input, 3, 1, 64, padding='SAME', name='layer_0', activation=tf.nn.relu)
    

    for i in range(1, layers-1):
        conv = conv_layer(conv, 3, 64, 64, padding='SAME', name='layer_{:}'.format(i), activation=tf.nn.relu)

    r_hat = conv_layer(conv, 3, 64, 1, padding='SAME', name='layer_19')

    out_tup = (r_hat, x_input)
    summary = tf.summary.merge_all()
    return (out_tup, summary)


    
def conv_layer(input, k, in_channel, out_channel, padding='VALID', activation=None, name=None):
    if name is not None:
        k_name = name+'/kernel'
        b_name = name+'/bias'

        #Weight initialization similar to He et al. as mentioned in paper
        w = tf.get_variable(k_name, [k, k, in_channel, out_channel], initializer=tf.initializers.random_normal(stddev=np.sqrt(2.0/(k*k)/in_channel)))
        b = tf.get_variable(b_name, initializer=tf.zeros([out_channel]))
        
        tf.summary.histogram(k_name, w)
        tf.summary.histogram(b_name, b)
        
        conv = tf.nn.conv2d(input, w, strides=[1,1,1,1], padding=padding, name=name+'/conv2d_op')
        conv = tf.nn.bias_add(conv, b, name=name+'/bias_add_op')
        #conv = prelu(conv, name+'prelu_acts')
        if activation is not None:
            conv = activation(conv)
            tf.summary.histogram('prelu_acts', conv)
        tf.summary.image('layer_image/layer' + name, conv[:, :, :, 0:1])

    return conv

def conv_transpose(input, k, in_channel, out_channel, strides, padding='valid', activation=None, name=None):
    if name is not None:
        k_name = name+'/kernel'
        b_name = name+'/bias'
        
        w = tf.get_variable(k_name, [k, k, in_channel, out_channel], initializer=tf.initializers.glorot_uniform())
        b = tf.get_variable(b_name, initializer=tf.zeros([in_channel]))
        
        tf.summary.histogram(k_name, w)
        tf.summary.histogram(b_name, b)
        
        output_shape = input.get_shape().as_list()
        output_shape[0] = 256
        output_shape[1], output_shape[2] = output_shape[1]*strides[1], output_shape[2]*strides[2]
        output_shape[3] = in_channel
        output_shape = tf.TensorShape(output_shape)
        
        conv = tf.nn.conv2d_transpose(input, w, output_shape, strides, padding=padding, name=name+'/conv2d_op')
        print(conv.get_shape().as_list())
        conv = tf.nn.bias_add(conv, b, name=name+'/bias_add_op')

        if activation is not None:
            conv = activation(conv)
            tf.summary.histogram('activations', conv)

    return conv

def prelu(_x, name):
    
    alphas = tf.get_variable('prelu_alpha_{:}'.format(name), _x.get_shape()[-1],
                               initializer=tf.constant_initializer(0.0),
                                dtype=tf.float32)
    pos = tf.nn.relu(_x)
    neg = tf.multiply(alphas, (_x - abs(_x))) * 0.5
    tf.summary.histogram('prelu_alpha_{:}'.format(name), alphas)

    return pos + neg




    
