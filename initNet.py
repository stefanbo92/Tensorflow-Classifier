import os
import numpy as np
import tensorflow as tf
import sys



# Load them!
cwd = os.getcwd()
loadpath = cwd + "/data.npz"
l = np.load(loadpath)

# See what's in here
print (l.files)

# Parse data
trainimg = l['trainimg']
trainlabel = l['trainlabel']
testimg = l['testimg']
testlabel = l['testlabel']
imgsize = l['imgsize']
use_gray = l['use_gray']
ntrain = trainimg.shape[0]
nclass = trainlabel.shape[1]
dim    = trainimg.shape[1]
ntest  = testimg.shape[0]
print ("%d train images loaded" % (ntrain))
print ("%d test images loaded" % (ntest))
print ("%d dimensional input" % (dim))
print ("Image size is %s" % (imgsize))
print ("%d classes" % (nclass))


#define variables
tf.set_random_seed(0)
n_input  = dim
n_output = nclass
if use_gray:
    weights  = { #weights for convolution layer 1 and 2 and dense (fully connected layer) 1 and 2
        'wc1': tf.Variable(tf.random_normal([5, 5, 1, 128], stddev=0.1),name="wc1"),
        'wc2': tf.Variable(tf.random_normal([5, 5, 128, 128], stddev=0.1),name="wc2"),
        'wd1': tf.Variable(tf.random_normal(
                [(int)(imgsize[0]/4*imgsize[1]/4)*128, 128], stddev=0.1),name="wd1"),
        'wd2': tf.Variable(tf.random_normal([128, n_output], stddev=0.1),name="wd2")
    }
else:
    weights  = {
        'wc1': tf.Variable(tf.random_normal([5, 5, 3, 128], stddev=0.1),name="wc1"),
        'wc2': tf.Variable(tf.random_normal([5, 5, 128, 128], stddev=0.1),name="wc2"),
        'wd1': tf.Variable(tf.random_normal(
                [(int)(imgsize[0]/4*imgsize[1]/4)*128, 128], stddev=0.1),name="wd1"),
        'wd2': tf.Variable(tf.random_normal([128, n_output], stddev=0.1),name="wd2")
    }
biases   = {
    'bc1': tf.Variable(tf.random_normal([128], stddev=0.1),name="bc1"),
    'bc2': tf.Variable(tf.random_normal([128], stddev=0.1),name="bc2"),
    'bd1': tf.Variable(tf.random_normal([128], stddev=0.1),name="bd1"),
    'bd2': tf.Variable(tf.random_normal([n_output], stddev=0.1),name="bd2")
}

#define network
def conv_basic(_input, _w, _b, _keepratio, _use_gray):
    # INPUT
    if _use_gray:
        _input_r = tf.reshape(_input, shape=[-1, imgsize[0], imgsize[1], 1])
    else:
        _input_r = tf.reshape(_input, shape=[-1, imgsize[0], imgsize[1], 3])
    # CONVOLUTION LAYER 1
    _conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(_input_r
        , _w['wc1'], strides=[1, 1, 1, 1], padding='SAME'), _b['bc1']))
    _pool1 = tf.nn.max_pool(_conv1, ksize=[1, 2, 2, 1]
        , strides=[1, 2, 2, 1], padding='SAME')
    _pool_dr1 = tf.nn.dropout(_pool1, _keepratio)
    # CONVOLUTION LAYER 2
    _conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(_pool_dr1
        , _w['wc2'], strides=[1, 1, 1, 1], padding='SAME'), _b['bc2']))
    _pool2 = tf.nn.max_pool(_conv2, ksize=[1, 2, 2, 1]
        , strides=[1, 2, 2, 1], padding='SAME')
    _pool_dr2 = tf.nn.dropout(_pool2, _keepratio)
    # VECTORIZE
    _dense1 = tf.reshape(_pool_dr2
                         , [-1, _w['wd1'].get_shape().as_list()[0]])
    # FULLY CONNECTED LAYER 1
    _fc1 = tf.nn.relu(tf.add(tf.matmul(_dense1, _w['wd1']), _b['bd1']))
    _fc_dr1 = tf.nn.dropout(_fc1, _keepratio)
    # FULLY CONNECTED LAYER 2
    _out = tf.add(tf.matmul(_fc_dr1, _w['wd2']), _b['bd2'])
    # RETURN
    out = {
        'out': _out
    }
    return out
print ("NETWORK READY")





