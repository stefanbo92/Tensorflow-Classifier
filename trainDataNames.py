import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
print ("Packages loaded")

# Load them!
cwd = os.getcwd()
loadpath = cwd + "/custom_data_signs.npz"
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


#define network
tf.set_random_seed(0)
n_input  = dim
n_output = nclass
if use_gray:
    weights  = {
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



#define functions
# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_output])
keepratio = tf.placeholder(tf.float32)

# Functions! 
_pred = conv_basic(x, weights, biases, keepratio, use_gray)['out']
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(_pred, y))
WEIGHT_DECAY_FACTOR = 0.0001
l2_loss = tf.add_n([tf.nn.l2_loss(v) 
            for v in tf.trainable_variables()])
cost = cost + WEIGHT_DECAY_FACTOR*l2_loss
optm = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
_corr = tf.equal(tf.argmax(_pred,1), tf.argmax(y,1)) # Count corrects
accr = tf.reduce_mean(tf.cast(_corr, tf.float32)) # Accuracy
init = tf.initialize_all_variables()
print ("FUNCTIONS READY")




#optimize
# Parameters
training_epochs = 400
batch_size      = 100
display_step    = 1

# Launch the graph
sess = tf.Session()
sess.run(init)

# Training cycle
#Saver 
save_step = 1;
#savedir = "nets/"
saver = tf.train.Saver(max_to_keep=3) 
print('Start time: {:[%H:%M:%S]}'.format(datetime.datetime.now()))
for epoch in range(training_epochs): 
    avg_cost = 0.
    num_batch = int(ntrain/batch_size)+1
    # Loop over all batches
    for i in range(num_batch): 
        randidx = np.random.randint(ntrain, size=batch_size)
        batch_xs = trainimg[randidx, :]
        batch_ys = trainlabel[randidx, :]                
        # Fit training using batch data
        sess.run(optm, feed_dict={x: batch_xs, y: batch_ys
                                  , keepratio:0.7})
        # Compute average loss
        avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys
                                , keepratio:1.})/num_batch

    # Display logs per epoch step
    if epoch % display_step == 0 or epoch == training_epochs-1:
        print ('{:[%H:%M:%S]  }'.format(datetime.datetime.now())+"Epoch: %03d/%03d cost: %.9f" % 
               (epoch, training_epochs, avg_cost))
        train_acc = sess.run(accr, feed_dict={x: batch_xs
                                , y: batch_ys, keepratio:1.})
        print (" Training accuracy: %.3f" % (train_acc))
        test_acc = sess.run(accr, feed_dict={x: testimg
                                , y: testlabel, keepratio:1.})
        print (" Test accuracy: %.3f" % (test_acc))

    #save epoch
    if epoch % save_step == 0:
            saver.save(sess, cwd + "/nets/cnn_facedetect.ckpt-" + str(epoch))
print ("Optimization Finished!")


sess.close()
print ("Session closed.")
