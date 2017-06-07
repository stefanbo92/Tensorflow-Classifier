import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
import initNet as iN
print ("Packages loaded")


#define functions
# tf Graph input
x = tf.placeholder(tf.float32, [None, iN.n_input])
y = tf.placeholder(tf.float32, [None, iN.n_output])
keepratio = tf.placeholder(tf.float32)

# Functions! 
_pred = iN.conv_basic(x, iN.weights, iN.biases, keepratio, iN.use_gray)['out']
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=_pred, labels=y))
WEIGHT_DECAY_FACTOR = 0.0001
l2_loss = tf.add_n([tf.nn.l2_loss(v) 
            for v in tf.trainable_variables()])
cost = cost + WEIGHT_DECAY_FACTOR*l2_loss
optm = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
_corr = tf.equal(tf.argmax(_pred,1), tf.argmax(y,1)) # Count corrects
accr = tf.reduce_mean(tf.cast(_corr, tf.float32)) # Accuracy
init = tf.global_variables_initializer()
print ("FUNCTIONS READY")


#optimize
# Parameters
training_epochs = 40
batch_size      = 64
display_step    = 1

# Launch the graph
sess = tf.Session()
sess.run(init)

# Training cycle
#Saver 
save_step = 1;
saver = tf.train.Saver(max_to_keep=3) 
print('Start time: {:[%H:%M:%S]}'.format(datetime.datetime.now()))
for epoch in range(training_epochs): 
    avg_cost = 0.
    num_batch = int(iN.ntrain/batch_size)+1
    # Loop over all batches
    for i in range(num_batch): 
        randidx = np.random.randint(iN.ntrain, size=batch_size)
        batch_xs = iN.trainimg[randidx, :]
        batch_ys = iN.trainlabel[randidx, :]                
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
        test_acc = sess.run(accr, feed_dict={x: iN.testimg
                                , y: iN.testlabel, keepratio:1.})
        print (" Test accuracy: %.3f" % (test_acc))

    #save epoch
    if epoch % save_step == 0:
            saver.save(sess, iN.cwd + "/checkpoints/weights-"+ str(epoch)+".ckpt")
print ("Optimization Finished!")


sess.close()
print ("Session closed.")
