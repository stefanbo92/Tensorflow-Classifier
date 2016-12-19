import initNet as iN
import os
import numpy as np
import tensorflow as tf
import datetime
import sys
import cv2
from scipy.misc import  imread,imresize
import operator

#define functions
# tf Graph input
x = tf.placeholder(tf.float32, [None, iN.n_input])
y = tf.placeholder(tf.float32, [None, iN.n_output])
keepratio = tf.placeholder(tf.float32)

# Functions! 
pred = iN.conv_basic(x, iN.weights, iN.biases, keepratio, 1)['out']
init = tf.initialize_all_variables()
print ("FUNCTIONS READY")

# Launch the graph
sess = tf.Session()
sess.run(init)

#Load weights from saver 
saver = tf.train.Saver() 
saver.restore(sess, "checkpoints/weights-1.ckpt")
print("Model restored.")

#openCV stuff
cap=cv2.VideoCapture(0)

while(True):
    #get current frame from camera
    ret,frame=cap.read()
    # turn img to grayscale and resize
    frame_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    img_gray_resize=imresize(frame_gray, [iN.imgsize[0], iN.imgsize[1]])/255.
    img_grayvec   = np.reshape(img_gray_resize, (1, -1))

    predictiton=sess.run(tf.nn.softmax(pred), feed_dict={x: img_grayvec,keepratio:1.}) #make prediction
    print ("Predictions are: ")
    print (predictiton)

    index, value = max(enumerate(predictiton[0]), key=operator.itemgetter(1)) #find highest value in output vector

    className=""
    if index==0:  
        className="Left"
    elif index ==1:
        className="Right"
    elif index ==2:
        className="Treasure"
    elif index ==3:
        className="Back"

    print ("Prediciton is class '%s' with accuracy %0.3f"%(className,value))
	
    
    cv2.putText(frame, className+" "+str(round(value,3)), (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,0),2)
			
    cv2.imshow('frame',frame)
    pressed=cv2.waitKey(1)
    if pressed==107: #if 'k' is pressed
        break

cap.release()
cv2.destroyAllWindows()

sess.close()
print ("Session closed.")
