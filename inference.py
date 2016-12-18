import initVars as iV
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
import sys
import cv2
from scipy.misc import  imread,imresize
import operator

#define functions
# tf Graph input
x = tf.placeholder(tf.float32, [None, iV.n_input])
y = tf.placeholder(tf.float32, [None, iV.n_output])
keepratio = tf.placeholder(tf.float32)

# Functions! 
pred = iV.conv_basic(x, iV.weights, iV.biases, keepratio, 1)['out']
init = tf.initialize_all_variables()
print ("FUNCTIONS READY")

# Launch the graph
sess = tf.Session()
sess.run(init)

#Load weights from saver 
saver = tf.train.Saver() 
saver.restore(sess, "/home/stefan/work/tensorflow/checkpoints/stefanduci.ckpt-30")
print("Model restored.")


#openCV stuff
cap=cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while(True):    
    ret,frame=cap.read()
    frame_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(frame_gray)
	
    for(xf,yf,w,h) in faces:
        if((w>100)and(h>100)):
            cv2.rectangle(frame,(xf,yf),(xf+w,yf+h),(255,0,0),2) #draw rectangle in video stream		
            img_gray  = frame_gray[yf:yf+h,xf:xf+w] #load grayscale

            img_gray_resize=imresize(img_gray, [64, 64])/255. #resize image to [64x64] and scale to [0,1]
            #perform normalization and mean equalization
            #dst=cv2.normalize(img_gray_resize,alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,dtype=-1) #normlize image
            #res = np.float32(dst)
            #imageMean=cv2.mean(img_gray_resize)[0]
            #print imageMean
            #dst=res*(0.588/imageMean)
            #print cv2.mean(dst)[0]

            img_grayvec   = np.reshape(img_gray_resize, (1, -1))
            #img_grayvec   = np.reshape(dst, (1, -1)) #reshape matrix to vector

            #print img_grayvec.shape
            #print img_grayvec

            predictiton=sess.run(tf.nn.softmax(pred), feed_dict={x: img_grayvec,keepratio:1.}) #make prediction
            print (predictiton)

            index, value = max(enumerate(predictiton[0]), key=operator.itemgetter(1)) #find highest value in output vector

            className=""
            if index==0:  
                className="Stefan"
            elif index ==1:
                className="Soeren"
            else:
                className="Mr.X"

            print ("Prediciton is class '%s' with accuracy %0.3f"%(className,value))
            textY=yf-10
            if textY<0:
                textY=yf
            cv2.putText(frame, className+" "+str(round(value,3)), (xf,textY), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,0),2)
			
    cv2.imshow('frame',frame)
    pressed=cv2.waitKey(1)
    if pressed==107: #if 'k' is pressed
        break

cap.release()
cv2.destroyAllWindows()

sess.close()
print ("Session closed.")
