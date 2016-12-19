# Tensorflow-Classifier

This repository contains a template for creating image classifier using a convolutional neural net in tensorflow.


## Usage
Here you can find a brief description on how to use the different python scripts for your neural net classifier.

#### dataGenerator.py

First you need to prepare the training and test data. Therefore you need to put your training images in the 'img_dataset' folder. Each class should have one own folder. Then the paths to these folders need to be set in the dataGenerator.py. For example:
```python
paths = {"img_dataset/train_signs/0","img_dataset/train_signs/1"}
```
If you have two different classes in the folders "img_dataset/train_signs/0" and "img_dataset/train_signs/1".


#### initNet.py

This python script is actually holding the structure of the neural net. In this example a simple convolutional net with two confolutional layer and two fully connected layers is employed. If you want to change the structure of the neural net to make it more complex you have to change the initNet.py file.


#### trainData.py

Now the training can start. Just run trainData.py and the training process will start automatically. It wil also show the progress for each epoch. The trained weights will be saved in the 'checkpoints' folder.


#### predictor.py

This program will finally use the trained neural net for making a classification of an image. So it opens the video camera and converts the images to the right format and passes it to the neural net classifier. You will need to set the right checkpoint file to load the correct weight to your neural net. Then it will print out the predictions.
