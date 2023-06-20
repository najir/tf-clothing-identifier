# tf-clothing-identifier
    Isaac Perks
    06/19/2023

# Description

A basic neural network built in TensorFlow following the basic fashion mnist dataset tutorial
- Imported fashion mnist data with keras
    - Sepereate training and eval data
    - created labels for future use based on datasets labels
    - Set data within range of networks data(0 - 1)
- Built a sequential model
    - flattened the input layer from 28,28 to a flat 784 layer
    - created a hidden layer of 128 nodes with rectified linear activation function
    - set up an output layer with probability distrobution via softmax on a label size of 10
- Compiled the model
    - adam optimizer
    - sparse categorical crossentropy for loss
    - accuracy for the metric
- Fit and trained the model on our training set
- Evaluated our data on our eval set and printed the accuracy metric
- Used evaluation set to create a prediction
    - Printed prediction
    - Showed image of clothing
