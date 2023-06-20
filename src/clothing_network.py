import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

#Fashion mnist is a built in dataset for practice
fashionData = keras.datasets.fashion_mnist

#Here we're seperating the dataset from training and evaluating
(e, trainLabel), (evalImage, evalLabel) = fashionData.load_data()

#Potential outputs for our model
labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 
        'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

#Putting our image data into the same range as the rest of our network
trainImage = trainImage / 255.0
evalImage = evalImage / 255.0

#Building our model architecture
imageModel = keras.Sequential([
    #Input Layer
    keras.layers.Flatten(input_shape=(28, 28)),
    #Hidden Layer
    keras.layers.Dense(128, activation='relu'),
    #Output Layer
    keras.layers.Dense(10, activation='softmax')
])

#Compiling the model
imageModel.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

#Training our model with the dataset
imageModel.fit(trainImage, trainLabel, epochs=5)

tLoss, tAccuracy = model.evaluate(evalImage, evalLabel, verbose=1)

print('Accuracy: ', tAccuracy)

predictions = imageModel.predict(evalImage)
print(labels[np.argmax(predictions[0])])
plt.figure()
plt.imshow(evalImage[0])
plt.colorbar()
plt.grid(False)
plt.show()