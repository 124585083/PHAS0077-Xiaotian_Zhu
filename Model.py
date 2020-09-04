import tensorflow as tf
import numpy as np
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix

class Model:
    """

    I used model to deal with the training and testing processes, 
    First, we use Choose_model(name, size) to choose the model you want: 'Resnet', 'Simple', 'CNN'
        with input size has 3 dimension (raws, columns, channels)
    Second, compile_it() and fit_it(images, labels)
    Finally we get the test result by get_confusion(test_images, test_labels).

    """

    def Choose_model(self, name, size):
        """
        Input name can be 'Resnet', 'Simple', 'CNN', 
        Input size should have three dimensions like (256, 256, 3)
        Without return but the self.model will changes.
        """
        self.name = name
        self.size = size

        if self.name == 'Resnet50':
            self.model = self.Resnet_50()
        
        elif self.name == 'Resnet101':
            self.model = self.Resnet_101()

        elif self.name == 'Resnet152':
            self.model = self.Resnet_152()
    
        elif self.name == 'Simple':
            self.model = self.Simple()

        elif self.name == 'CNN':
            self.model = self.CNN()

    # Compile the self.model.
    def compile_it(self):
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics = ['accuracy'])
    
    def fit_it(self, images, labels, epoch_num=20, shuf=True):
        """
        Input images are the training date of the model,
        Input labels are the training labels of the model, both should be numpy format.
        To fit the model, I set the default epoches number as 20 and shuffle is True.
        """
        self.model.fit(images, labels, epochs=epoch_num, shuffle=shuf)

    # Test the model and get the confusion matrix to this model.
    def get_confusion(self, test_images, test_labels):
        _, test_acc = self.model.evaluate(test_images, test_labels)
        print('\nTest accuracy:', test_acc)
        Prediction = self.model.predict(test_images)
        Predict_labels = []
        for i in Prediction:
            Predict_labels.append(np.argmax(i))
        print(confusion_matrix(test_labels, Predict_labels))
    
    # The Resnet152 model, return a keras model.
    def Resnet_152(self):
        model = keras.Sequential([
            keras.applications.ResNet152(input_shape=self.size),
            keras.layers.Dense(5, activation='softmax')
        ])
        return model

    # The Resnet101 model, return a keras model.
    def Resnet_101(self):
        model = keras.Sequential([
            keras.applications.ResNet101(input_shape=self.size),
            keras.layers.Dense(5, activation='softmax')
        ])
        return model

    # The Resnet50 model, return a keras model.
    def Resnet_50(self):
        model = keras.Sequential([
            keras.applications.ResNet50(input_shape=self.size),
            keras.layers.Dense(5, activation='softmax')
        ])
        return model

    # Return a Simple network model.
    def Simple(self):
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=self.size),
            keras.layers.Dense(2240, activation='relu'),
            keras.layers.Dense(224, activation='relu'),
            keras.layers.Dense(112, activation='relu'),
            keras.layers.Dense(56, activation='relu'),
            keras.layers.Dense(28, activation='relu'),
            keras.layers.Dense(5, activation='softmax'),       
        ])
        return model
    
    # Return a CNN network model.
    def CNN(self):
        model = keras.Sequential([
            # convolutional layer
            keras.layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=self.size),
            keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
            keras.layers.MaxPooling2D(pool_size=(2,2)),
            keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
            keras.layers.Conv2D(8, 3, padding='same', activation='relu'),
            keras.layers.MaxPooling2D(pool_size=(2,2)),
            # flatten output of conv
            keras.layers.Flatten(),
            keras.layers.Dense(224, activation='relu'),
            keras.layers.Dense(112, activation='relu'),
            keras.layers.Dense(56, activation='relu'),
            keras.layers.Dense(28, activation='relu'),
            keras.layers.Dense(5, activation='softmax'),       
        ])
        return model