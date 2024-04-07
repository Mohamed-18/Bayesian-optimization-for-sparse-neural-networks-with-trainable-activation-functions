"""
created on Mai 05 2023

@author: Mohamed Fakhfakh
"""
from utils import *
from sampling import *
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from keras import layers
import tensorflow as tf

# Load data

# Exemple : COVID‑19 classification using CT images (COVID / NonCovid)
# https://www.kaggle.com/datasets/luisblanche/covidct

data = []
labels = []


x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=5)
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)



class MyModel(keras.Model):
    def __init__(self, c_star, lambda_star, gamma_star):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, (3, 3))
        self.conv2 = Conv2D(64, (3, 3))
        self.conv3 = Conv2D(128, (3, 3))
        self.maxPooling2D = MaxPooling2D((2,2))
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = Dense(32)
        self.dense2 = Dense(2)
        self.c_star = c_star
        self.lambda_star = lambda_star
        self.gamma_star = gamma_star
        self.mmelu = Modified_MeLU(c_star, lambda_star, gamma_star)

    def call(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.mmelu(x)
        x = self.maxPooling2D(x)

        x = self.conv2(x)
        x = self.mmelu(x)
        x = self.maxPooling2D(x)

        x = self.conv3(x)
        x = self.mmelu(x)
        x = self.maxPooling2D(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.mmelu(x)
        x = self.dense2(x)
        x = tf.nn.sigmoid(x)
        return x



class Modified_MeLU(layers.Layer):
    def __init__(self, c_star, lambda_star, gamma_star):
        super(Modified_MeLU, self).__init__()
        self.c_star = c_star
        self.lambda_star = lambda_star
        self.gamma_star = gamma_star

    def call(self, x):
        return ((1- self.c_star)*tf.math.maximum(x, 0)) + (self.c_star*(tf.math.maximum(self.lambda_star - abs(x-self.gamma_star),0)))

c_initial = np.random.uniform(0, 1)
lambda_b = 1
exp_value = np.random.exponential(scale=1 / lambda_b)
lambda_initial = exp_value / (exp_value + 1)


max_val, min_val = 1, -1
range_size = (max_val - min_val)
gamma_initial = np.random.randn(1) * range_size + min_val

model = MyModel(c_initial, lambda_initial, gamma_initial)


nsHMC_PlugPlay, Compute_K, Compute_E_theta, ComputeProx, accracy = utils_fn(model)

# sampling using the proposed optimizer. This function returns the accuracy and loss of the model at each step.
sampling_fn(MyModel, x_train, y_train, ComputeProx, nsHMC_PlugPlay, Compute_K, Compute_E_theta, accracy, x_test, y_test)




