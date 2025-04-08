import os
import numpy as np 
from keras.datasets import mnist
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA 
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score   
import tensorflow as tf


import bloqade 
from bloqade.ir.location import Chain, start

physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)
    print("GPU: ", gpu)
print("Num GPUs Available: ", len(physical_devices))
