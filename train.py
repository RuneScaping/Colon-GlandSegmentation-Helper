import cv2
import numpy as np
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Input, merge, SpatialDropout2D
from keras.layers import Convolution2D, AveragePooling2D, UpSampling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras import backe