import cv2
import numpy as np
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Input, merge, SpatialDropout2D
from keras.layers import Convolution2D, AveragePooling2D, UpSampling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

from data import load_train_data, load_test_data

K.set_image_dim_ordering('th')  # Theano dimension ordering in this code

img_rows = 100
img_cols = 160
stack = 10

smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def create_model():
    input = Input(shape=(1, img_rows, img_cols))
    
    conv1 = Convolution2D(32, 3, 3, border_mode='same', init='he_normal')(input)
    conv1 = LeakyReLU()(conv1)
    conv1 = SpatialDropout2D(0.2)(conv1)
    conv1 = Convolution2D(32, 3, 3, border_mode='same', init='he_normal')(conv1)
    conv1 = LeakyReLU()(conv1)
    conv1 = SpatialDropout2D(0.2)(conv1)
    pool1 = AveragePooling2D(pool_size=(2,2))(conv1)
    
    conv2 = Convolution2D(64, 3, 3, border_mode='same', init='he_normal')(pool1)
    conv2 = LeakyReLU()(conv2)
    conv2 = SpatialDropout2D(0.2)(conv2)
    conv2 = Convolution2D(64, 3, 3, border_mode='same', init='he_normal')(conv2)
    conv2 = LeakyReLU()(conv2)
    