import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Dense
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from keras.layers import LeakyReLU
from scipy import ndimage
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Reshape,BatchNormalization
import matplotlib.pyplot as plt
from keras import backend as K
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Model
import keras
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Input,Dropout
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
import matplotlib.pyplot as plt
from keras import backend as K
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers import BatchNormalization,MaxPooling3D, Reshape,RNN,LSTM,SimpleRNN
from keras.layers import Concatenate,TimeDistributed
from keras.models import Model


def ReturnModel():
    inp1 = Input((128,128,1))

    x = Conv2D(16, (3, 3), padding='same',)(inp1)
    #x = Activation('relu')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = MaxPooling2D(pool_size=(2,2), padding='same')(x)
    #---------------------------------------------------------
    x = Conv2D(32,(3, 3), padding='same')(x)
    #x = Activation('relu')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = MaxPooling2D(pool_size=(2,2), padding='same')(x)
    #---------------------------------------------------------
    x = Conv2D(64,(3, 3), padding='same')(x)
    #x = Activation('relu')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = MaxPooling2D(pool_size=(2,2), padding='same')(x)
    #---------------------------------------------------------
    x = Conv2D(64,(3, 3), padding='same')(x)
    #x = Activation('relu')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = MaxPooling2D(pool_size=(2,2), padding='same')(x)
    #---------------------------------------------------------
    x = Conv2D(64,(3, 3), padding='same')(x)
    x = Activation('tanh')(x)
    x = MaxPooling2D(pool_size=(2,2), padding='same')(x)
    #---------------------------------------------------------
    encoder = Model(inputs=inp1,outputs=x)
    rms = keras.optimizers.Adam(lr=0.001)
    encoder.compile(optimizer=rms, loss='mse')

    encoder.load_weights("models/tanh_en_2_24_en.h5")
    encoder.trainable=False
    rms = keras.optimizers.Adam(lr=0.001)
    encoder.compile(optimizer=rms, loss='mse')
    encoder.summary()



    #Time distributed ENCODER
    modelpre = Sequential()
    modelpre.add(TimeDistributed(encoder, input_shape=(3,128,128,1)))
    modelpre.add(TimeDistributed(Flatten(), input_shape=(3,1024)))
    modelpre.trainable=False

    rms = keras.optimizers.Adam(lr=0.001)
    modelpre.compile(optimizer=rms, loss='mse')

    inp = Input((4,4,64))
    x = UpSampling2D((2, 2))(inp)
    #---------------------------------------------------------
    x = Conv2D(64,(3, 3), padding='same')(x)
    #x = Activation('relu')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = UpSampling2D((2, 2))(x)
    #---------------------------------------------------------
    x = Conv2D(32,(3, 3), padding='same')(x)
    #x = Activation('relu')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = UpSampling2D((2, 2))(x)
    #---------------------------------------------------------
    x = Conv2D(16,(3, 3), padding='same')(x)
    #x = Activation('relu')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = UpSampling2D((2, 2))(x)
    #---------------------------------------------------------
    x = Conv2D(16,(3, 3), padding='same')(x)
    #x = Activation('relu')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = UpSampling2D((2, 2))(x)
    #---------------------------------------------------------
    x = Conv2D(1,(3, 3), padding='same')(x)
    output = Activation('sigmoid')(x)
    #---------------------------------------------------------
    decoder = Model(inputs=inp, outputs = output)
    rms = keras.optimizers.Adam(lr=0.001)
    decoder.compile(optimizer=rms, loss='mse')

    decoder.load_weights("models/tanh_de_2_24_en.h5")
    decoder.trainable=False
    rms = keras.optimizers.Adam(lr=0.001)
    decoder.compile(optimizer=rms, loss='mse')
    decoder.summary()
    #####################################################################################################################
    #---------------------------------MIDDLE LAYER----------------------------------------------------------------------------
    #####################################################################################################################

    encoder_inputs = Input(shape=(3,1024))
    encoder = LSTM(1024, input_shape=(3,1024), return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]



    decoder_inputs = Input((1,1024))
    decoder_lstm = LSTM(1024, return_sequences=True, return_state=False)
    decoder_outputs = decoder_lstm(decoder_inputs,initial_state=encoder_states)

    modelmid = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    rms = keras.optimizers.Adam(lr=0.001)
    modelmid.compile(optimizer=rms, loss='mse')

    
    modelmid.load_weights("models/tanh_seq2seq.h5")


    #Dense layer and output a RELU 1024-1024 dense layer. Add 

    rms = keras.optimizers.Adam(lr=0.001)
    modelmid.compile(optimizer=rms, loss='mse',
                metrics=['accuracy'])
    modelmid.summary()



    #SUPERMODEL
    inp1 = Input((3,128, 128, 1))
    inp2 = Input((1,1024))

    x = modelpre(inp1)
    x = modelmid([x,inp2])
    x = Reshape((4, 4, 64))(x)
    output = decoder(x)

    generator = Model(inputs=[inp1,inp2],outputs=output)
    rms = keras.optimizers.Adam(lr=0.001)
    generator.compile(optimizer=rms, loss='mse')
    return generator
