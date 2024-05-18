#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import tensorflow.keras as keras
import os
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
import tensorflow as tf
import pdb
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
#np.random.seed(1)
#tf.random.set_seed(2)
def dense_decoder(ncell):
    model = Sequential(name = 'dense')
    model.add(Dense(16384, input_shape = (ncell,)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(8192))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096))
    
    model.add(Activation('sigmoid'))
    model.add(Reshape(target_shape = (64, 64, 1), name = 'dense_out'))

    return model
def CAE(input_shape):
    model = Sequential(name = 'cae')

    #CAE encoder
    model.add(Conv2D(256, (7, 7), strides = (2, 2), padding = 'same', input_shape = input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(512, (5, 5), strides = (2, 2), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Conv2D(1024, (3, 3), strides = (2, 2), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Conv2D(1024, (3, 3), strides = (2, 2), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    
    #CAE decoder
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(1024, (3, 3), strides = (1, 1), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(512, (3, 3), strides = (1, 1), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
        
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(256, (3, 3), strides = (1, 1), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(256, (3, 3), strides = (1, 1), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(128, (5, 5), strides = (1, 1), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(1, (7, 7), strides = (1, 1), padding = 'same'))
    #cae_out = Activation('tanh')(cae)
    model.add(BatchNormalization(name = 'cae_out'))
    #cae_out = Cropping2D(cropping = ((3, 3), (3, 3)), name = 'cae_out')(cae)
    return model
if __name__ == '__main__':




    data_dir = 'D:\\20201015_decodingDataset_code\\simulated data1'
    img_test = np.load(os.path.join(data_dir, 'movie_03_test_pic_1999_nuc_800.npy')) #image
    spike_test = np.load(os.path.join(data_dir, 'movie_03_test_spike_1999_nuc_800.npy')) #spike
    input_x = Input(shape = (800,))
    model_dense = dense_decoder(800)
    model_cae = CAE((64, 64, 1))

    dense_out = model_dense(input_x)
    cae_out = model_cae(dense_out)
    optimizer = keras.optimizers.Adam(lr=0.001)

    end2end_model = Model(input_x, cae_out)
    end2end_model.summary()
    end2end_model.compile(loss = 'mse', optimizer = optimizer)
    
    weight_dir = 'D:\\model_weights'
    result_dir = '../paper_results'
    end2end_model.load_weights(os.path.join(weight_dir, "movie_03_iter_1999_nuc_800_400.h5"))
    multiout_model = Model(input_x, [dense_out, cae_out])

    pred_dense, pred_cae = multiout_model.predict(spike_test)

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    np.save(os.path.join(result_dir, 'movie_03_iter_1999_nuc_nuc_800_400.npy'), pred_cae)
    #np.save(os.path.join(result_dir, 'end2end_train_cifar10_dense_iter50_only_bn.npy'), pred_dense)


# In[ ]:




