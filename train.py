import numpy as np
import tensorflow.keras as keras
import os
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
import tensorflow as tf
import pdb
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
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
    model.add(BatchNormalization(name = 'cae_out'))
    return model
if __name__ == '__main__':




    data_dir = './data'
    
    img_train = np.load(os.path.join(data_dir, 'movie_03_train_pic_1999_VISp_800.npy')) #image
    img_test = np.load(os.path.join(data_dir, 'movie_03_test_pic_1999_VISp_800.npy')) #image
    spike_train = np.load(os.path.join(data_dir, 'movie_03_train_spike_1999_VISp_800.npy')) #spike
    spike_test = np.load(os.path.join(data_dir, 'movie_03_test_spike_1999_VISp_800.npy')) #spike
    input_x = Input(shape = (800,))
    model_dense = dense_decoder(800)
    model_cae = CAE((64, 64, 1))

    dense_out = model_dense(input_x)
    cae_out = model_cae(dense_out)
    optimizer = keras.optimizers.Adam(lr=0.001)
    end2end_model = Model(input_x, cae_out)
    end2end_model.summary()
    end2end_model.compile(loss = 'mse', optimizer = optimizer)

    weight_dir = './model_weights'
    result_dir = './paper_results'

    end2end_model.fit(spike_train, img_train, batch_size = 16, epochs = 400, validation_data = (spike_test, img_test))

    if not os.path.exists(weight_dir):
        os.mkdir(weight_dir)
    end2end_model.save_weights(os.path.join(weight_dir, 'movie_03_iter_1999_VISp_800_400.h5'))

    multiout_model = Model(input_x, [dense_out, cae_out])

    pred_dense, pred_cae = multiout_model.predict(spike_test)

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    np.save(os.path.join(result_dir, 'movie_03_iter_1999_VISp_800_400.npy'), pred_cae)