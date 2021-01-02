'''Uses Keras to train and test a 2dconvlstm on parameterized CHEC waveform data.
Written by S.T. Spencer 8/8/2018'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import h5py
import keras
import os
import tempfile
import sys
from keras.utils import HDF5Matrix
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Conv2D, ConvLSTM2D, MaxPooling2D, BatchNormalization, Conv3D, GlobalAveragePooling3D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import AveragePooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers import Input, GaussianNoise
from keras.models import Model
from keras.layers import concatenate
from keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping
from keras import regularizers
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy.ma as ma
from matplotlib.pyplot import cm
from sklearn.preprocessing import scale
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy.ma as ma
from matplotlib.pyplot import cm
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from keras.metrics import binary_accuracy
from sklearn.metrics import roc_curve, auc
from net_utils import *
import time

plt.ioff()
# os.environ['CUDA_VISIBLE_DEVICES'] = '' # Uncomment this to use cpu
# rather than gpu.

# Finds all the hdf5 files in a given directory
global onlyfiles
onlyfiles = sorted(glob.glob('/mnt/extraspace/exet4487/diffuserun1/*.hdf5'))
runname = 'diffuserun_allparamsES'
shilonflag = False #Whether to sort telescopes in the LSTM using integrated charge or median arrival time
global Trutharr
Trutharr = []
Train2=[]
# Find true event classes for test data to construct confusion matrix.
for file in onlyfiles[140:239]:
    try:
        inputdata = h5py.File(file, 'r')
    except Exception:
        print(file)
    labelsarr = np.asarray(inputdata['event_label'][:])
    for value in labelsarr:
        Trutharr.append(value)
    inputdata.close()

for file in onlyfiles[:140]:
    try:
        inputdata = h5py.File(file, 'r')
    except Exception:
        print(file)
    labelsarr = np.asarray(inputdata['event_label'][:])
    for value in labelsarr:
        Train2.append(value)
    inputdata.close()

print('lentruth', len(Trutharr))
print('lentrain',len(Train2))
# Define model architecture.

def get_reg_loss(reg_layers):
    def reg_loss_term(y_true, y_pred):
        x=tf.add_n([r.losses[0] for r in reg_layers])
        y=keras.losses.categorical_crossentropy(y_true,y_pred,from_logits=False)
        tl=x+y #Include regularization in loss
        met=1.0/(0.7-0.8*tl)
        return met
    return reg_loss_term
'''
def custom_metric(y_true,y_pred):
    loss=keras.losses.categorical_crossentropy(y_true,y_pred,from_logits=False)
    z=1.0/(0.7-0.8*loss)
    return z

def custom_metric(y_true,y_pred):                                                                                                                                                                                                                                                                                           
    loss=keras.losses.categorical_crossentropy(y_true,y_pred,from_logits=False)                                                                                                                                                                                                                                             
    #z=1.0/(0.7-0.8*loss)                                                                                                                                                                                                                                                                                                    
    return loss
'''
strategy=tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
with strategy.scope():
    model = Sequential()
    l1=ConvLSTM2D(filters=30, kernel_size=(3, 3),input_shape=(None, 32, 32, 8),padding='same', return_sequences=True,recurrent_regularizer=keras.regularizers.l2())
    model.add(l1)
    l2=Dropout(0.3)
    model.add(l2)
    l3=BatchNormalization()
    model.add(l3)
    l4=ConvLSTM2D(filters=30, kernel_size=(3, 3),padding='same', return_sequences=True,recurrent_regularizer=keras.regularizers.l2())
    model.add(l4)
    l5=Dropout(0.3)
    model.add(l5)
    l6=BatchNormalization()
    model.add(l6)
    l7=ConvLSTM2D(filters=30, kernel_size=(3, 3),padding='same', return_sequences=True)
    model.add(l7)
    l8=Dropout(0.3)
    model.add(l8)
    l9=BatchNormalization()
    model.add(l9)
    l10=ConvLSTM2D(filters=30, kernel_size=(3, 3),padding='same', return_sequences=True)
    model.add(l10)
    l11=Dropout(0.3)
    model.add(l11)
    l12=BatchNormalization()
    model.add(l12)
    l13=ConvLSTM2D(filters=30, kernel_size=(3, 3),padding='same', return_sequences=True)
    model.add(l13)
    l14=Dropout(0.5)
    model.add(l14)
    l15=BatchNormalization()
    model.add(l15)
    l16=GlobalAveragePooling3D()
    model.add(l16)
    l17=Dense(3, activation='softmax')
    model.add(l17)
    opt = keras.optimizers.Adadelta()
    # Compile the model
    layers=[l1,l4]
    lr_metric=get_reg_loss(layers)
    model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['categorical_accuracy',lr_metric])

early_stop = EarlyStopping(monitor='loss',
                           min_delta=0,
                           patience=3,
                           verbose=1,
                           mode='min',
                           restore_best_weights=True)

custom_stop = EarlyStopping(monitor='reg_loss_term',
                            min_delta=0,
                            patience=3,
                            verbose=1,
                            mode='min',
                            restore_best_weights=True)

# Code for ensuring no contamination between training and test data.
print(model.summary())

t1=time.time()
# Train the network
history = model.fit(
    generate_training_sequences(onlyfiles,
        50,
        'Train',shilonflag),
    steps_per_epoch=21951,
    epochs=30,
    verbose=2,
    callbacks=[early_stop,custom_stop],
    use_multiprocessing=False,
    shuffle=False)

model.save('/users/exet4487/Models/'+runname+'model.hdf5')
# Plot training accuracy/loss.
fig = plt.figure()
plt.subplot(2, 1, 1)
print(history.history)
plt.plot(history.history['categorical_accuracy'], label='Train')
# plt.plot(history.history['val_binary_accuracy'],label='Validation')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='lower right')

plt.subplot(2, 1, 2)
plt.plot(history.history['loss'], label='Train')
# plt.plot(history.history['val_loss'],label='Validation')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')

plt.tight_layout()

plt.savefig('/users/exet4487/Figures/'+runname+'trainlog.png')

# Test the network
print('Predicting')
pred = model.predict(
    generate_training_sequences(onlyfiles,
        50,
        'Test',shilonflag),
    verbose=1,
     use_multiprocessing=False,
    steps=14418)
np.save('/users/exet4487/predictions/'+runname+'_predictions.npy', pred)

# Check for cross-contamination
#print(np.shape(trainevents), np.shape(testevents), np.shape(validevents))
'''
# Code to check for event contamination, only use with single simtel file.
print(
    'Events both in training and testing', list(
        set(trainevents) & set(testevents)), list(

            set(train2) & set(test2)))
print(
    'Events both in training and validation', list(
        set(trainevents) & set(validevents)), list(
            set(train2) & set(valid2)))
print(
    'Events both in validation and testing', list(
        set(validevents) & set(testevents)), list(
            set(valid2) & set(test2)))
print(len(np.unique(trainevents)),
      len(np.unique(testevents)),
      len(np.unique(validevents)))
'''
print('Evaluating')

score = model.evaluate(
    generate_training_sequences(onlyfiles,
        50,
        'Test',shilonflag),
    use_multiprocessing=False,
    steps=14418)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Plot confusion matrix
t2=time.time()

print('Time Taken',t2-t1)
print(get_confusion_matrix_one_hot(runname,pred, Trutharr))
