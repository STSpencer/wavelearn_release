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
from keras.backend.tensorflow_backend import set_session
from keras.utils import plot_model
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
from deepexplain.tensorflow import DeepExplain
from sklearn.metrics import roc_curve, auc
from net_utils import *
import time

plt.ioff()
# os.environ['CUDA_VISIBLE_DEVICES'] = '' # Uncomment this to use cpu
# rather than gpu.

# Finds all the hdf5 files in a given directory
global onlyfiles
onlyfiles = sorted(glob.glob('/store/spencers/Data/pointrun3/*.hdf5'))
runname = 'pointrun_chargetime'
shilonflag = False #Whether to sort telescopes in the LSTM using integrated charge or median arrival time
global Trutharr
Trutharr = []
Train2=[]
# Find true event classes for test data to construct confusion matrix.
for file in onlyfiles[140:239]:
    inputdata = h5py.File(file, 'r')
    labelsarr = np.asarray(inputdata['event_label'][:])
    for value in labelsarr:
        Trutharr.append(value)
    inputdata.close()

for file in onlyfiles[:140]:
    inputdata = h5py.File(file, 'r')
    labelsarr = np.asarray(inputdata['event_label'][:])
    for value in labelsarr:
        Train2.append(value)
        inputdata.close()

print('lentruth', len(Trutharr))
print('lentrain',len(Train2))
# Define model architecture.

model = Sequential()
model.add(ConvLSTM2D(filters=30, kernel_size=(3, 3),
                     input_shape=(None, 32, 32, 1),
                     padding='same', return_sequences=True,recurrent_regularizer=keras.regularizers.l2()))
model.add(Dropout(0.3))
model.add(BatchNormalization())

model.add(ConvLSTM2D(filters=30, kernel_size=(3, 3),
                     padding='same', return_sequences=True,recurrent_regularizer=keras.regularizers.l2()))
model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(ConvLSTM2D(filters=30, kernel_size=(3, 3),
                     padding='same', return_sequences=True))
model.add(Dropout(0.3))
model.add(BatchNormalization())

model.add(ConvLSTM2D(filters=30, kernel_size=(3, 3),
                     padding='same', return_sequences=True))
model.add(Dropout(0.3))
model.add(BatchNormalization())

model.add(ConvLSTM2D(filters=30, kernel_size=(3, 3),
                     padding='same', return_sequences=True))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(GlobalAveragePooling3D())
model.add(Dense(3, activation='softmax'))
opt = keras.optimizers.Adadelta()

# Compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer=opt,
    metrics=['categorical_accuracy'])

early_stop = EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=10,
    verbose=1,
    mode='auto')

# Code for ensuring no contamination between training and test data.
print(model.summary())

plot_model(
    model,
    to_file='/home/spencers/Figures/'+runname+'_model.png',
    show_shapes=True)

t1=time.time()
# Train the network
history = model.fit_generator(
    generate_training_sequences(onlyfiles,
        200,
        'Train',shilonflag),
    steps_per_epoch=5416,
    epochs=30,
    verbose=1,
    use_multiprocessing=False,
    shuffle=False)

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

plt.savefig('/home/spencers/Figures/'+runname+'trainlog.png')

# Test the network
print('Predicting')
pred = model.predict_generator(
    generate_training_sequences(onlyfiles,
        200,
        'Test',shilonflag),
    verbose=0,
     use_multiprocessing=False,
    steps=3587)
np.save('/home/spencers/predictions/'+runname+'_predictions.npy', pred)

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

score = model.evaluate_generator(
    generate_training_sequences(onlyfiles,
        200,
        'Test',shilonflag),
    use_multiprocessing=False,
    steps=3587)
model.save('/home/spencers/Models/'+runname+'model.hdf5')

print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Plot confusion matrix
t2=time.time()

print('Time Taken',t2-t1)
print(get_confusion_matrix_one_hot(runname,pred, Trutharr))
