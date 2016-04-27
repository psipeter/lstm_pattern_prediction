import numpy as np
import ipdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from helper import *
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM

#Parameters
timesteps=20 #length of data (columns)
items=10 #height of data (rows)
t_back=5 #length of sequence to input to LSTM for training/prediction
t_forward=5 #length of sequence for LSTM to predict
train_on=0 #using the column that is n-1 timesteps ahead of the training data as y_train
batch_size=16
epochs=20
lstm_size=64
samples_train=1000
samples_test=500
optimizer='adam' #rmsprop, sgd, adam
final_activation='hard_sigmoid'
loss='mse' #mse, categorical_crossentropy
pattern='random'
filename='plotter'

root=os.getcwd()
os.chdir(root+'/data/')
addon=str(np.random.randint(0,100000))
fname=filename+addon

print "Generating Data..."
X_train, y_train, X_test, y_test, raw = make_data(timesteps, items, samples_train,
                                                samples_test, t_back, t_forward, pattern, True)

print "Building LSTM..."
model = Sequential()
model.add(LSTM(lstm_size, input_shape=(t_back, items)))
model.add(Dropout(0.25))
model.add(Dense(items))
model.add(Dropout(0.25))
model.add(Activation(final_activation))
model.compile(loss=loss, optimizer=optimizer)


print 'Training...'
model.fit(X_train, y_train[:,train_on], batch_size=batch_size, nb_epoch=epochs,
            validation_data=(X_test, y_test[:,train_on]))


print 'Predicting and Evaluating...'
predict = predict_generative(model,X_test,batch_size,t_forward)

print 'Plotting...'
sns.set(context='talk',style='white')
figure1, (ax1, ax2) = plt.subplots(2, 1)
ax1.imshow(raw.T, cmap='gray', interpolation='none',vmin=0, vmax=1)
ax2.imshow(X_test[0,0:t_back].T, cmap='gray', interpolation='none',vmin=0, vmax=1)
ax1.set(xlabel='time',ylabel='data',title='raw dataset',xticks=[],yticks=[])
ax2.set(xlabel='time',ylabel='data',title='train/test subset',xticks=[],yticks=[])
figure1.savefig(fname+'_x.png')

figure2, (ax3, ax4) = plt.subplots(1, 2)
ax3.imshow(predict[0,:t_forward].T, cmap='gray', interpolation='none',vmin=0, vmax=1)
ax4.imshow(y_test[0,:t_forward].T, cmap='gray', interpolation='none',vmin=0, vmax=1)
ax3.set(xlabel='time',ylabel='',title='predictions',xticks=[],yticks=[])
ax4.set(xlabel='time',ylabel='',title='correct',xticks=[],yticks=[])
figure2.savefig(fname+'_y.png')