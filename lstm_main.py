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
timesteps=100 #length of data (columns)
items=20 #height of data (rows)
t_back=5 #length of sequence to input to LSTM for training/prediction
t_forward=50 #length of sequence for LSTM to predict
train_on=0 #using the column that is n-1 timesteps ahead of the training data as y_train
batch_size=16
epochs=200
lstm_size=64
samples_train=1000
samples_test=500
optimizer='adam' #rmsprop, sgd, adam
final_activation='hard_sigmoid'
loss='mse' #mse, categorical_crossentropy
pattern='linear-time'
filename='lstm_main'

params={
    'timesteps':timesteps, 
    'items':items, 
    't_back':t_back,
    't_forward':t_forward,
    'train_on':train_on,
    'batch_size':batch_size,
    'epochs':epochs,
    'lstm_size':lstm_size,
    'samples_train':samples_train,
    'samples_test':samples_test,
    'optimizer':optimizer,
    'final_activation':final_activation,
    'loss':loss,
    'pattern':pattern,
    'filename':filename,
}

print "Generating Data..."
X_train, y_train, X_test, y_test = make_data(timesteps, items, samples_train,
                                                samples_test, t_back, t_forward, pattern)


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
df_evaluate = evaluate_generative(predict, y_test, loss) #returns pandas dataframe

print 'Exporting Data...'
root=os.getcwd()
os.chdir(root+'/data/')
addon=str(np.random.randint(0,100000))
fname=filename+addon
df_evaluate.to_pickle(fname+'_data.pkl')
param_df=pd.DataFrame([params])
param_df.reset_index().to_json(fname+'_params.json',orient='records')

print 'Plotting...'
sns.set(context='talk',style='white')
figure1, (ax1, ax2) = plt.subplots(2, 1)
ax1.imshow(predict[0,:t_forward].T, cmap='gray', interpolation='none',vmin=0, vmax=1)
ax2.imshow(y_test[0,:t_forward].T, cmap='gray', interpolation='none',vmin=0, vmax=1)
ax1.set(xlabel='time',ylabel='',title='predictions',xticks=[],yticks=[])
ax2.set(xlabel='time',ylabel='',title='correct',xticks=[],yticks=[])
figure1.savefig(fname+'_example.png')
plt.show()

#search for an appropriate predict/y_test pair
# for i in range(predict.shape[0]):
#     f, (ax1, ax2) = plt.subplots(2, 1)
#     ax1.imshow(predict[i,:t_forward].T, cmap='gray', interpolation='none',vmin=0, vmax=1)
#     ax2.imshow(y_test[i,:t_forward].T, cmap='gray', interpolation='none',vmin=0, vmax=1)
#     ax1.set(xlabel='',ylabel='',title='predictions',xticks=[],yticks=[])
#     ax2.set(xlabel='time',ylabel='',title='correct',xticks=[],yticks=[])
#     plt.show()

figure2, ax3, = plt.subplots(1, 1)
sns.tsplot(data=df_evaluate, time="t_forward", value="mse", unit="trial", ax=ax3)
figure2.savefig(fname+'_error.png')
plt.show()

#pretty colors, same data as figure2
figure3, ax4, = plt.subplots(1, 1)
sns.barplot(x="t_forward",y="mse",data=df_evaluate,ax=ax4)
figure3.savefig(fname+'_error_color.png')
plt.show()