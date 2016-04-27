import numpy as np
import ipdb
import pandas as pd

def make_data(timesteps, items, samples_train, samples_test, t_back, t_forward,
                pattern='random', get_raw=False):

    if pattern == 'random':
        raw = np.random.random_integers(0,1,size=(timesteps,items))

        #put into Keras data format
        X_train=np.zeros((samples_train,t_back,items), dtype=np.int32)
        X_test=np.zeros((samples_test,t_back,items), dtype=np.int32)
        y_train=np.zeros((samples_train,t_forward,items), dtype=np.int32)
        y_test=np.zeros((samples_test,t_forward,items), dtype=np.int32)

        # cut the image in semi-redundant sequences of length 't_back' to use as training data
        # wrap data to deal with indexing errors caused by large t_back/t_forward
        for i in range(samples_train):
            start=np.random.randint(timesteps-1)
            indices_x=range(start, start+t_back)
            indices_y=range(start+t_back,start+t_back+t_forward)
            X_train[i]=np.take(raw,indices_x,axis=0,mode='wrap')
            y_train[i]=np.take(raw,indices_y,axis=0,mode='wrap')
        for i in range(samples_test):
            start=np.random.randint(timesteps-1) #for whole dataset
            indices_x=range(start, start+t_back)
            indices_y=range(start+t_back,start+t_back+t_forward)
            X_test[i]=np.take(raw,indices_x,axis=0,mode='wrap')
            y_test[i]=np.take(raw,indices_y,axis=0,mode='wrap')

    elif pattern == 'linear-time':
        raw = np.zeros((timesteps,items))
        for i in range(timesteps):
            for j in range(items):
                raw[i][j]=(1.0*i)/(1.0*timesteps) #+np.random.normal(0,0.3)

        #put into Keras data format
        X_train=np.zeros((samples_train,t_back,items), dtype=np.float32)
        X_test=np.zeros((samples_test,t_back,items), dtype=np.float32)
        y_train=np.zeros((samples_train,t_forward,items), dtype=np.float32)
        y_test=np.zeros((samples_test,t_forward,items), dtype=np.float32)

        # cut the image in semi-redundant sequences of length 't_back' to use as training data
        # wrap data to deal with indexing errors caused by large t_back/t_forward
        for i in range(samples_train):
            start=np.random.randint(int((timesteps-t_back-t_forward))) #ensures no wrapping, untrained columns
            indices_x=range(start, start+t_back)
            indices_y=range(start+t_back,start+t_back+t_forward)
            X_train[i]=np.take(raw,indices_x,axis=0,mode='wrap')
            y_train[i]=np.take(raw,indices_y,axis=0,mode='wrap')
        for i in range(samples_test):
            start=np.random.randint(int((timesteps-t_back-t_forward))) #ensures no wrapping, untrained columns
            indices_x=range(start, start+t_back)
            indices_y=range(start+t_back,start+t_back+t_forward)
            X_test[i]=np.take(raw,indices_x,axis=0,mode='wrap')
            y_test[i]=np.take(raw,indices_y,axis=0,mode='wrap')

    # print X_train.shape,y_train.shape
    # print X_test.shape,y_test.shape
    # print raw[0],X_train[0],y_train[0]
    # print raw[0],X_test[0],y_test[0]

    if get_raw==True:
        return X_train, y_train, X_test, y_test, raw
    return X_train, y_train, X_test, y_test

def predict_generative(model, x_test, batch_size, t_forward):
    t_back=x_test.shape[1]
    predict=np.zeros((x_test.shape[0],t_forward,x_test.shape[2]))
    #to generate next predict, take an decreasing number of timesteps from xtest data
    #and append i predictions timesteps onto the end.
    for i in range(t_forward):
        #from x_test: all sample points, t_back-i timesteps forward (min prevents index error)
        #from predict: i timesteps forward
        new_xtest=np.concatenate((x_test[:,min(i,t_back):], predict[:,:i]),axis=1)
        # ipdb.set_trace()
        predict[:,i]=model.predict(new_xtest[:,-t_back:], batch_size=batch_size) #window must slide forward
    return predict 


def evaluate_generative(predict, y_test, loss):
    if loss == 'mse':
        columns=('trial','t_forward','mse')
        df_evaluate = pd.DataFrame(index=np.arange(0,y_test.shape[0]*y_test.shape[1]),columns=columns)
        t=0 #index of dataframe
        for i in range(y_test.shape[0]): #samples
            for j in range(y_test.shape[1]):
                evaluate = np.mean((predict[i][j]-y_test[i][j])**2)
                df_evaluate.loc[t]=[i,j,evaluate]
                t+=1
    return df_evaluate


def evaluate_generative_tback(df_evaluate, predict, y_test, t_back, loss):
    if loss == 'mse':
        t=(t_back-1)*y_test.shape[0]*y_test.shape[1] #start index of new data in dataframe
        for i in range(y_test.shape[0]): #samples
            for j in range(y_test.shape[1]):
                evaluate = np.mean((predict[i][j]-y_test[i][j])**2)
                df_evaluate.loc[t]=[i,t_back,j,evaluate]
                t+=1
    return df_evaluate

def evaluate_generative_epochs(df_evaluate, predict, y_test, epochs, loss):
    if loss == 'mse':
        t=(epochs-1)*y_test.shape[0]*y_test.shape[1] #start index of new data in dataframe
        for i in range(y_test.shape[0]): #samples
            for j in range(y_test.shape[1]):
                evaluate = np.mean((predict[i][j]-y_test[i][j])**2)
                df_evaluate.loc[t]=[i,epochs,j,evaluate]
                t+=1
    return df_evaluate