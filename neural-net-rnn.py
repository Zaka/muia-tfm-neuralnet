import time
import numpy as np
np.random.seed(9000)

import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape
from keras.layers.recurrent import SimpleRNN, LSTM
from keras.wrappers.scikit_learn import KerasRegressor
from keras.regularizers import l2
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import TimeSeriesSplit

def denormalize_market_price(normalized_market_price):
    market_price_stddev = 218.3984954558443689621
    market_price_mean = 155.3755406486043852965
    
    return (normalized_market_price *
            market_price_stddev) + market_price_mean

def get_dataset(num_instances = 20, lags = 3):
    # load dataset
    dataframe = pd.read_csv("../muia-tfm-dataset/dataset-v2.csv")

    columns = [ 'MarketPrice',
                'TotalBitcoins',
                'MedianConfirmationTime',
                'CostPerTransactionPercent' ]

    length = len(dataframe)
    dataframe = dataframe[(length - num_instances):length]
    dataframe = dataframe[columns]
    
    dataset = dataframe.values
    # split into input (X) and output (Y) variables
    X = dataset[:,1:-1]
    y = dataset[:,-1]

    X = np.expand_dims(X, axis = 2)
    y = np.expand_dims(y, axis = 1)

    print("Dataset shape:")
    print("X.shape: ", X.shape)
    print("y.shape: ", y.shape)    

    return X,y

def rnn_model(nb_samples, timesteps, input_dim):
    # create model
    model = Sequential()
    model.add(SimpleRNN(1, unroll = True, activation='relu',
                        input_shape = (timesteps,
                                       1),
                        W_regularizer = l2(0.001),
                        return_sequences = False))

    model.add(Dense(1, activation = 'linear'))
    
    # Compile model
    model.compile(loss='mean_absolute_error',
                  optimizer='adam')

    return model

X,y = get_dataset(num_instances = 500,
                  lags = 3)

nb_samples, timesteps, input_dim = X.shape

t0 = time.time()

model = rnn_model(nb_samples = nb_samples,
                  timesteps = timesteps,
                  input_dim = input_dim)

print("Performing Time Series Cross Validation.")

tscv = TimeSeriesSplit(n_splits = len(X) - 1)

error_list = []

for train_index, test_index in tscv.split(X):
    X_train_partition = X[train_index[0]:train_index[-1] + 1]
    y_train_partition = y[train_index[0]:train_index[-1] + 1]
        
    X_test_partition = X[test_index[0]:test_index[-1] + 1]
    y_test_partition = y[test_index[0]:test_index[-1] + 1]
    
    fitted = model.fit(X_train_partition, y_train_partition,
                       nb_epoch = 1000)

    prediction = model.predict(X_test_partition)

    error_list.append((denormalize_market_price(y_test_partition[-1])
                      - denormalize_market_price(prediction))[0][0])
    
mae = np.mean([abs(x) for x in error_list])
    
t1 = time.time()

print("MAE: %.4f USD" %(mae))
print("It took %f seconds" % (t1 - t0))
print("Finished.")
