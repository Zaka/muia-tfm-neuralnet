import time
import numpy as np
np.random.seed(9000)

import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.recurrent import SimpleRNN
from keras.wrappers.scikit_learn import KerasRegressor
from keras.regularizers import l2
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import TimeSeriesSplit



def get_dataset(num_instances = 20):
    # load dataset
    # dataframe = pd.read_csv("../muia-tfm-data/data-set.csv")

    global dataframe

    length = len(dataframe)
    dataframe = dataframe[(length - num_instances):length]

    dataset = dataframe.values
    # split into input (X) and output (Y) variables
    X = dataset[:,1:-1]
    Y = dataset[:,-1]

    return [X,Y]

# TODO:
def vectorize(dataset, input_length):
    result = []
    for start,end in zip(range(len(dataset) - input_length + 1),
                      range(input_length, len(dataset) + 1)):
        result.append(X[start:end])

    return np.array(result)

def rnn_model(input_length, input_dim):
    # create model
    model = Sequential()
    model.add(SimpleRNN(1, activation='relu',
                        input_shape = (input_length,
                                       input_dim),
                        W_regularizer = l2(0.001)))

    model.add(Dense(input_dim))
    model.add(Activation('linear'))
    
    # Compile model
    model.compile(loss='mean_squared_error',
                  optimizer='adam')
    return model

dataframe = pd.read_csv("../muia-tfm-data/data-set.csv")

input_length = 10

X,y = get_dataset(num_instances = input_length)

input_dim = len(X[0])

t0 = time.time()

model = rnn_model(input_length = input_length,
                  input_dim = input_dim)

# mlp = KerasRegressor(build_fn = rnn_model(input_dim = input_dim,
#                                           input_length = input_length),
#                      nb_epoch=10, batch_size=5,
#                      verbose=0)

print("Performing Time Series Cross Validation.")

tscv = TimeSeriesSplit(n_splits = len(X) - 1)

# model.fit(X, y, nb_epoch = 1)

# results = cross_val_score(mlp, X, Y, cv=tscv)
# t1 = time.time()
# mae = mean_absolute_error(Y[1:], results)

# market_price_stddev = 218.3984954558443689621
# market_price_mean = 155.3755406486043852965
# real_mae = (mae * market_price_stddev) + market_price_mean
# print("Normalized MAE: %.4f" % (mae))
# print("\"Denormalized\" MAE: %.4f USD" %(real_mae))
# print("It took %f seconds" % (t1 - t0))
# print("Finished.")
