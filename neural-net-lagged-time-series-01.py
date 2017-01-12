import numpy as np
import pandas as pd
import time

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from keras.regularizers import l2
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

# Visualization
from keras.utils.visualize_util import plot

def get_lagged_dataframe(df, lags = 1):
    lagged_tss = []
    
    for ts in df:
        lagged_tss.append(get_lagged_ts(df[ts], lags))

    return np.hstack(tuple(lagged_tss))
        
def get_lagged_ts(ts, lags = 1):
    temp = []

    for i in range(lags):
        temp.append(ts[i:i-lags])

    temp.append(ts[lags:])

    return np.array(temp).T

# load dataset
dataframe = pd.read_csv("../muia-tfm-data/data-set.csv")

num_instances = 500

length = len(dataframe)
dataframe = dataframe[(length - num_instances):length]

df_x = dataframe.drop(['Index','Y'], 1)
# DEBUG
# df_x = dataframe[['EuroPriceInUSD', 'MarketPrice', 'MedianConfirmationTime']]

lags = 3

X = get_lagged_dataframe(df_x, lags = lags)

Y = dataframe['Y'].values[lags:]

seed = 7
np.random.seed(seed)

num_inputs = len(X[0])

print("Building the model.")
# Define NNs structure
def deeper_model():
    # create model
    model = Sequential()
    model.add(Dense(num_inputs * 2, input_dim = num_inputs,
                    init='normal',
                    W_regularizer = l2(0.001),
                    activation='relu'))
    model.add(Dense(num_inputs * 2, init='normal',
                    activation='relu',
                    W_regularizer = l2(0.001)))
    model.add(Dense(1, activation='linear',
                    W_regularizer = l2(0.001)))
    # Compile model
    model.compile(loss='mean_squared_error',
                  optimizer='adam')
    return model

t0 = time.time()
mlp = KerasRegressor(build_fn=deeper_model,
                     nb_epoch=1000, batch_size=32,
                     verbose=0)

print("Performing Time Series Cross Validation.")

tscv = TimeSeriesSplit(n_splits = len(X) - 1)
results = cross_val_score(mlp, X, Y, cv=tscv)
t1 = time.time()
mae = mean_absolute_error(Y[1:], results)

market_price_stddev = 218.3984954558443689621
market_price_mean = 155.3755406486043852965
real_mae = (mae * market_price_stddev) + market_price_mean
print("Normalized MAE: %.4f" % (mae))
print("\"Denormalized\" MAE: %.4f USD" %(real_mae))
print("It took %f seconds" % (t1 - t0))
print("Finished.")
