import numpy
import pandas
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

# load dataset
dataframe = pandas.read_csv("../muia-tfm-data/data-set.csv")

# length = len(dataframe)
# dataframe = dataframe[(length - 20):length]

dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,1:-1]
Y = dataset[:,-1]

seed = 7
numpy.random.seed(seed)

print("Building the model.")
# Define NNs structure
def deeper_model():
    # create model
    model = Sequential()
    model.add(Dense(10, input_dim=18,
                    init='normal',
                    W_regularizer = l2(0.001),
                    activation='relu'))
    model.add(Dense(10, init='normal',
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
