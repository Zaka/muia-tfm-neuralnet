import numpy
import pandas
import time

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

# Visualization
from keras.utils.visualize_util import plot

# load dataset
dataframe = pandas.read_csv("../muia-tfm-data/data-set.csv")

dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,1:-1]
Y = dataset[:,-1]

seed = 7
numpy.random.seed(seed)

print("Building the model.")
# Define NNs structure
def wider_model():
    # create model
    model = Sequential()
    model.add(Dense(20, input_dim=18, init='normal', activation='relu'))
    model.add(Dense(1, init='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

t0 = time.time()
mlp = KerasRegressor(build_fn=wider_model,
                     nb_epoch=100, batch_size=5,
                     verbose=0)

print("Performing Time Series Cross Validation.")

tscv = TimeSeriesSplit(n_splits = len(X) - 1)
results = cross_val_score(mlp, X, Y, cv=tscv)
t1 = time.time()
mae = mean_absolute_error(Y[1:], results)
real_mae = (mae * 218.3984954558443689621) + 155.3755406486043852965
print("Normalized MAE: %.4f" % (mae))
print("\"Denormalized\" MAE: %.4f USD" %(real_mae))
print("It took %f seconds" % (t1 - t0))
print("Finished.")
