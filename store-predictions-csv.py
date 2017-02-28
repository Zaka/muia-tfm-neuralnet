import sys
import pdb
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

class timer:
    def __init__(self, msg = ""):
        self.msg = msg
        
    def __enter__(self):
        self.init_time = time.time()
        return
    
    def __exit__(self, type, value, traceback):
        self.end_time = time.time()

        print("Time elapsed with ", self.msg, " is ",
              self.end_time - self.init_time, "seconds")

class Dataset:
    def __init__(self, num_instances = 20):
        df = pd.read_csv("../muia-tfm-dataset/dataset-v2.csv")

        length = len(df)
        df = df[(length - num_instances):length]

        self.df_y = df['Y']
        self.df_X = df.drop('Y', 1)

    def get_dataset(self):
        X = self.df_X.values
        y = self.df_y.values

        X = np.expand_dims(X, axis = 2)
        y = np.expand_dims(y, axis = 1)

        return X,y
    
    def get_columns(self, columns):
        if columns == []:
            print("At least one column needed for loading dataset.")
            return

        # TODO: Un buen momento para añadir unittests

        df_X_temp = self.df_X[columns]
        
        X = df_X_temp.values
        y = self.df_y.values

        X = np.expand_dims(X, axis = 2)
        y = np.expand_dims(y, axis = 1)

        return X,y

def denormalize_market_price(normalized_market_price):
    market_price_stddev = 218.3984954558443689621
    market_price_mean = 155.3755406486043852965
    
    return (normalized_market_price *
            market_price_stddev) + market_price_mean

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

def main():
    columns = [ 'NumTransactions', 'TotalBitcoins', 'MarketPrice',
                'SP500-Close', 'Difficulty', 'BitcoinDaysDestroyed',
                'CostPerTransaction', 'TransactionFeesUSD',
                'TransactionFees', 'MedianConfirmationTime',
                'TxTradeRatio', 'EuroPriceInUSD', 'SP500-Volume',
                'TradeVolume', 'OutputVolume',
                'EstimatedTransactionVolume', 'WikipediaTrend',
                'CostPerTransactionPercent' ]
    
    dataset = Dataset(num_instances = 2673)
    X,y = dataset.get_columns(columns = columns)
    
    # Run tscv_score with those columns
    predictions = tscv_score(X, y)

    df = pd.DataFrame({"RNN-Prediction" : predictions})
    df.to_csv("output_all_variables_rnn.csv", index = False)

def tscv_score(X, y): 
    nb_samples, timesteps, input_dim = X.shape

    model = rnn_model(nb_samples = nb_samples,
                      timesteps = timesteps,
                      input_dim = input_dim)

    print("Performing Time Series Cross Validation.")


    with timer(msg = "Split"):
        tscv_split = TimeSeriesSplit(n_splits = len(X) - 1)

        predictions = []    

        # Use 1095 as the smalles partition size. This is imposed by
        # VAR because if we use more instances to create the
        # partitions for TSCV VAR doesn't work. This behaviour can be
        # caused by the correlation of the time series in their
        # beginning values.

        SMALLEST_PARTITION_SIZE = 1095
        
        for train_index, test_index in tscv_split.split(X):
            if len(train_index) >= SMALLEST_PARTITION_SIZE:
                X_train_partition = X[train_index[0]:train_index[-1] + 1]
                y_train_partition = y[train_index[0]:train_index[-1] + 1]

                X_test_partition = X[test_index[0]:test_index[-1] + 1]
                y_test_partition = y[test_index[0]:test_index[-1] + 1]

                fitted = model.fit(X_train_partition, y_train_partition,
                                   nb_epoch = 1000, verbose = 1)

                predictions.append(
                    denormalize_market_price(
                        model.predict(X_test_partition))[0][0])

    return predictions 

if __name__ == '__main__':
    main()
