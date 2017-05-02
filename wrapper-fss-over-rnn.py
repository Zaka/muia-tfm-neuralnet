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

def wrapper_fss():
    columns = [ 'NumTransactions', 'TotalBitcoins', 'MarketPrice',
                'SP500-Close', 'Difficulty', 'BitcoinDaysDestroyed',
                'CostPerTransaction', 'TransactionFeesUSD', 'TransactionFees',
                'MedianConfirmationTime', 'TxTradeRatio', 'EuroPriceInUSD',
                'SP500-Volume', 'TradeVolume', 'OutputVolume',
                'EstimatedTransactionVolume', 'WikipediaTrend',
                'CostPerTransactionPercent' ]

    dataset = Dataset(num_instances = 2673)

    selected_vars = ['MarketPrice']
    selected_vars_score = sys.maxsize
    not_selected_vars = columns
    not_selected_vars.remove('MarketPrice')
    
    with timer(msg = "WFSS"):
        while not_selected_vars:
            score = sys.maxsize
            selected_var = ''

            for col in not_selected_vars:
                # pdb.set_trace()
                
                candidate_columns = selected_vars.copy()
                candidate_columns.append(col)
                
                X,y = dataset.get_columns(columns = candidate_columns)

                candidates_score = tscv_score(X,y)

                print("Candidate columns: ", candidate_columns)
                print("Candidates score: ", candidates_score)
                
                if candidates_score < score:
                    score = candidates_score
                    selected_var = col

            if score < selected_vars_score:
                selected_vars.append(selected_var)
                not_selected_vars.remove(selected_var)
                selected_vars_score = score

                print("Selected vars: ", selected_vars)
                print("Current MAE: ", selected_vars_score)
            else:
                break

    print("Selected features: ", selected_vars)
    print("Selected features score: ", selected_vars_score)
    print("Finished.")

def tscv_score(X, y):
    nb_samples, timesteps, input_dim = X.shape

    model = rnn_model(nb_samples = nb_samples,
                      timesteps = timesteps,
                      input_dim = input_dim)

    print("-"*50)
    print("batch_input_shape: ", model.input_shape)
    print("nb_samples: ", nb_samples)
    print("timesteps: ", timesteps)
    print("input_dim: ", input_dim)
    print("-"*50)
    
    print(model.summary())
    # pdb.set_trace()
    
    print("Performing Time Series Cross Validation.")

    with timer(msg = "Split"):
        tscv_split = TimeSeriesSplit(n_splits = len(X) - 1)

        error_list = []

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
                                   nb_epoch = 1, verbose = 0)

                prediction = model.predict(X_test_partition, verbose = 0)

                error_list.append((denormalize_market_price(y_test_partition[-1][0])
                                   - denormalize_market_price(prediction))[0][0])

                mae = np.mean([abs(x) for x in error_list])
    
    return mae

if __name__ == '__main__':
    wrapper_fss()
