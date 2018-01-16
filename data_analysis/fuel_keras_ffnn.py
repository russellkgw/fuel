import numpy as np
import pandas as pd
import tensorflow
import keras

from data_connector import DataConnector
from keras.models import Sequential
from keras.layers import Dense, Activation, SimpleRNN, TimeDistributed

# Data
# Fuel data
data_conn = DataConnector()

exchange = data_conn.exchange_month_changes()
oil = data_conn.oil_month_changes()
exchange_future = data_conn.exchange_future_month_changes()
oil_future = data_conn.oil_future_month_changes()
fuel = data_conn.fuel_month_changes()

data_size_trn = 150
data_size_tst = 165

training_set = pd.DataFrame({'exchange_rate': exchange[:data_size_trn], 'oil_price': oil[:data_size_trn],
                             'exchange_future': exchange_future[:data_size_trn], 'oil_future': oil_future[:data_size_trn], 'fuel': fuel[:data_size_trn]})

data = training_set.values
labels = np.array(fuel[:data_size_trn])


# test_set = pd.DataFrame({'exchange_rate': exchange[data_size_trn:data_size_tst], 'oil_price': oil[data_size_trn:data_size_tst],
#                          'exchange_future': exchange_future[data_size_trn:data_size_tst],
#                          'oil_future': oil_future[data_size_trn:data_size_tst],
#                          'fuel_price': fuel[data_size_trn:data_size_tst]})

# import pdb; pdb.set_trace()

# FFNN

# model = Sequential()

# model.add(Dense(12, activation='relu', input_dim=4))

# model.add(Dense(6, activation='relu'))

# model.add(Dense(1, activation='sigmoid')) # linear # sigmoid
# model.compile(optimizer='sgd',
#               loss='mean_squared_error',
#               metrics=['accuracy'])

# # # Train the model
# model.fit(data, labels, epochs=10)


# RRNN

# model = Sequential()

# model.add(TimeDistributed(Dense(4, input_dim=4)))

# model.add(SimpleRNN(units=4, activation='relu', return_sequences=True, input_shape=(4,4)))

# model.add(SimpleRNN(units=4, activation='relu', return_sequences=False))

# model.add(Dense(1, activation='sigmoid')) # linear # sigmoid
# model.compile(optimizer='sgd',
#               loss='mean_squared_error',
#               metrics=['accuracy'])

# # # Train the model
# model.fit(data, labels, epochs=10, batch_size=4)