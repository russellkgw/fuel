from keras.models import Sequential
from keras.layers import LSTM, GRU
from keras.layers import Dense
from keras import optimizers
from pandas import Series
from sklearn.preprocessing import MinMaxScaler
import numpy as np


from data_connector import DataConnector

data_conn = DataConnector()
# fuel_data = data_conn.fuel_prices_dates()
fuel_data = data_conn.fuel_month_changes()
data = []

# exr_data = np.array(data_conn.exchange_rates(fuel_data[0].date))
# oil_data = np.array(data_conn.oil_prices(fuel_data[0].date))
# x = np.column_stack((exr_data, oil_data))
# x = x.reshape(1, 62, 2)
# y = np.array([fuel_data[0].price]).reshape(1, 1)

# import pdb; pdb.set_trace()

for fp in fuel_data:
    exr_data = np.array(data_conn.exchange_rates(fp.date))
    oil_data = np.array(data_conn.oil_prices(fp.date))
    x = np.column_stack((exr_data, oil_data))
    x = x.reshape(1, 62, 2)
    y = np.array([fp.price]).reshape(1, 1)
    
    data.append({'date': fp.date, 'y': y, 'x': x})  # y reshape(1, 1)

# import pdb; pdb.set_trace()
model = Sequential()
model.add(LSTM(240, return_sequences=True, input_shape=(62, 2)))  # LSTM GRU  return_state=True
model.add(LSTM(120, return_sequences=True))  # activation='tanh', recurrent_activation='hard_sigmoid'
model.add(LSTM(60))
model.add(Dense(60, activation='relu'))
model.add(Dense(1, activation='linear'))
# sgd = optimizers.SGD(lr=0.1)  # , decay=1e-6, momentum=0.9, nesterov=True
model.compile(loss='mse', optimizer='sgd', metrics=['acc'])  # adagrad adam sgd rmsprop
print(model.summary())

for e in range(100):
    print('Epoch: ' + str(e))
    for d in data:
        model.fit(d['x'], d['y'], epochs=1, verbose=2)
