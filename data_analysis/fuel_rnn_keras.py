from keras.models import Sequential
from keras.layers import LSTM, GRU
from keras.layers import Dense
from keras import optimizers
# from pandas import Series
# from sklearn.preprocessing import MinMaxScaler
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

exr_min, exr_max = 0.0, 0.0
oil_min, oil_max = 0.0, 0.0
fp_min, fp_max = 0.0, 0.0


for fp in fuel_data:
    exr_data = data_conn.exchange_rates(fp.date)
    for e in exr_data:
        if e < exr_min:
            exr_min = e
        if e > exr_max:
            exr_max = e

    oil_data = data_conn.oil_prices(fp.date)
    for o in oil_data:
        if o < oil_min:
            oil_min = o
        if o > oil_max:
            oil_max = o

    if fp.price < fp_min:
            fp_min = fp.price
    if fp.price > fp_max:
        fp_max = fp.price

    data.append({'date': fp.date, 'y': [fp.price], 'exr': exr_data, 'oil': oil_data})


# def norm_data(x, min, max):
#     return (x - min) / (max - min)


def norm_array(input, min, max):
    res = []
    for i in input:  # range(len(input)):
        res.append((i - min) / (max - min))  # norm_data(input[i], min, max)
    return np.array(res)


feed_data = []
for d in data:
    y = norm_array(d['y'], fp_min, fp_max)
    x1 = norm_array(d['exr'], exr_min, exr_max)
    x2 = norm_array(d['oil'], oil_min, oil_max)

    x = np.column_stack((x1, x2))
    x = x.reshape(1, 62, 2)
    y = np.array(y).reshape(1, 1)
    
    # if len(feed_data) == 0:
    #     print(str(d['exr']))
    #     print(str(x1))

    feed_data.append({'date': d['date'], 'x': x, 'y': y})


print('fp min: ' + str(fp_min)), print('fp max: ' + str(fp_max))
print('ex min: ' + str(exr_min)), print('ex max: ' + str(exr_max))
print('o min: ' + str(oil_min)), print('o max: ' + str(oil_max))

# import pdb; pdb.set_trace()
model = Sequential()  # dropout=0.1, recurrent_dropout=0.1
model.add(LSTM(240, activation='relu', return_sequences=True, input_shape=(62, 2)))  # LSTM GRU  return_state=True
model.add(LSTM(120, activation='relu', return_sequences=True))  # activation='tanh', recurrent_activation='hard_sigmoid'
model.add(LSTM(60, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(1, activation='linear'))
# sgd = optimizers.SGD(lr=0.1)  # , decay=1e-6, momentum=0.9, nesterov=True
model.compile(loss='mse', optimizer='sgd', metrics=['acc'])  # adagrad adam sgd rmsprop
# print(model.summary())

for e in range(100):
    print('Epoch: ' + str(e))
    for d in feed_data:
        model.fit(d['x'], d['y'], epochs=1, verbose=2)
