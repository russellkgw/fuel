from keras.models import Sequential
from keras.layers import LSTM, GRU
from keras.layers import Dense, Dropout
from keras import optimizers
import numpy as np

from data_connector import DataConnector

SEQ_LENGTH = 60  # 20 40 60
PRE_SET = 5
PRE_SET_VAL = 0.0

data_conn = DataConnector()
x_train_array, y_train_array = data_conn.fuel_prices_dates(start_date='2004-04-03', data_set='training', seq_length=SEQ_LENGTH, pre_set=PRE_SET, pre_set_val=PRE_SET_VAL)  # start_date='2004-04-03'    None is all
x_test_array, y_test_array = data_conn.fuel_prices_dates(start_date='2004-04-03', data_set='testing', seq_length=SEQ_LENGTH, pre_set=PRE_SET, pre_set_val=PRE_SET_VAL)  # start_date='2004-04-03'    None is all
x_validate_array, y_validate_array = data_conn.fuel_prices_dates(start_date='2004-04-03', data_set='validation', seq_length=SEQ_LENGTH, pre_set=PRE_SET, pre_set_val=PRE_SET_VAL)  # start_date='2004-04-03'    None is all

# import pdb; pdb.set_trace()

INPUT_DIM = len(x_train_array[0])
DROP = 0.1

# import pdb; pdb.set_trace()
model = Sequential()  # dropout=0.1, recurrent_dropout=0.1
model.add(Dense(124, activation='relu', input_dim=INPUT_DIM))
# model.add(Dropout(DROP))
model.add(Dense(248, activation='relu'))
# model.add(Dropout(DROP))
model.add(Dense(124, activation='relu'))
# model.add(Dropout(DROP))
model.add(Dense(62, activation='relu'))
# model.add(Dropout(DROP))
model.add(Dense(31, activation='relu'))
# model.add(Dropout(DROP))
model.add(Dense(1, activation='linear'))
# sgd = optimizers.SGD(lr=0.1)  # , decay=1e-6, momentum=0.9, nesterov=True
model.compile(loss='mse', optimizer='sgd')  # adagrad adam sgd rmsprop     , metrics=['acc']
# print(model.summary())

# print('size of x: ' + str(len(x_array)))

# import pdb; pdb.set_trace()
PRECISION = 6
# SPLIT = 140  # 240

### FIT ###
print(' ### Fit the model ### ')
x_fit = x_train_array # x_array #[:SPLIT]
y_fit = y_train_array # y_array #[:SPLIT]
EPOCHS = 10
epoch_loss_list = []
epoch_train_loss_avg = 0.0
for e in range(EPOCHS):
    epoch_loss = 0.0
    for i in range(len(x_fit)):
        temp_x = x_fit[i].reshape(1, INPUT_DIM)
        temp_y = y_fit[i].reshape(1, 1)
        res = model.fit(temp_x, temp_y, verbose=0, epochs=1)
        epoch_loss += res.history['loss'][0]

    print('Training EPOCH: ' + str(e))
    epoch_loss_list.append(epoch_loss/len(x_fit))
    epoch_train_loss_avg += (epoch_loss/len(x_fit)) / EPOCHS

print('AVERAGE TRAIN LOSS: ' + str(round(epoch_train_loss_avg, PRECISION)))
# print('Epoch Losses:')
# for e in range(len(epoch_loss_list)):
#     print('EPOCH: ' + str(e + 1) + ' AVG LOSS: ' + str(round(epoch_loss_list[e], PRECISION)))


### EVALUATE ###

# print(' ### Evaluate the model ### ')
# x_test = x_test_array
# y_test = y_test_array
# epoch_fit_loss_avg = 0.0
# for i in range(len(x_test)):
#     temp_x = x_test[i].reshape(1, INPUT_DIM)
#     temp_y = y_test[i].reshape(1, 1)
#     res = model.evaluate(temp_x, temp_y, verbose=0)
#     # print('EVAL: ' + str(i + 1) + ' LOSS: ' + str(res))
#     epoch_fit_loss_avg += res / len(x_test)

# print('AVERAGE FIT LOSS: ' + str(round(epoch_fit_loss_avg, PRECISION)))
