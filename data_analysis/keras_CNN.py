import datetime
from keras.models import Sequential
from keras.layers import LSTM, GRU
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras import optimizers
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from data_connector import DataConnector

SEQ_LENGTH = 50  # 45 50 55 60
PRE_SET = 10
PRE_SET_VAL = None # None

data_conn = DataConnector()
x_array, y_array, train_norm = data_conn.fuel_prices_dates(start_date=None, flatten=False, percentage_change=False, data_set='training', seq_length=SEQ_LENGTH, pre_set=PRE_SET, pre_set_val=PRE_SET_VAL)

SEQ_LEN = len(x_array[0])
INPUT_DIM = len(x_array[0][0])

DROP = 0.04 # 0.08  # 0.2
EPOCHS = 50
PRECISION = 6
LR = 0.01
DECAY = 1e-6
MOMENTUM = 0.5


model = Sequential()
model.add(Conv2D(1, (2,2), activation='relu', padding='same', input_shape=(INPUT_DIM, SEQ_LEN, 1))) # ([width, height, channels])
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(DROP))
model.add(Conv2D(1, (2,2), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(DROP))
model.add(Flatten())
model.add(Dense(30, activation='relu'))
model.add(Dropout(DROP))
model.add(Dense(10, activation='relu'))
model.add(Dropout(DROP))
model.add(Dense(1))

sgd = optimizers.SGD(lr=LR, decay=DECAY, momentum=MOMENTUM)
model.compile(loss='mse', optimizer=sgd, metrics=['acc'])

# import pdb; pdb.set_trace()
PRECISION = 6
s = datetime.datetime.now()

### FIT ###
print(' ### Fit the model ### ')
x_fit = x_array
y_fit = y_array

epoch_loss_list = []
epoch_train_loss_avg = 0.0
mape_train_list = []
mape_train_avg = 0.0

for e in range(EPOCHS):
    epoch_loss = 0.0
    mape_train_avg_per_e = 0.0
    for i in range(len(x_fit)):
        temp_x = x_fit[i].reshape(1, INPUT_DIM, SEQ_LEN, 1)
        temp_y = y_fit[i].reshape(1, 1)
        res = model.fit(temp_x, temp_y, verbose=0, epochs=1)
        epoch_loss += res.history['loss'][0]/len(x_fit)

        predicted = model.predict(temp_x, verbose=0)
        y_unorm = data_conn.un_norm(y_fit[i], train_norm['fp_min'], train_norm['fp_max'])
        predicted_unorm = data_conn.un_norm(predicted[0], train_norm['fp_min'], train_norm['fp_max'])[0]
        mape_train_avg_per_e += (abs(y_unorm - predicted_unorm) / y_unorm) / len(x_fit)

    print('Training EPOCH: ' + str(e + 1) + ' LOSS: ' + str(epoch_loss) + ' MAPE: ' + str(mape_train_avg_per_e * 100.0))
    epoch_loss_list.append(epoch_loss)
    epoch_train_loss_avg += epoch_loss / EPOCHS

    mape_train_list.append(mape_train_avg_per_e  * 100.0)
    mape_train_avg += mape_train_avg_per_e / EPOCHS

e = datetime.datetime.now()
print('-------------------   TIME TO TRAIN IN SECONDS: ' + str((e - s).seconds))
print('AVERAGE TRAIN LOSS: ' + str(round(epoch_train_loss_avg, PRECISION)) + ' AVERAGE TRAIN MAPE: ' + str(round(mape_train_avg * 100.0, PRECISION)))

# import pdb; pdb.set_trace()

# Validation
x_validate_array, y_validate_array, vali_norm = data_conn.fuel_prices_dates(start_date=None, flatten=False, percentage_change=False, data_set='validation', seq_length=SEQ_LENGTH, pre_set=PRE_SET, pre_set_val=PRE_SET_VAL)

epoch_fit_loss_avg_valid = 0.0
mape_valid = 0.0
validation_eval = []
validation_act = []
mape_valid_list = []

for i in range(len(x_validate_array)):
    temp_x = x_validate_array[i].reshape(1, INPUT_DIM, SEQ_LEN, 1)
    temp_y = y_validate_array[i].reshape(1, 1)
    res = model.evaluate(temp_x, temp_y, verbose=0)[0]

    predicted = model.predict(temp_x, verbose=0)

    y_unorm = data_conn.un_norm(y_validate_array[i], vali_norm['fp_min'], vali_norm['fp_max'])
    predicted_unorm = data_conn.un_norm(predicted[0], vali_norm['fp_min'], vali_norm['fp_max'])[0]

    mape_i = (abs(y_unorm - predicted_unorm) / y_unorm)
    mape_valid += mape_i / len(y_validate_array)

    mape_valid_list.append(mape_i * 100.0)

    validation_act.append(y_unorm)
    validation_eval.append(predicted_unorm)

    epoch_fit_loss_avg_valid += res / len(y_validate_array)

print('AVERAGE VALIDATION LOSS: ' + str(round(epoch_fit_loss_avg_valid, PRECISION)) + ' AVERAGE VALIDATION MAPE: ' + str(round(mape_valid * 100.0, PRECISION)))

# import pdb; pdb.set_trace()

### TEST ###

# trained on lenght 60 seq so test seq needs to be 60 in total.
x_test_array, y_test_array, test_norm = data_conn.fuel_prices_dates(start_date=None, flatten=False, percentage_change=False, data_set='testing', seq_length=SEQ_LENGTH, pre_set=PRE_SET, pre_set_val=PRE_SET_VAL)
x_test = x_test_array
y_test = y_test_array
epoch_fit_loss_avg = 0.0
mape = 0.0

test_eval = []
test_act = []
mape_test_list = []

for i in range(len(x_test)):
    temp_x = x_test[i].reshape(1, INPUT_DIM, SEQ_LEN, 1)
    temp_y = y_test[i].reshape(1, 1)
    res = model.evaluate(temp_x, temp_y, verbose=0)[0]

    predicted = model.predict(temp_x, verbose=0)

    y_unorm = data_conn.un_norm(y_test[i], test_norm['fp_min'], test_norm['fp_max'])
    predicted_unorm = data_conn.un_norm(predicted[0], test_norm['fp_min'], test_norm['fp_max'])[0]

    mape_i = (abs(y_unorm - predicted_unorm) / y_unorm)
    mape += mape_i / len(y_test)

    mape_test_list.append(mape_i * 100.0)

    test_act.append(y_unorm)
    test_eval.append(predicted_unorm)
    epoch_fit_loss_avg += res / len(y_test)

print('TEST OFFSET: ' + str(PRE_SET) + ' AVERAGE TEST LOSS: ' + str(round(epoch_fit_loss_avg, PRECISION)) + ' AVERAGE TEST MAPE: ' + str(round(mape * 100.0, PRECISION)))


# print(' ### Evaluate the model ### ')
# x_test = x_array[SPLIT:]
# y_test = y_array[SPLIT:]
# epoch_fit_loss_avg = 0.0
# for i in range(len(x_test)):
#     temp_x = x_test[i].reshape(1, INPUT_DIM, SEQ_LEN, 1)
#     temp_y = y_test[i].reshape(1, 1)
#     res = model.evaluate(temp_x, temp_y, verbose=0)[0]
#     epoch_fit_loss_avg += res / len(x_test)

# print('AVERAGE FIT LOSS: ' + str(round(epoch_fit_loss_avg, PRECISION)))