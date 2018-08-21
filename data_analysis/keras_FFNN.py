import datetime
from keras.models import Sequential
from keras.layers import LSTM, GRU
from keras.layers import Dense, Dropout
from keras import optimizers
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from data_connector import DataConnector

# Data Params
SEQ_LENGTH = 60  # 45 50 55 60
PRE_SET = 0
PRE_SET_VAL = 0.0 # None

data_conn = DataConnector()

x_train_array, y_train_array, train_norm = data_conn.fuel_prices_dates(start_date=None, data_set='training', seq_length=SEQ_LENGTH, pre_set=PRE_SET, pre_set_val=PRE_SET_VAL)

# import pdb; pdb.set_trace()

INPUT_DIM = len(x_train_array[0])
DROP = 0.08  # 0.2
EPOCHS = 50

# import pdb; pdb.set_trace()
model = Sequential()  # dropout=0.1, recurrent_dropout=0.1
model.add(Dense(124, activation='relu', input_dim=INPUT_DIM))
model.add(Dropout(DROP))
model.add(Dense(248, activation='relu'))
model.add(Dropout(DROP))
model.add(Dense(124, activation='relu'))
model.add(Dropout(DROP))
model.add(Dense(62, activation='relu'))
model.add(Dropout(DROP))
model.add(Dense(31, activation='relu'))
model.add(Dropout(DROP))
model.add(Dense(1, activation='linear'))
# sgd = optimizers.SGD(lr=0.1)  # , decay=1e-6, momentum=0.9, nesterov=True
model.compile(loss='mse', optimizer='sgd')  # adagrad adam sgd rmsprop     , metrics=['acc']

PRECISION = 6

s = datetime.datetime.now()

### FIT ###
x_fit = x_train_array
y_fit = y_train_array

epoch_loss_list = []
epoch_train_loss_avg = 0.0
mape_train_list = []
mape_train_avg = 0.0

for e in range(EPOCHS):
    epoch_loss = 0.0
    mape_train_avg_per_e = 0.0
    for i in range(len(x_fit)):
        temp_x = x_fit[i].reshape(1, INPUT_DIM)
        temp_y = y_fit[i].reshape(1, 1)
        res = model.fit(temp_x, temp_y, verbose=0, epochs=1)
        epoch_loss += res.history['loss'][0]/len(x_fit)

        predicted = model.predict(temp_x, verbose=0)
        y_unorm = data_conn.un_norm(y_fit[i], train_norm['fp_min'], train_norm['fp_max'])
        predicted_unorm = data_conn.un_norm(predicted[0], train_norm['fp_min'], train_norm['fp_max'])[0]
        mape_train_avg_per_e += (abs(y_unorm - predicted_unorm) / y_unorm) / len(x_fit)

        # import pdb; pdb.set_trace()


    print('Training EPOCH: ' + str(e + 1) + ' LOSS: ' + str(epoch_loss) + ' MAPE: ' + str(mape_train_avg_per_e * 100.0))
    epoch_loss_list.append(epoch_loss)
    epoch_train_loss_avg += epoch_loss / EPOCHS

    mape_train_list.append(mape_train_avg_per_e  * 100.0)
    mape_train_avg += mape_train_avg_per_e / EPOCHS

e = datetime.datetime.now()
print('-------------------   TIME TO TRAIN IN SECONDS: ' + str((e - s).seconds))
print('AVERAGE TRAIN LOSS: ' + str(round(epoch_train_loss_avg, PRECISION)) + ' AVERAGE TRAIN MAPE: ' + str(round(mape_train_avg * 100.0, PRECISION)))


### Validation ###
x_validate_array, y_validate_array, vali_norm = data_conn.fuel_prices_dates(start_date=None, data_set='validation', seq_length=SEQ_LENGTH, pre_set=PRE_SET, pre_set_val=PRE_SET_VAL)

# print(' ### Validation of the model ### ')
epoch_fit_loss_avg_valid = 0.0
mape_valid = 0.0

validation_eval = []
validation_act = []
mape_valid_list = []

for i in range(len(x_validate_array)):
    temp_x = x_validate_array[i].reshape(1, INPUT_DIM)
    temp_y = y_validate_array[i].reshape(1, 1)
    res = model.evaluate(temp_x, temp_y, verbose=0)

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



### TESTING ###
# print(' ### Testing of the model ### ')

for offset in [PRE_SET]:  # 0, 5, 10, 15
    # trained on lenght 60 seq so test seq needs to be 60 in total.
    x_test_array, y_test_array, test_norm = data_conn.fuel_prices_dates(start_date=None, data_set='testing', seq_length=SEQ_LENGTH, pre_set=offset, pre_set_val=PRE_SET_VAL)
    x_test = x_test_array
    y_test = y_test_array
    epoch_fit_loss_avg = 0.0
    mape = 0.0

    test_eval = []
    test_act = []
    mape_test_list = []

    for i in range(len(x_test)):
        temp_x = x_test[i].reshape(1, INPUT_DIM)
        temp_y = y_test[i].reshape(1, 1)
        res = model.evaluate(temp_x, temp_y, verbose=0)

        predicted = model.predict(temp_x, verbose=0)

        y_unorm = data_conn.un_norm(y_test[i], test_norm['fp_min'], test_norm['fp_max'])
        predicted_unorm = data_conn.un_norm(predicted[0], test_norm['fp_min'], test_norm['fp_max'])[0]

        mape_i = (abs(y_unorm - predicted_unorm) / y_unorm)
        mape += mape_i / len(y_test)

        mape_test_list.append(mape_i * 100.0)

        test_act.append(y_unorm)
        test_eval.append(predicted_unorm)
        epoch_fit_loss_avg += res / len(y_test)

    print('TEST OFFSET: ' + str(offset) + ' AVERAGE TEST LOSS: ' + str(round(epoch_fit_loss_avg, PRECISION)) + ' AVERAGE TEST MAPE: ' + str(round(mape * 100.0, PRECISION)))



###  Plot  ### 

# Data for plotting
t = [1,2,3,4,5]
s = [0.1,0.2,0.3,0.4,0.5]

fig, ax = plt.subplots()

e_plot = [i + 1 for i in range(EPOCHS)]

x1 = e_plot
x2 = e_plot

y1 = epoch_loss_list
y2 = mape_train_list

# plt.subplot(1, 2, 1)  # 2, 1, 1
# plt.plot(x1, y1, 'o-')
# plt.title('LOSS MAPE')
# plt.ylabel('Loss')

# plt.subplot(1, 2, 2)  # 2, 1, 2
# plt.plot(x2, y2, '.-')
# plt.xlabel('Epochs')
# plt.ylabel('MAPE')

# plt.show()
