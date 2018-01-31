from keras.models import Sequential
from keras.layers import LSTM, GRU
from keras.layers import Dense
from keras import optimizers
import numpy as np

from data_connector import DataConnector


data_conn = DataConnector()
x_array, y_array = data_conn.fuel_prices_dates(percentage_change=True)

# import pdb; pdb.set_trace()

INPUT_DIM = 124

# import pdb; pdb.set_trace()
model = Sequential()  # dropout=0.1, recurrent_dropout=0.1
model.add(Dense(124, activation='relu', input_dim=INPUT_DIM))
model.add(Dense(248, activation='relu'))
model.add(Dense(124, activation='relu'))
model.add(Dense(62, activation='relu'))
model.add(Dense(31, activation='relu'))
model.add(Dense(1, activation='linear'))
# sgd = optimizers.SGD(lr=0.1)  # , decay=1e-6, momentum=0.9, nesterov=True
model.compile(loss='mse', optimizer='sgd')  # adagrad adam sgd rmsprop     , metrics=['acc']
# print(model.summary())

print('size of x: ' + str(len(x_array)))

# import pdb; pdb.set_trace()
PRECISION = 6

# y_array
### FIT ###
print(' ### Fit the model ### ')
x_fit = x_array[:240]
y_fit = y_array[:240]
EPOCHS = 10
epoch_loss_list = []
epoch_train_loss_avg = 0.0
for e in range(EPOCHS):
    epoch_loss = 0.0
    for i in range(len(x_fit)):
        temp_x = np.array(x_fit[i]).reshape(1, INPUT_DIM)
        temp_y = np.array(y_fit[i]).reshape(1, 1)
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
print(' ### Evaluate the model ### ')
x_test = x_array[240:]
y_test = y_array[240:]
epoch_fit_loss_avg = 0.0
for i in range(len(x_test)):
    temp_x = np.array(x_test[i]).reshape(1, INPUT_DIM)
    temp_y = np.array(y_test[i]).reshape(1, 1)
    res = model.evaluate(temp_x, temp_y, verbose=0)
    # print('EVAL: ' + str(i + 1) + ' LOSS: ' + str(res))
    epoch_fit_loss_avg += res / len(x_test)

print('AVERAGE FIT LOSS: ' + str(round(epoch_fit_loss_avg, PRECISION)))
