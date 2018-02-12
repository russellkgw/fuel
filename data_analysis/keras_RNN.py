from keras.models import Sequential
from keras.layers import LSTM, GRU
from keras.layers import Dense
from keras import optimizers
import numpy as np


from data_connector import DataConnector

data_conn = DataConnector()
x_array, y_array = data_conn.fuel_prices_dates(start_date='2004-04-03', flatten=False, percentage_change=False)  # start_date='2004-04-03'    None is all

SEQ_LEN = len(x_array[0])
INPUT_DIM = len(x_array[0][0])


# NB!!!
# Keras provides exibility to decouple the resetting of internal state from updates to network
# weights by dening an LSTM layer as stateful. This can be done by setting the stateful
# argument on the LSTM layer to True. When stateful LSTM layers are used, you must also
# dene the batch size as part of the input shape in the denition of the network by setting the
# batch input shape argument and the batch size must be a factor of the number of samples in
# the training dataset. The batch input shape argument requires a 3-dimensional tuple dened
# as batch size, time steps, and features.
# For example, we can dene a stateful LSTM to be trained on a training dataset with 100
# samples, a batch size of 10, and 5 time steps for 1 feature, as follows.
# model.add(LSTM(2, stateful=True, batch_input_shape=(10, 5, 1)))


# import pdb; pdb.set_trace()
model = Sequential()  # dropout=0.1, recurrent_dropout=0.1
model.add(LSTM(240, activation='relu', return_sequences=True, input_shape=(SEQ_LEN, INPUT_DIM)))  # LSTM GRU  return_state=True
model.add(LSTM(120, activation='relu', return_sequences=True))  # activation='tanh', recurrent_activation='hard_sigmoid'
model.add(LSTM(60, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(1, activation='linear'))
# sgd = optimizers.SGD(lr=0.1)  # , decay=1e-6, momentum=0.9, nesterov=True
model.compile(loss='mse', optimizer='sgd', metrics=['acc'])  # adagrad adam sgd rmsprop
# print(model.summary())

# import pdb; pdb.set_trace()
PRECISION = 6
SPLIT = 140  # 240

### FIT ###
print(' ### Fit the model ### ')
x_fit = x_array[:SPLIT]
y_fit = y_array[:SPLIT]
EPOCHS = 10
epoch_loss_list = []
epoch_train_loss_avg = 0.0
for e in range(EPOCHS):
    epoch_loss = 0.0
    for i in range(len(x_fit)):
        print
        temp_x = x_fit[i].reshape(1, SEQ_LEN, INPUT_DIM)
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
print(' ### Evaluate the model ### ')
x_test = x_array[SPLIT:]
y_test = y_array[SPLIT:]
epoch_fit_loss_avg = 0.0
for i in range(len(x_test)):
    temp_x = x_test[i].reshape(1, SEQ_LEN, INPUT_DIM)
    temp_y = y_test[i].reshape(1, 1)
    res = model.evaluate(temp_x, temp_y, verbose=0)
    print('EVAL: ' + str(i + 1) + ' LOSS: ' + str(res))
    epoch_fit_loss_avg += res / len(x_test)

print('AVERAGE FIT LOSS: ' + str(round(epoch_fit_loss_avg, PRECISION)))