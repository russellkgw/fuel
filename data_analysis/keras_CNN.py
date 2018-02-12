from keras.models import Sequential
from keras.layers import LSTM, GRU
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras import optimizers
import numpy as np


from data_connector import DataConnector

data_conn = DataConnector()
x_array, y_array = data_conn.fuel_prices_dates(start_date='2004-04-03', flatten=False, percentage_change=False)

SEQ_LEN = len(x_array[0])
INPUT_DIM = len(x_array[0][0])

# import pdb; pdb.set_trace()


# feed_data = []
# for d in data:
#     y = norm_array(d['y'], fp_min, fp_max)
#     x1 = norm_array(d['exr'], exr_min, exr_max)
#     x2 = norm_array(d['oil'], oil_min, oil_max)

#     x = np.column_stack((x1, x2))
#     x = x.reshape(1, 2, 62, 1)

#     feed_data.append({'date': d['date'], 'x': x, 'y': y})


model = Sequential()
model.add(Conv2D(1, (2,2), activation='relu', padding='same', input_shape=(INPUT_DIM, SEQ_LEN, 1))) # ([width, height, channels])
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='sgd', metrics=['acc'])
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
        temp_x = x_fit[i].reshape(1, INPUT_DIM, SEQ_LEN, 1)
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
    temp_x = x_test[i].reshape(1, INPUT_DIM, SEQ_LEN, 1)
    temp_y = y_test[i].reshape(1, 1)
    res = model.evaluate(temp_x, temp_y, verbose=0)
    print('EVAL: ' + str(i + 1) + ' LOSS: ' + str(res))
    # epoch_fit_loss_avg += res / len(x_test)

print('AVERAGE FIT LOSS: ' + str(round(epoch_fit_loss_avg, PRECISION)))