import numpy as np
import pandas as pd
import tensorflow
import keras

from data_connector import DataConnector
from keras.models import Sequential
from keras.layers import Dense, Activation

# Data
data_conn = DataConnector()

exchange = data_conn.exchange_month_changes()
oil = data_conn.oil_month_changes()
fuel = data_conn.fuel_month_changes()

x_values = [] #list()
y_values = []

for i in range(255):
  x_values.append(np.array([exchange[i], oil[i]]))
  y_values.append(fuel[i])

x_values = np.array(x_values)
y_values = np.array(y_values)

# exchange_pred_pd = []
# oil_pred_pd = []
# fuel_pred_pd = []

# for i in range(235, 255):
#   exchange_pred_pd.append(exchange[i])
#   oil_pred_pd.append(oil[i])
#   fuel_pred_pd.append(fuel[i])

# data_predict = {'exchange_rate' : exchange_pred_pd, 'oil_price' : oil_pred_pd, 'fuel_price' : fuel_pred_pd}
# pd_df_pred = pd.DataFrame(data_predict)

# Keras Model

model = Sequential()
model.add(Dense(100, activation='relu', input_dim=2))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

data = x_values
labels = y_values

# Train the model
model.fit(data, labels, epochs=100, batch_size=32)

print("X: " + str(x_values[249]) + " Y: " + str(y_values[249]))
print("X: " + str(x_values[250]) + " Y: " + str(y_values[250]))
print("X: " + str(x_values[251]) + " Y: " + str(y_values[251]))

res = model.predict(np.array([x_values[249], x_values[250], x_values[251]]))
print("Res: " + str(res))