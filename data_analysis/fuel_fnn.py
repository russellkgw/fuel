from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from data_connector import DataConnector

import itertools
import pandas as pd
import tensorflow as tf

# TF logging
tf.logging.set_verbosity(tf.logging.INFO)

COLUMNS = ["exchange_rate", "oil_price", "fuel_price"]
FEATURES = ["exchange_rate", "oil_price"]
LABEL = "fuel_price"

# Fuel data
data_conn = DataConnector()

exchange = data_conn.exchange_month_changes()
oil = data_conn.oil_month_changes()
fuel = data_conn.fuel_month_changes()

exchange_pd = []
oil_pd = []
fuel_pd = []

for i in range(255):
  exchange_pd.append(exchange[i])
  oil_pd.append(oil[i])
  fuel_pd.append(fuel[i])

data = {'exchange_rate' : exchange_pd, 'oil_price' : oil_pd, 'fuel_price' : fuel_pd}
pd_df = pd.DataFrame(data)

exchange_pred_pd = []
oil_pred_pd = []
fuel_pred_pd = []

for i in range(235, 255):
  exchange_pred_pd.append(exchange[i])
  oil_pred_pd.append(oil[i])
  fuel_pred_pd.append(fuel[i])

data_predict = {'exchange_rate' : exchange_pred_pd, 'oil_price' : oil_pred_pd, 'fuel_price' : fuel_pred_pd}
pd_df_pred = pd.DataFrame(data_predict)


def input_fn(data_set):
  feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES}
  labels = tf.constant(data_set[LABEL].values)
  return feature_cols, labels

def main(unused_argv):
  # Load datasets
  training_set = pd_df
  prediction_set = pd_df_pred

  # Feature cols
  feature_cols = [tf.contrib.layers.real_valued_column(k) for k in FEATURES]

  model = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols,hidden_units=[30, 30, 30])
  model.fit(input_fn=lambda: input_fn(training_set), steps=15000)

  # Score accuracy
  # ev = model.evaluate(input_fn=lambda: input_fn(test_set), steps=1) # test_set = pd.read_csv("boston_test.csv", skipinitialspace=True, skiprows=1, names=COLUMNS)
  # loss_score = ev["loss"]
  # print("Loss: {0:f}".format(loss_score))

  # Print out predictions
  y = model.predict(input_fn=lambda: input_fn(prediction_set))
  predictions = list(itertools.islice(y, 20))
  print("Predictions: {}".format(str(predictions)))
  print("Actual vals: " + str(fuel_pred_pd))

if __name__ == "__main__":
  tf.app.run()
