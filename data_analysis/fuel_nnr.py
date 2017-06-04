from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from data_connector import DataConnector

import itertools
import pandas as pd
import tensorflow as tf

# TF logging
tf.logging.set_verbosity(tf.logging.INFO)

# Fuel data
data_conn = DataConnector()
data = {'exchange_rate' : data_conn.exchange_month_changes(), 'oil_price' : data_conn.oil_month_changes(), 'fuel_price' : data_conn.fuel_month_changes()}

# print(data_conn.exchange_month_changes()) # NB! last is 0 

COLUMNS = ["exchange_rate", "oil_price", "fuel_price"]
FEATURES = ["exchange_rate", "oil_price"]
LABEL = "fuel_price"

pd_df = pd.DataFrame(data)

# print(pd_df)

def input_fn(data_set):
  feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES}
  labels = tf.constant(data_set[LABEL].values)
  return feature_cols, labels

# def main(unused_argv):
#   # Load datasets
#   training_set = pd.read_csv("boston_train.csv", skipinitialspace=True,
#                              skiprows=1, names=COLUMNS)
  
#   test_set = pd.read_csv("boston_test.csv", skipinitialspace=True,
#                          skiprows=1, names=COLUMNS)

#   # Set of 6 examples for which to predict median house values
#   prediction_set = pd.read_csv("boston_predict.csv", skipinitialspace=True,
#                                skiprows=1, names=COLUMNS)

#   # Feature cols
#   feature_cols = [tf.contrib.layers.real_valued_column(k)
#                   for k in FEATURES]

#   # Build 2 layer fully connected DNN with 10, 10 units respectively.
#   regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols,
#                                             hidden_units=[10, 10],
#                                             model_dir="/tmp/boston_model")

#   # Fit
#   regressor.fit(input_fn=lambda: input_fn(training_set), steps=5000)

#   # Score accuracy
#   ev = regressor.evaluate(input_fn=lambda: input_fn(test_set), steps=1)
#   loss_score = ev["loss"]
#   print("Loss: {0:f}".format(loss_score))

#   # Print out predictions
#   y = regressor.predict(input_fn=lambda: input_fn(prediction_set))
#   # .predict() returns an iterator; convert to a list and print predictions
#   predictions = list(itertools.islice(y, 6))
#   print("Predictions: {}".format(str(predictions)))

# if __name__ == "__main__":
#   tf.app.run()
