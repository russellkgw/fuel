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

BATCH_SIZE = 32
SEQUENCE_LENGTH = 16

def get_train_inputs():
    x = tf.random_uniform([BATCH_SIZE, SEQUENCE_LENGTH])
    y = tf.reduce_mean(x, axis=1)
    x = tf.expand_dims(x, axis=2)
    return {"": x}, y

def main(unused_argv):
  # Load datasets
  training_set = pd_df
  prediction_set = pd_df_pred

  # Feature cols
  feature_cols = [tf.contrib.layers.real_valued_column(k) for k in FEATURES]

  prediction_type = { 'SINGLE_VALUE': 1, 'MULTIPLE_VALUE': 2 } # problem_type = 2 # { 'UNSPECIFIED': 0, 'CLASSIFICATION': 1, 'LINEAR_REGRESSION': 2, 'LOGISTIC_REGRESSION': 3 }

  # Recurrent - Mine
  # model = tf.contrib.learn.DynamicRnnEstimator(tf.contrib.learn.ProblemType.LINEAR_REGRESSION, 
  #   prediction_type['SINGLE_VALUE'], 
  #   feature_cols, 
  #   num_units=None,
  #   cell_type='basic_rnn',
  #   optimizer='SGD',
  #   learning_rate=0.1)

  # model.fit(input_fn=lambda: input_fn(training_set), steps=1000)

  # x = tf.random_uniform([BATCH_SIZE, SEQUENCE_LENGTH])
  # print('x: ' + str(x))
  # y = tf.reduce_mean(x, axis=1)
  # print('y: ' + str(y))
  # x = tf.expand_dims(x, axis=2)
  # print('x: ' + str(x))


  xc = tf.contrib.layers.real_valued_column("")
  print('xc:' + str(xc))
  estimator = tf.contrib.learn.DynamicRnnEstimator(problem_type = tf.contrib.learn.ProblemType.LINEAR_REGRESSION,
                                                 prediction_type = prediction_type['SINGLE_VALUE'],
                                                 sequence_feature_columns = [xc],
                                                 context_feature_columns = None,
                                                 num_units = 5,
                                                 cell_type = 'lstm', 
                                                 optimizer = 'SGD',
                                                 learning_rate = 0.1)

  estimator.fit(input_fn=get_train_inputs, steps=100)


  # Score accuracy
  # ev = regressor.evaluate(input_fn=lambda: input_fn(test_set), steps=1)
  # loss_score = ev["loss"]
  # print("Loss: {0:f}".format(loss_score))

  # Print out predictions
  # y = model.predict(input_fn=lambda: input_fn(prediction_set))
  # .predict() returns an iterator; convert to a list and print predictions
  # predictions = list(itertools.islice(y, 20))
  # print("Predictions: {}".format(str(predictions)))
  # print("Actual vals: " + str(fuel_pred_pd))

if __name__ == "__main__":
  tf.app.run()
