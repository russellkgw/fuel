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

exchange_pd = exchange[:255]
oil_pd = oil[:255]
fuel_pd = fuel[:255]

data = {'exchange_rate' : exchange_pd, 'oil_price' : oil_pd, 'fuel_price' : fuel_pd}
pd_df = pd.DataFrame(data)

exchange_pred_pd = exchange[235:255]
oil_pred_pd = oil[235:255]
fuel_pred_pd = fuel[235:255]

data_predict = {'exchange_rate' : exchange_pred_pd, 'oil_price' : oil_pred_pd, 'fuel_price' : fuel_pred_pd}
pd_df_pred = pd.DataFrame(data_predict)

BATCH_SIZE = 32
SEQUENCE_LENGTH = 16


# If you specify the shape of your tensor explicitly:
# tf.constant(df[k].values, shape=[df[k].size, 1])  # {k: tf.constant(df[k].values, shape=[df[k].size, 1]) for k in CONTINUOUS_COLUMNS}
# the warning should go away.

def input_fn(data_set):
  feature_cols = {k: tf.constant(data_set[k].values, shape=[data_set[k].size, 1]) for k in FEATURES}
  labels = tf.constant(data_set[LABEL].values)
  return feature_cols, labels

# def get_train_inputs():
#   x = tf.random_uniform([BATCH_SIZE, SEQUENCE_LENGTH])
#   y = tf.reduce_mean(x, axis=1)
#   x = tf.expand_dims(x, axis=2)
#   return {"": x}, y

def main(unused_argv):
  # Load datasets
  training_set = pd_df
  prediction_set = pd_df_pred

  # Feature cols
  feature_cols = [tf.contrib.layers.real_valued_column(k) for k in FEATURES]

  prediction_type = { 'SINGLE_VALUE': 1, 'MULTIPLE_VALUE': 2 } # problem_type = 2 # { 'UNSPECIFIED': 0, 'CLASSIFICATION': 1, 'LINEAR_REGRESSION': 2, 'LOGISTIC_REGRESSION': 3 }

  # Recurrent - Mine
  model = tf.contrib.learn.DynamicRnnEstimator(tf.contrib.learn.ProblemType.LINEAR_REGRESSION, 
    prediction_type['SINGLE_VALUE'], 
    feature_cols, 
    num_units=[10, 10, 10],
    cell_type='basic_rnn', # 'basic_rnn', 'gru', 'lstm'
    optimizer='Adagrad', #  'Adagrad', 'Adam', 'Ftrl', 'Momentum', 'RMSProp', 'SGD'
    learning_rate=0.1)

  # feature_cols = {k: tf.constant(training_set[k].values, shape=[training_set[k].size, 1]) for k in FEATURES}
  # print('FEATURE COLS: ' + str(feature_cols))
  
  model.fit(input_fn=lambda: input_fn(training_set), steps=10000)

  # x = tf.random_uniform([BATCH_SIZE, SEQUENCE_LENGTH])
  # print('x: ' + str(x))
  # y = tf.reduce_mean(x, axis=1)
  # print('y: ' + str(y))
  # x = tf.expand_dims(x, axis=2)
  # print('x: ' + str(x))


  # xc = tf.contrib.layers.real_valued_column("")
  # print('xc:' + str(xc))
  # estimator = tf.contrib.learn.DynamicRnnEstimator(problem_type = tf.contrib.learn.ProblemType.LINEAR_REGRESSION,
  #                                                prediction_type = prediction_type['SINGLE_VALUE'],
  #                                                sequence_feature_columns = [xc],
  #                                                context_feature_columns = None,
  #                                                num_units = 5,
  #                                                cell_type = 'lstm', 
  #                                                optimizer = 'SGD',
  #                                                learning_rate = 0.1)

  # estimator.fit(input_fn=get_train_inputs, steps=100)


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
