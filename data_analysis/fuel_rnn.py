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

exchange = data_conn.exchange_month_changes()
oil = data_conn.oil_month_changes()
exchange_future = data_conn.exchange_future_month_changes()
oil_future = data_conn.oil_future_month_changes()
fuel = data_conn.fuel_month_changes()

training_set = pd.DataFrame({'exchange_rate': exchange[:235], 'oil_price': oil[:235],
                             'exchange_future': exchange_future[:235], 'oil_future': oil_future[:235],
                             'fuel_price': fuel[:235]})

test_set = pd.DataFrame({'exchange_rate': exchange[235:255], 'oil_price': oil[235:255],
                         'exchange_future': exchange_future[235:255], 'oil_future': oil_future[235:255],
                         'fuel_price': fuel[235:255]})

COLUMNS = ["exchange_rate", "oil_price", "fuel_price"]
FEATURES = ["exchange_rate", "oil_price"]
LABEL = "fuel_price"

# BATCH_SIZE = 32
# SEQUENCE_LENGTH = 16

# import pdb; pdb.set_trace()


def input_fn(data_set):
    feature_columns = {k: tf.constant(data_set[k], shape=[data_set[k].size, 1]) for k in FEATURES}
    labels = tf.constant(data_set[LABEL])
    return feature_columns, labels


def main(unused_argv):
    feature_cols = [tf.contrib.layers.real_valued_column(k) for k in FEATURES]
    prediction_type = {'SINGLE_VALUE': 1, 'MULTIPLE_VALUE': 2}  # problem_type = 2 # { 'UNSPECIFIED': 0, 'CLASSIFICATION': 1, 'LINEAR_REGRESSION': 2, 'LOGISTIC_REGRESSION': 3 }

    # Recurrent - Mine
    model = tf.contrib.learn.DynamicRnnEstimator(tf.contrib.learn.ProblemType.LINEAR_REGRESSION,
                                                 prediction_type['SINGLE_VALUE'],
                                                 feature_cols,
                                                 num_units=[10, 10, 10],
                                                 cell_type='gru',  # 'basic_rnn', 'gru', 'lstm'
                                                 optimizer='Adagrad',  # 'Adagrad', 'Adam', 'Ftrl', 'Momentum', 'RMSProp', 'SGD'
                                                 learning_rate=0.1)

    # Fit Model
    model.fit(input_fn=lambda: input_fn(training_set), steps=10000)

    # Test Accuracy
    model_eval = model.evaluate(input_fn=lambda: input_fn(test_set), steps=10)
    print("Test Loss: {0:f}".format(model_eval["loss"]))

    # Print out predictions
    # y = model.predict(input_fn=lambda: input_fn(prediction_set))
    # .predict() returns an iterator; convert to a list and print predictions
    # predictions = list(itertools.islice(y, 20))
    # print("Predictions: {}".format(str(predictions)))
    # print("Actual vals: " + str(fuel_pred_pd))

if __name__ == "__main__":
    tf.app.run()
