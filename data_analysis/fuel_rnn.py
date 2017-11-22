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

data_size_trn = 150
data_size_tst = 165

training_set = pd.DataFrame({'exchange_rate': exchange[:data_size_trn], 'oil_price': oil[:data_size_trn],
                             'exchange_future': exchange_future[:data_size_trn], 'oil_future': oil_future[:data_size_trn],
                             'fuel_price': fuel[:data_size_trn]})

test_set = pd.DataFrame({'exchange_rate': exchange[data_size_trn:data_size_tst], 'oil_price': oil[data_size_trn:data_size_tst],
                         'exchange_future': exchange_future[data_size_trn:data_size_tst],
                         'oil_future': oil_future[data_size_trn:data_size_tst],
                         'fuel_price': fuel[data_size_trn:data_size_tst]})

COLUMNS = ['exchange_rate', 'oil_price', 'fuel_price']  # 'exchange_future', 'oil_future',
FEATURES = ['exchange_rate', 'oil_price']  # 'exchange_future', 'oil_future'
LABEL = 'fuel_price'

# BATCH_SIZE = 32
# SEQUENCE_LENGTH = 16


# import pdb; pdb.set_trace()
def ms_error(actual, prediction):
    return ((actual - prediction) ** 2).sum() / len(actual.values)


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
                                                 num_units=[12, 12, 12],
                                                 cell_type='gru',  # 'basic_rnn', 'gru', 'lstm'
                                                 optimizer='Adagrad',  # 'Adagrad', 'Adam', 'Ftrl', 'Momentum', 'RMSProp', 'SGD'
                                                 learning_rate=0.1,
                                                 gradient_clipping_norm=0.2)

    # Fit Model
    model.fit(input_fn=lambda: input_fn(training_set), steps=500)

    # Test Accuracy
    model_eval = model.evaluate(input_fn=lambda: input_fn(test_set), steps=10)
    print("***** Test Loss: " + str(model_eval))

    # Predictions + Accuracy
    y = model.predict(input_fn=lambda: input_fn(test_set), as_iterable=False)

    mse = ms_error(test_set['fuel_price'], pd.DataFrame({'scores': y['scores']})['scores'])
    print('***** MSE: ' + str(mse))

    # accu = tf.contrib.metrics.accuracy(tf.constant(test_set['fuel_price'].values), tf.constant(y['scores']))
    # print('***** TF accuracy: ' + str(accu))

    # import pdb; pdb.set_trace()

    accu = tf.contrib.metrics.accuracy(tf.constant(test_set['fuel_price'].values), tf.constant(y['scores']))
    print('***** TF accuracy: ' + str(accu))

if __name__ == "__main__":
    tf.app.run()
