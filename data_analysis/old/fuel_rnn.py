from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from data_connector import DataConnector
from tensorflow.contrib.learn.python.learn.estimators import rnn_common

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
fuel = data_conn.fuel_month_changes()  # first is -0.0007298

# import pdb; pdb.set_trace()

data_size_trn = 9  # 240  # 150
data_size_tst = 260  # 165

training_set = pd.DataFrame({'exchange_rate': exchange[:data_size_trn], 'oil_price': oil[:data_size_trn],
                             # 'exchange_future': exchange_future[:data_size_trn],
                             # 'oil_future': oil_future[:data_size_trn],
                             'fuel_price': fuel[:data_size_trn]})  # , rnn_common.RNNKeys.SEQUENCE_LENGTH_KEY: [3]

test_set = pd.DataFrame({'exchange_rate': exchange[data_size_trn:data_size_tst], 'oil_price': oil[data_size_trn:data_size_tst],
                         # 'exchange_future': exchange_future[data_size_trn:data_size_tst],
                         # 'oil_future': oil_future[data_size_trn:data_size_tst],
                         'fuel_price': fuel[data_size_trn:data_size_tst]})

COLUMNS = ['exchange_rate', 'oil_price', 'fuel_price']  # 'exchange_future', 'oil_future',
FEATURES = ['exchange_rate', 'oil_price']  # 'exchange_future', 'oil_future'   #  rnn_common.RNNKeys.SEQUENCE_LENGTH_KEY
LABEL = 'fuel_price'

BATCH_SIZE = 3  # 3  # 32
# SEQUENCE_LENGTH = 16

# import pdb; pdb.set_trace()

# The input function passed to this Estimator optionally contains keys RNNKeys.SEQUENCE_LENGTH_KEY.
# The value corresponding to RNNKeys.SEQUENCE_LENGTH_KEY must be vector of size 'batch_size' where entry n corresponds to the length of the nth sequence in the batch.

# The sequence length feature is required for batches of varying sizes. It will be used to calculate loss and evaluation metrics.
# If RNNKeys.SEQUENCE_LENGTH_KEY is not included, all sequences are assumed to have length equal to the size of dimension 1 of the input to the RNN.

# import pdb; pdb.set_trace()
def ms_error(actual, prediction):
    return ((actual - prediction) ** 2).sum() / len(actual.values)


def input_fn(data_set):
    # import pdb; pdb.set_trace()
    # col_list = [tf.constant(data_set[k], shape=[data_set[k].size, 1]) for k in COLUMNS]
    #
    # slice_list = tf.train.slice_input_producer(col_list, shuffle=False, num_epochs=1)
    # # er, op, fp = tf.train.slice_input_producer(col_list, shuffle=False, num_epochs=3)
    #
    # # dataset_dict = dict(exchange_rate=er, oil_price=op, fuel_price=fp)
    # dataset_dict = {'exchange_rate': slice_list[0], 'oil_price': slice_list[1], 'fuel_price': slice_list[2]}
    #
    # batch_dict = tf.train.batch(dataset_dict,
    #                             BATCH_SIZE,
    #                             # capacity=BATCH_SIZE * 2,  # Remove ?
    #                             enqueue_many=False,
    #                             dynamic_pad=True,
    #                             allow_smaller_final_batch=True)
    #
    # labels = batch_dict.pop('fuel_price')
    # batch_dict[rnn_common.RNNKeys.SEQUENCE_LENGTH_KEY] = tf.constant([BATCH_SIZE])
    # return batch_dict, labels

    import pdb; pdb.set_trace()

    feature_cols = {k: tf.constant(data_set[k], shape=[data_set[k].size, 1]) for k in FEATURES}
    feature_cols[rnn_common.RNNKeys.SEQUENCE_LENGTH_KEY] = tf.constant([BATCH_SIZE])

    labels = tf.constant(data_set[LABEL])
    return feature_cols, labels


# DynamicRnnEstimator

def main(unused_argv):
    feature_columns = [tf.contrib.layers.real_valued_column(k) for k in FEATURES]
    # feature_columns.append(tf.contrib.layers.real_valued_column(rnn_common.RNNKeys.SEQUENCE_LENGTH_KEY))

    # Recurrent - Mine
    model = tf.contrib.learn.DynamicRnnEstimator(tf.contrib.learn.ProblemType.LINEAR_REGRESSION,
                                                 rnn_common.PredictionType.SINGLE_VALUE,
                                                 feature_columns,
                                                 # sequence_feature_columns=feature_columns,  # ???
                                                 num_units=[20, 20, 20],
                                                 cell_type='gru',  # 'basic_rnn', 'gru', 'lstm'
                                                 optimizer='Adagrad',  # 'Adagrad', 'Adam', 'Ftrl', 'Momentum', 'RMSProp', 'SGD'
                                                 learning_rate=0.1,
                                                 gradient_clipping_norm=0.2)

    # Fit Model
    model.fit(input_fn=lambda: input_fn(training_set), steps=9)  #

    # # Test Accuracy
    # model_eval = model.evaluate(input_fn=lambda: input_fn(test_set), steps=10)
    # print("***** Test Loss: " + str(model_eval))
    #
    # # Predictions + Accuracy
    # y = model.predict(input_fn=lambda: input_fn(test_set), as_iterable=False)
    #
    # mse = ms_error(test_set['fuel_price'], pd.DataFrame({'scores': y['scores']})['scores'])
    # print('***** MSE: ' + str(mse))

    # accu = tf.contrib.metrics.accuracy(tf.constant(test_set['fuel_price'].values), tf.constant(y['scores']))
    # print('***** TF accuracy: ' + str(accu))

    # import pdb; pdb.set_trace()

    # accu = tf.contrib.metrics.accuracy(tf.constant(test_set['fuel_price'].values), tf.constant(y['scores']))
    # print('***** TF accuracy: ' + str(accu))

if __name__ == "__main__":
    tf.app.run()
