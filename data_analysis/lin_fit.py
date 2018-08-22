import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

from data_connector import DataConnector
data_conn = DataConnector()


for i in [60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5]:
    # Data Params
    SEQ_LENGTH = i  # 45 50 55 60
    PRE_SET = 0
    PRE_SET_VAL = 0.0 # None

    # Two x to 1y varible
    x_train_array, y_train_array, train_norm = data_conn.fuel_prices_dates(start_date=None, flatten=False, percentage_change=False, normilize=False, data_set='training', seq_length=SEQ_LENGTH, pre_set=PRE_SET, pre_set_val=PRE_SET_VAL)
    x_array = [np.column_stack((np.average(x[:,0]), np.average(x[:,1])))[0] for x in x_train_array]  # rand value of oil price
    x_array = sm.add_constant(x_array)

    results = sm.OLS(y_train_array, x_array).fit()
    # print(results.summary())

    # Test
    x_test_array, y_test_array, test_norm = data_conn.fuel_prices_dates(start_date=None, flatten=False, percentage_change=False, normilize=False, data_set='testing', seq_length=SEQ_LENGTH, pre_set=PRE_SET, pre_set_val=PRE_SET_VAL)
    x_array_test = [np.column_stack((np.average(x[:,0]), np.average(x[:,1])))[0] for x in x_test_array]
    x_array_test = sm.add_constant(x_array_test)

    predicted = results.predict(x_array_test)

    mape_train_avg = 0.0
    for i in range(len(y_test_array)):
        mape_train_avg += (abs(y_test_array[i] - predicted[i]) / y_test_array[i]) / len(y_test_array)

    print('TEST MAPE: ' + str(mape_train_avg * 100.0) + ' SEQ_LENGTH: ' + str(SEQ_LENGTH))
