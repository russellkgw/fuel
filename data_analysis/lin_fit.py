import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

from data_connector import DataConnector
data_conn = DataConnector()

# Single varible 1x to 1y
# x_array, y_array = data_conn.fuel_prices_dates(start_date='2004-04-03', flatten=False, percentage_change=False, normilize=False)  # start_date='2004-04-03'    None is all
# x_array = [np.average(x[:,0] * x[:,1] for x in x_array]  # rand value of oil price

# SPLIT = 140
# x_array = sm.add_constant(x_array)
# x_fit = x_array[:SPLIT]
# y_fit = y_array[:SPLIT]

# results = sm.OLS(y_fit, x_fit).fit()
# print(results.summary())

# print('Test x values:')
# print(str(x_array[SPLIT:][:,1]))

# print('Test y values:')
# print(str(y_array[SPLIT:]))

# print('Predicted y values:')
# ans = results.predict(x_array[SPLIT:])
# print(str(ans))

# print('MSE:')
# mse = np.average((np.array(y_array[SPLIT:]) - np.array(ans)) * (np.array(y_array[SPLIT:]) - np.array(ans)))
# print(str(mse))

# Two x to 1y varible
# x_array, y_array = data_conn.fuel_prices_dates(start_date='2004-04-03', flatten=False, percentage_change=False, normilize=False)  # start_date='2004-04-03'    None is all
# x_array = [np.column_stack((np.average(x[:,0]), np.average(x[:,1])))[0] for x in x_array]  # rand value of oil price

# import pdb; pdb.set_trace()

# SPLIT = 140
# x_array = sm.add_constant(x_array)
# x_fit = x_array[:SPLIT]
# y_fit = y_array[:SPLIT]

# results = sm.OLS(y_fit, x_fit).fit()
# print(results.summary())

# print('Test x values:')
# print(str(x_array[SPLIT:][:,1]))

# print('Test y values:')
# print(str(y_array[SPLIT:]))

# print('Predicted y values:')
# ans = results.predict(x_array[SPLIT:])
# print(str(ans))

# print('MSE:')
# mse = np.average((np.array(y_array[SPLIT:]) - np.array(ans)) * (np.array(y_array[SPLIT:]) - np.array(ans)))
# print(str(mse))

# Two 62x to 1y varible
x_array, y_array = data_conn.fuel_prices_dates(start_date='2004-04-03', flatten=False, percentage_change=False, normilize=False)  # start_date='2004-04-03'    None is all
x_array = [np.column_stack((x[:,0], x[:,1])) for x in x_array]  # rand value of oil price

import pdb; pdb.set_trace()

SPLIT = 140
x_array = sm.add_constant(x_array)
x_fit = x_array[:SPLIT]
y_fit = y_array[:SPLIT]

results = sm.OLS(y_fit, x_fit).fit()
print(results.summary())

# print('Test x values:')
# print(str(x_array[SPLIT:][:,1]))

# print('Test y values:')
# print(str(y_array[SPLIT:]))

# print('Predicted y values:')
# ans = results.predict(x_array[SPLIT:])
# print(str(ans))

# print('MSE:')
# mse = np.average((np.array(y_array[SPLIT:]) - np.array(ans)) * (np.array(y_array[SPLIT:]) - np.array(ans)))
# print(str(mse))