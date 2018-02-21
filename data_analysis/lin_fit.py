import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

from data_connector import DataConnector
data_conn = DataConnector()

X = [1, 2, 3, 4, 5]
X = sm.add_constant(X)  # Add intercept component
y = [2, 4, 6, 8, 10]

# Fit regression model
# results = sm.OLS(y, X).fit()

# Inspect the results
# print(results.summary())

x_array, y_array = data_conn.fuel_prices_dates(start_date='2004-04-03', flatten=False, percentage_change=False, normilize=False)  # start_date='2004-04-03'    None is all

x_array = [np.average(x[:,0] * x[:,1]) for x in x_array]  # rand value of oil price

# import pdb; pdb.set_trace()