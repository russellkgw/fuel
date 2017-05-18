from fuel_data import FuelData
from datetime import date, timedelta
from dateutil import parser
from dateutil.relativedelta import relativedelta
import sqlite3

fd = FuelData('../db/development.sqlite3')

current_date = date.today()

# Exchange rates

# Start on 28th to 27th, inclusive
start_date = date(1995, 10, 28)
end_date = date(1995, 11, 27)

monthly_exchange_rates = []

while (start_date <= current_date):
  ave_rate = fd.exchange_rate_cycle(start_date, end_date)
  monthly_exchange_rates.append([start_date, end_date, ave_rate])
  
  start_date = start_date + relativedelta(months=1)
  end_date = end_date + relativedelta(months=1)

# Oil prices

# Start on 28th to 27th, inclusive
start_date = date(1995, 10, 28)
end_date = date(1995, 11, 27)

monthly_oil_prices = []

while (start_date <= current_date):
  ave_price = fd.oil_price_cycle(start_date, end_date)
  monthly_oil_prices.append([start_date, end_date, ave_price])
  
  start_date = start_date + relativedelta(months=1)
  end_date = end_date + relativedelta(months=1)

fd.close_db_connection()

print('-fin')