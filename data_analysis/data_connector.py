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

exchange_rate_changes = []

while (start_date <= current_date):
  rate_change = fd.exchange_rate_cycle_change(start_date, end_date)
  exchange_rate_changes.append([start_date, end_date, rate_change])

  start_date = start_date + relativedelta(months=1)
  end_date = end_date + relativedelta(months=1)

# Oil prices

# Start on 28th to 27th, inclusive
start_date = date(1995, 10, 28)
end_date = date(1995, 11, 27)

oil_price_changes = []

while (start_date <= current_date):
  change_price = fd.oil_price_cycle_change(start_date, end_date)
  oil_price_changes.append([start_date, end_date, change_price])
  
  start_date = start_date + relativedelta(months=1)
  end_date = end_date + relativedelta(months=1)

fd.close_db_connection()

print('-fin')