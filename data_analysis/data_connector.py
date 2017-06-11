from fuel_data import FuelData
from datetime import date, timedelta
from dateutil import parser
from dateutil.relativedelta import relativedelta
import sqlite3

class DataConnector(object):

  def __init__(self):
    self.con_string = '../db/development.sqlite3'
    self.current_date = date.today()
  
  # Exchange rates
  def exchange_month_changes(self):
    fd = FuelData(self.con_string)
    start_date = date(1995, 11, 28)
    end_date = date(1995, 12, 27)

    exchange_rate_changes = []

    while (start_date <= self.current_date):
      rate_change = fd.exchange_rate_cycle_change(start_date, end_date)
      exchange_rate_changes.append(rate_change)

      start_date = start_date + relativedelta(months=1)
      end_date = end_date + relativedelta(months=1)
    
    fd.close_db_connection()
    return exchange_rate_changes

  def exchange_month_averages(self):
    fd = FuelData(self.con_string)
    start_date = date(1995, 10, 28)
    end_date = date(1995, 11, 27)

    exchange_rate_averages = []

    while (start_date <= self.current_date):
      rate_average = fd.exchange_rate_cycle_average(start_date, end_date)
      exchange_rate_averages.append(rate_average)

      start_date = start_date + relativedelta(months=1)
      end_date = end_date + relativedelta(months=1)
    
    fd.close_db_connection()
    return exchange_rate_averages

  # Oil prices
  def oil_month_changes(self):
    fd = FuelData(self.con_string)
    start_date = date(1995, 11, 28)
    end_date = date(1995, 12, 27)

    oil_price_changes = []

    while (start_date <= self.current_date):
      change_price = fd.oil_price_cycle_change(start_date, end_date)
      oil_price_changes.append(change_price)
      
      start_date = start_date + relativedelta(months=1)
      end_date = end_date + relativedelta(months=1)
    
    fd.close_db_connection()
    return oil_price_changes

  def oil_month_averages(self):
    fd = FuelData(self.con_string)
    start_date = date(1995, 10, 28)
    end_date = date(1995, 11, 27)

    oil_price_averages = []

    while (start_date <= self.current_date):
      change_average = fd.oil_price_cycle_average(start_date, end_date)
      oil_price_averages.append(change_average)
      
      start_date = start_date + relativedelta(months=1)
      end_date = end_date + relativedelta(months=1)
    
    fd.close_db_connection()
    return oil_price_averages

  # Fuel Prices
  def fuel_month_changes(self):
    fd = FuelData(self.con_string)
    start_date = date(1995, 12, 1)
    end_date = date(1996, 1, 1)

    fuel_price_changes = []

    while (start_date <= self.current_date):
      fuel_price_change = fd.fuel_price_cycle_change(start_date, end_date)
      fuel_price_changes.append(fuel_price_change)
      
      start_date = start_date + relativedelta(months=1)
      end_date = end_date + relativedelta(months=1)

    fd.close_db_connection()
    return fuel_price_changes

  def fuel_month_value(self):
    fd = FuelData(self.con_string)
    start_date = date(1995, 12, 1)

    fuel_price_values = []

    while (start_date <= self.current_date):
      fuel_price_value = fd.fuel_price_month_value(start_date)
      fuel_price_values.append(fuel_price_value)
      
      start_date = start_date + relativedelta(months=1)

    fd.close_db_connection()
    return fuel_price_values
