import sqlite3

class FuelData(object):

  # DB connection init and close

  def __init__(self, db_con_str):
    self.db_con = sqlite3.connect(db_con_str)

  def close_db_connection(self):
    self.db_con.close()

  # Exchange rates

  def exchange_rates(self):
    return self.db_con.execute("SELECT * FROM exchange_rates WHERE CAST(STRFTIME('%w', date) AS INTEGER) IN (1, 2, 3, 4, 5) AND date >= '1995-11-01' ORDER BY date;").fetchall()

  def exchange_rate_cycle(self, start_date, end_date):
    return self.db_con.execute("SELECT AVG(rate) FROM exchange_rates WHERE CAST(STRFTIME('%w', date) AS INTEGER) IN (1, 2, 3, 4, 5) AND date >= '" + str(start_date) + "' AND date <= '" + str(end_date) + "';").fetchone()[0]

  # Oil prices

  def oil_prices(self):
    return self.db_con.execute("SELECT * FROM oil_prices WHERE CAST(STRFTIME('%w', date) AS INTEGER) IN (1, 2, 3, 4, 5) AND date >= '1995-11-01' ORDER BY date;").fetchall()

  def oil_price_cycle(self, start_date, end_date):
    return self.db_con.execute("SELECT AVG(price) FROM oil_prices WHERE CAST(STRFTIME('%w', date) AS INTEGER) IN (1, 2, 3, 4, 5) AND date >= '" + str(start_date) + "' AND date <= '" + str(end_date) + "';").fetchone()[0]

  # Fuel Prices

  def fuel_prices(self):
    return self.db_con.execute("SELECT * FROM fuel_prices ORDER BY date;").fetchall()