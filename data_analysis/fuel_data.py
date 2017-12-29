import sqlite3

class FuelData(object):

    def __init__(self, db_con_str):
        self.db_con = sqlite3.connect(db_con_str)

    def close_db_connection(self):
        self.db_con.close()

    # Helper

    def percent_change(self, items):
        res = 0.0

        for i, val in enumerate(items):
            if (i + 1) <= (len(items) - 1):
                a, b = val[1], items[i + 1][1]
                res = res + ((b - a) / a)

        return res * 100.0

    def percent_change_daily(self, items):
        changes = []

        for i, val in enumerate(items):
            if (i + 1) <= (len(items) - 1):
                a, b = items[i], items[i + 1]
                changes.append(((b - a) / a) * 100.0)

        return changes
    
    # Exchange rates

    # use 28th
    def exchange_rates(self, start_date, end_date, offset=-62, percentage=False):
        data = self.db_con.execute("SELECT * FROM exchange_rates WHERE CAST(STRFTIME('%w', date) AS INTEGER) IN (1, 2, 3, 4, 5) AND date > '{0}' AND date <= '{1}' ORDER BY date;".format(start_date, end_date)).fetchall()
        # import pdb; pdb.set_trace()
        data = [e[3] for e in data]
        if percentage:
            data = self.percent_change_daily(data)
        return data[offset:]  # last 62

    def exchange_rate_cycle_change(self, start_date, end_date):
        rates = self.db_con.execute("SELECT date, rate FROM exchange_rates WHERE CAST(STRFTIME('%w', date) AS INTEGER) IN (1, 2, 3, 4, 5) AND date >= '" + str(start_date) + "' AND date <= '" + str(end_date) + "' ORDER BY date;").fetchall()
        return self.percent_change(rates)

    def exchange_rate_cycle_average(self, start_date, end_date):
        return self.db_con.execute("SELECT AVG(rate) FROM exchange_rates WHERE CAST(STRFTIME('%w', date) AS INTEGER) IN (1, 2, 3, 4, 5) AND date >= '" + str(start_date) + "' AND date <= '" + str(end_date) + "';").fetchone()[0]

    # Exchange rate futures

    def exchange_future_cycle_change(self, start_date, end_date):
        rates = self.db_con.execute("SELECT date, settle FROM exchange_futures WHERE CAST(STRFTIME('%w', date) AS INTEGER) IN (1, 2, 3, 4, 5) AND date >= '" + str(start_date) + "' AND date <= '" + str(end_date) + "' ORDER BY date;").fetchall()
        return self.percent_change(rates)

    # Oil prices

    def oil_prices(self, start_date, end_date, offset=-62, percentage=False):
        data = self.db_con.execute("SELECT * FROM oil_prices WHERE CAST(STRFTIME('%w', date) AS INTEGER) IN (1, 2, 3, 4, 5) AND date > '{0}' AND date <= '{1}' ORDER BY date;".format(start_date, end_date)).fetchall()
        data = [o[2] for o in data]
        if percentage:
            data = self.percent_change_daily(data)
        return data[offset:]

    def oil_price_cycle_change(self, start_date, end_date):
        prices = self.db_con.execute("SELECT date, price FROM oil_prices WHERE CAST(STRFTIME('%w', date) AS INTEGER) IN (1, 2, 3, 4, 5) AND date >= '" + str(start_date) + "' AND date <= '" + str(end_date) + "' ORDER BY date;").fetchall()
        return self.percent_change(prices)

    def oil_price_cycle_average(self, start_date, end_date):
        return self.db_con.execute("SELECT AVG(price) FROM oil_prices WHERE CAST(STRFTIME('%w', date) AS INTEGER) IN (1, 2, 3, 4, 5) AND date >= '" + str(start_date) + "' AND date <= '" + str(end_date) + "';").fetchone()[0]

    # Oil Futures

    def oil_future_cycle_change(self, start_date, end_date):
        rates = self.db_con.execute("SELECT date, settle FROM oil_futures WHERE CAST(STRFTIME('%w', date) AS INTEGER) IN (1, 2, 3, 4, 5) AND date >= '" + str(start_date) + "' AND date <= '" + str(end_date) + "' ORDER BY date;").fetchall()
        return self.percent_change(rates)

    # Fuel Prices

    def fuel_prices(self):
        return self.db_con.execute("SELECT * FROM fuel_prices ORDER BY date;").fetchall()

    def fuel_price_cycle_change(self, start_date, end_date):
        prices = self.db_con.execute("SELECT date, basic_fuel_price FROM fuel_prices WHERE date = '" + str(start_date) + "' OR date = '" + str(end_date) + "' ORDER BY date;").fetchall()
        return self.percent_change(prices)

    def fuel_price_month_value(self, start_date):
        res = self.db_con.execute("SELECT basic_fuel_price FROM fuel_prices WHERE date = '" + str(start_date) + "';").fetchone()
        if res is None:
            return 0.0
        else:
            return self.db_con.execute("SELECT basic_fuel_price FROM fuel_prices WHERE date = '" + str(start_date) + "';").fetchone()[0]
