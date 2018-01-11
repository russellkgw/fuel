from fuel_data import FuelData
from datetime import date
from dateutil.relativedelta import relativedelta


class DataConnector(object):

    def __init__(self):
        self.con_string = '../db/development.sqlite3'
        self.current_date = date.today()

    # Exchange rates
    def exchange_month_changes(self):
        fd = FuelData(self.con_string)
        start_date = date(1995, 11, 28)  # 1995, 11, 28 # 2003, 12, 28
        end_date = date(1995, 12, 27)  # 1995, 12, 27 # 2004, 1, 27

        exchange_rate_changes = []

        while start_date <= self.current_date:
            rate_change = fd.exchange_rate_cycle_change(start_date, end_date)
            exchange_rate_changes.append(rate_change)

            start_date = start_date + relativedelta(months=1)
            end_date = end_date + relativedelta(months=1)

        fd.close_db_connection()
        return exchange_rate_changes

    def exchange_rates(self, fuel_date):
        fuel_date = str(fuel_date)
        fd = FuelData(self.con_string)
        split_date = fuel_date.split('-')
        end_date = date(int(split_date[0]), int(split_date[1]), 27) - relativedelta(months=1)
        start_date = end_date - relativedelta(months=3)
        data = fd.exchange_rates(str(start_date), str(end_date), percentage=True)
        fd.close_db_connection()

        return data


    # Exchange futures
    def exchange_future_month_changes(self):
        fd = FuelData(self.con_string)
        start_date = date(2003, 12, 28)  # add offset (2wks) ?  1995, 11, 28
        end_date = date(2004, 1, 27)  # add offset (2wks) ?  1995, 12, 27

        exchange_future_changes = []

        while start_date <= self.current_date:
            future_change = fd.exchange_future_cycle_change(start_date, end_date)
            exchange_future_changes.append(future_change)

            start_date = start_date + relativedelta(months=1)
            end_date = end_date + relativedelta(months=1)

        fd.close_db_connection()
        return exchange_future_changes

    # Oil prices
    def oil_month_changes(self):
        fd = FuelData(self.con_string)
        start_date = date(1995, 11, 28)  # 1995, 11, 28 # 2003, 12, 28
        end_date = date(1995, 12, 27)  # 1995, 12, 27 # 2004, 1, 27

        oil_price_changes = []

        while start_date <= self.current_date:
            change_price = fd.oil_price_cycle_change(start_date, end_date)
            oil_price_changes.append(change_price)

            start_date = start_date + relativedelta(months=1)
            end_date = end_date + relativedelta(months=1)

        fd.close_db_connection()
        return oil_price_changes

    def oil_prices(self, fuel_date):
        fuel_date = str(fuel_date)
        fd = FuelData(self.con_string)
        split_date = fuel_date.split('-')
        end_date = date(int(split_date[0]), int(split_date[1]), 27) - relativedelta(months=1)
        start_date = end_date - relativedelta(months=3)
        data = fd.oil_prices(str(start_date), str(end_date), percentage=True)
        fd.close_db_connection()
        return data
    
    # Oil Future
    def oil_future_month_changes(self):
        fd = FuelData(self.con_string)
        start_date = date(2003, 12, 28)  # add offset (2wks) ?  1995, 11, 28
        end_date = date(2004, 1, 27)  # add offset (2wks) ?  1995, 12, 27

        oil_future_changes = []

        # import pdb;
        # pdb.set_trace()

        while start_date <= self.current_date:
            change_future = fd.oil_future_cycle_change(start_date, end_date)
            oil_future_changes.append(change_future)

            # print('start date: ' + str(start_date) + ' end date: ' + str(end_date))

            start_date = start_date + relativedelta(months=1)
            end_date = end_date + relativedelta(months=1)

        fd.close_db_connection()
        return oil_future_changes

    # Fuel Prices
    def fuel_month_changes(self, step=1):
        fd = FuelData(self.con_string)
        start_date = date(1995, 12, 3)  # 2004, 1, 1
        end_date = date(1996, 1, 3)  # 2004, 2, 1

        fuel_price_changes = []

        stop_date = date(2017, 8, 3)

        while start_date <= stop_date:
            dpp = DatePricePair(end_date, fd.fuel_price_cycle_change(start_date, end_date))
            fuel_price_changes.append(dpp)
            start_date = start_date + relativedelta(months=1)
            end_date = end_date + relativedelta(months=1)

        fd.close_db_connection()
        return fuel_price_changes

    def fuel_prices_dates(self, start_date=None, end_date=None):
        fd = FuelData(self.con_string)
        fuel_prices = fd.fuel_prices()
        fd.close_db_connection()
        return [DatePricePair(fp[3], fp[1]) for fp in fuel_prices]


class DatePricePair():
    def __init__(self, date, price):
        self.date = date
        self.price = price
