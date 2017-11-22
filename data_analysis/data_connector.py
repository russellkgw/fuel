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
        start_date = date(2003, 12, 28)  # 1995, 11, 28
        end_date = date(2004, 1, 27)  # 1995, 12, 27

        exchange_rate_changes = []

        while start_date <= self.current_date:
            rate_change = fd.exchange_rate_cycle_change(start_date, end_date)
            exchange_rate_changes.append(rate_change)

            start_date = start_date + relativedelta(months=1)
            end_date = end_date + relativedelta(months=1)

        fd.close_db_connection()
        return exchange_rate_changes

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
        start_date = date(2003, 12, 28)  # 1995, 11, 28
        end_date = date(2004, 1, 27)  # 1995, 12, 27

        oil_price_changes = []

        while start_date <= self.current_date:
            change_price = fd.oil_price_cycle_change(start_date, end_date)
            oil_price_changes.append(change_price)

            start_date = start_date + relativedelta(months=1)
            end_date = end_date + relativedelta(months=1)

        fd.close_db_connection()
        return oil_price_changes

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
    def fuel_month_changes(self):
        fd = FuelData(self.con_string)
        start_date = date(2004, 1, 1)
        end_date = date(2004, 2, 1)

        fuel_price_changes = []

        while start_date <= self.current_date:
            fuel_price_change = fd.fuel_price_cycle_change(start_date, end_date)
            fuel_price_changes.append(fuel_price_change)

            start_date = start_date + relativedelta(months=1)
            end_date = end_date + relativedelta(months=1)

        fd.close_db_connection()
        return fuel_price_changes
