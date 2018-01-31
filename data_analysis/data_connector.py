from fuel_data import FuelData
from datetime import date
from dateutil.relativedelta import relativedelta
import numpy as np


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

    def exchange_rates(self, fuel_date, percentage_change=False):
        fuel_date = str(fuel_date)
        fd = FuelData(self.con_string)
        split_date = fuel_date.split('-')
        end_date = date(int(split_date[0]), int(split_date[1]), 27) - relativedelta(months=1)
        start_date = end_date - relativedelta(months=3)
        data = fd.exchange_rates(str(start_date), str(end_date), percentage=percentage_change)
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

    def oil_prices(self, fuel_date, percentage_change=False):
        fuel_date = str(fuel_date)
        fd = FuelData(self.con_string)
        split_date = fuel_date.split('-')
        end_date = date(int(split_date[0]), int(split_date[1]), 27) - relativedelta(months=1)
        start_date = end_date - relativedelta(months=3)
        data = fd.oil_prices(str(start_date), str(end_date), percentage=percentage_change)
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

    def fuel_prices_dates(self, percentage_change=False, normilize=True, start_date=None, end_date=None):
        # import pdb; pdb.set_trace()
        data = None
        if percentage_change:
            data = self.fuel_month_changes()
        else:
            fd = FuelData(self.con_string)
            fuel_prices = fd.fuel_prices()
            fd.close_db_connection()
            data = [DatePricePair(fp[3], fp[1]) for fp in fuel_prices]

        if normilize:
            data = self.get_and_normilize_data(data, percentage_change=percentage_change)

        x_array = []
        y_array = []

        for item in data:
            x_array.append(item['x'])
            y_array.append(item['y'][0])
        
        return x_array, y_array



    def get_and_normilize_data(self, data, percentage_change=False):
        data_map = []
        exr_min, exr_max = 1000.0, 0.0
        oil_min, oil_max = 1000.0, 0.0
        fp_min, fp_max = 1000.0, 0.0

        # Normilize the data
        for fp in data:
            exr_data = self.exchange_rates(fp.date, percentage_change=percentage_change)  # percentage=True
            for e in exr_data:
                if e < exr_min:
                    exr_min = e
                if e > exr_max:
                    exr_max = e

            oil_data = self.oil_prices(fp.date, percentage_change=percentage_change)  # percentage=True
            for o in oil_data:
                if o < oil_min:
                    oil_min = o
                if o > oil_max:
                    oil_max = o

            if fp.price < fp_min:
                    fp_min = fp.price
            if fp.price > fp_max:
                fp_max = fp.price

            data_map.append({'date': fp.date, 'y': [fp.price], 'exr': exr_data, 'oil': oil_data})

        print('fp min: ' + str(fp_min)), print('fp max: ' + str(fp_max))
        print('ex min: ' + str(exr_min)), print('ex max: ' + str(exr_max))
        print('o min: ' + str(oil_min)), print('o max: ' + str(oil_max))
        
        feed_data = []
        for d in data_map:
            y = self.norm_array(d['y'], fp_min, fp_max)
            x1 = self.norm_array(d['exr'], exr_min, exr_max)
            x2 = self.norm_array(d['oil'], oil_min, oil_max)

            x = np.append(x1, x2).tolist()

            x_new = []
            for i in x:
                x_new.append(i)

            y_new = []
            for i in y:
                y_new.append(i)

            feed_data.append({'date': d['date'], 'x': x_new, 'y': y_new})
        
        return feed_data

    def norm_array(self, input, min, max):
        res = []
        for i in input:  # range(len(input)):
            res.append((i - min) / (max - min))  # norm_data(input[i], min, max)
        return np.array(res)


class DatePricePair():
    def __init__(self, date, price):
        self.date = date
        self.price = price
