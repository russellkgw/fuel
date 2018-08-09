from fuel_data import FuelData
from datetime import date
from dateutil.relativedelta import relativedelta
from dateutil.parser import parse
import numpy as np


class DataConnector(object):

    def __init__(self):
        self.con_string = '../db/development.sqlite3'
        self.current_date = date.today()

    def fuel_date_range(self, fuel_date, num_months=3):
        # fuel_date = str(fuel_date)
        # split_date = fuel_date.split('-')

        # import pdb; pdb.set_trace()

        end_date = parse(fuel_date)  # date(int(split_date[0]), int(split_date[1]), 27) - relativedelta(months=1)
        start_date = end_date - relativedelta(months=num_months)

        # import pdb; pdb.set_trace()

        return {'start_date': str(start_date).split(' ')[0], 'end_date': str(end_date).split(' ')[0]}


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

    def exchange_rates(self, fuel_date, percentage_change=False, seq_length=60, pre_set=0, pre_set_val=0.0):
        date_range = self.fuel_date_range(fuel_date)
        
        fd = FuelData(self.con_string)
        data = fd.exchange_rates(date_range['start_date'], date_range['end_date'], percentage=percentage_change, offset=seq_length, pre_set=pre_set, pre_set_val=pre_set_val)
        fd.close_db_connection()

        return data


    # Exchange futures
    # def exchange_future_month_changes(self):
    #     fd = FuelData(self.con_string)
    #     start_date = date(2003, 12, 28)  # add offset (2wks) ?  1995, 11, 28
    #     end_date = date(2004, 1, 27)  # add offset (2wks) ?  1995, 12, 27

    #     exchange_future_changes = []

    #     while start_date <= self.current_date:
    #         future_change = fd.exchange_future_cycle_change(start_date, end_date)
    #         exchange_future_changes.append(future_change)

    #         start_date = start_date + relativedelta(months=1)
    #         end_date = end_date + relativedelta(months=1)

    #     fd.close_db_connection()
    #     return exchange_future_changes

    def exchange_futures(self, fuel_date, percentage_change=False, seq_length=60, pre_set=0, pre_set_val=0.0):
        date_range = self.fuel_date_range(fuel_date)
        
        fd = FuelData(self.con_string)
        data = fd.exchange_future(date_range['start_date'], date_range['end_date'], percentage=percentage_change, offset=seq_length, pre_set=pre_set, pre_set_val=pre_set_val)
        fd.close_db_connection()

        return data

    # Oil prices
    # def oil_month_changes(self):
    #     fd = FuelData(self.con_string)
    #     start_date = date(1995, 11, 28)  # 1995, 11, 28 # 2003, 12, 28
    #     end_date = date(1995, 12, 27)  # 1995, 12, 27 # 2004, 1, 27

    #     oil_price_changes = []

    #     while start_date <= self.current_date:
    #         change_price = fd.oil_price_cycle_change(start_date, end_date)
    #         oil_price_changes.append(change_price)

    #         start_date = start_date + relativedelta(months=1)
    #         end_date = end_date + relativedelta(months=1)

    #     fd.close_db_connection()
    #     return oil_price_changes

    def oil_prices(self, fuel_date, percentage_change=False, seq_length=60, pre_set=0, pre_set_val=0.0):
        date_range = self.fuel_date_range(fuel_date)

        fd = FuelData(self.con_string)
        data = fd.oil_prices(date_range['start_date'], date_range['end_date'], percentage=percentage_change, offset=seq_length, pre_set=pre_set, pre_set_val=pre_set_val)
        fd.close_db_connection()
        return data
    
    # Oil Future
    # def oil_future_month_changes(self):
    #     fd = FuelData(self.con_string)
    #     start_date = date(2003, 12, 28)  # add offset (2wks) ?  1995, 11, 28
    #     end_date = date(2004, 1, 27)  # add offset (2wks) ?  1995, 12, 27

    #     oil_future_changes = []

    #     # import pdb; pdb.set_trace()

    #     while start_date <= self.current_date:
    #         change_future = fd.oil_future_cycle_change(start_date, end_date)
    #         oil_future_changes.append(change_future)

    #         # print('start date: ' + str(start_date) + ' end date: ' + str(end_date))

    #         start_date = start_date + relativedelta(months=1)
    #         end_date = end_date + relativedelta(months=1)

    #     fd.close_db_connection()
    #     return oil_future_changes

    def oil_futures(self, fuel_date, percentage_change=False, seq_length=60, pre_set=0, pre_set_val=0.0):
        date_range = self.fuel_date_range(fuel_date)

        fd = FuelData(self.con_string)
        data = fd.oil_future(date_range['start_date'], date_range['end_date'], percentage=percentage_change, offset=seq_length, pre_set=pre_set, pre_set_val=pre_set_val)
        fd.close_db_connection()
        return data

    # Fuel Prices
    # def fuel_month_changes(self, step=1):
    #     fd = FuelData(self.con_string)
    #     start_date = date(1995, 12, 3)  # 2004, 1, 1
    #     end_date = date(1996, 1, 3)  # 2004, 2, 1

    #     fuel_price_changes = []

    #     stop_date = date(2017, 8, 3)

    #     while start_date <= stop_date:
    #         dpp = DatePricePair(end_date, fd.fuel_price_cycle_change(start_date, end_date))
    #         fuel_price_changes.append(dpp)
    #         start_date = start_date + relativedelta(months=1)
    #         end_date = end_date + relativedelta(months=1)

    #     fd.close_db_connection()
    #     return fuel_price_changes

    def fuel_prices_dates(self, percentage_change=False, normilize=True, start_date=None, end_date=None, flatten=True, lin=False, data_set='training', seq_length=60, pre_set=0, pre_set_val=0.0):
        # import pdb; pdb.set_trace()
        data = None
        if percentage_change:
            data = self.fuel_month_changes()
        else:
            fd = FuelData(self.con_string)
            fuel_prices = fd.fuel_prices(start_date, data_set)
            fd.close_db_connection()
            data = [DatePricePair(fp[3], fp[1]) for fp in fuel_prices]

        data = self.get_data(data, flatten=flatten, percentage_change=percentage_change, normilize=normilize, lin=lin, seq_length=seq_length, pre_set=pre_set, pre_set_val=pre_set_val)

        x_array = []
        y_array = []

        for item in data:
            x_array.append(item['x'])
            y_array.append(item['y'][0])

        return x_array, y_array


    def get_data(self, data, flatten, percentage_change, normilize, lin, seq_length, pre_set, pre_set_val):
        data_map = []
        exr_min, exr_max = 1000.0, 0.0
        oil_min, oil_max = 1000.0, 0.0
        exr_f_min, exr_f_max = 1000.0, 0.0
        oil_f_min, oil_f_max = 1000.0, 0.0
        fp_min, fp_max = 1000.0, 0.0

        for fp in data:
            exr_data = self.exchange_rates(fp.date, percentage_change=percentage_change, seq_length=seq_length, pre_set=pre_set, pre_set_val=pre_set_val)  # percentage=True
            for e in exr_data:
                if e < exr_min:
                    exr_min = e
                if e > exr_max:
                    exr_max = e

            oil_data = self.oil_prices(fp.date, percentage_change=percentage_change, seq_length=seq_length, pre_set=pre_set, pre_set_val=pre_set_val)  # percentage=True
            for o in oil_data:
                if o < oil_min:
                    oil_min = o
                if o > oil_max:
                    oil_max = o

            exr_f_data = self.exchange_futures(fp.date, percentage_change=percentage_change, seq_length=seq_length, pre_set=pre_set, pre_set_val=pre_set_val)
            for ef in exr_f_data:
                if ef < exr_f_min:
                    exr_f_min = ef
                if ef > exr_f_max:
                    exr_f_max = ef

            oil_f_data = self.oil_futures(fp.date, percentage_change=percentage_change, seq_length=seq_length, pre_set=pre_set, pre_set_val=pre_set_val)
            for of in oil_f_data:
                if of < oil_f_min:
                    oil_f_min = of
                if of > oil_f_max:
                    oil_f_max = of

            if fp.price < fp_min:
                    fp_min = fp.price
            if fp.price > fp_max:
                fp_max = fp.price

            data_map.append({'date': fp.date, 'y': [fp.price], 'exr': exr_data, 'oil': oil_data, 'exr_f': exr_f_data, 'oil_f': oil_f_data})

        print('fp min: ' + str(fp_min)), print('fp max: ' + str(fp_max))
        print('ex min: ' + str(exr_min)), print('ex max: ' + str(exr_max))
        print('o min: ' + str(oil_min)), print('o max: ' + str(oil_max))
        print('ef min: ' + str(exr_f_min)), print('ef max: ' + str(exr_f_max))
        print('of min: ' + str(oil_f_min)), print('of max: ' + str(oil_f_max))
        
        feed_data = []
        for d in data_map:
            y = self.norm_array(d['y'], fp_min, fp_max) if normilize else d['y']
            x1 = self.norm_array(d['exr'], exr_min, exr_max) if normilize else d['exr']
            x2 = self.norm_array(d['oil'], oil_min, oil_max) if normilize else d['oil']
            x3 = self.norm_array(d['exr_f'], exr_f_min, exr_f_max) if normilize else d['exr_f']
            x4 = self.norm_array(d['oil_f'], oil_f_min, oil_f_max) if normilize else d['oil_f']

            if flatten:
                x = np.append(x1, x2)  # .tolist()
                x = np.append(x, x3)
                x = np.append(x, x4)
            else:
                x = np.column_stack((x1, x2, x3, x4))

            feed_data.append({'date': d['date'], 'x': x, 'y': y})
        
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
