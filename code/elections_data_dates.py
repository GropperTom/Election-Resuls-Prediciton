import pandas as pd
import numpy as np
from datetime import datetime

#INPUT_FNAME_BY_DATES = r'C:\Users\Student\PycharmProjects\untitled\2006_fixed.csv'
INPUT_FNAME_BY_DATES = 'C:\\Users\\dom\\PycharmProjects\\mnist\\elections_data\\2006_fixed.csv'

NGRAM_SZ = 3

def arrange_by_dates(fname):
    # Init
    df_csv = pd.read_csv(fname)
    # Remove some columns
    for cname in ["prev_elections", "12", "14", "15", "18"]:
        del df_csv[cname]
    days_cols = df_csv.columns[1:-1]
    curr_cname = ''
    same_day_lst = []
    all_data_lst = []
    date_by_day_lst = []
    cnt = 0
    for cname in days_cols.values:
        cnt += 1
        if curr_cname == cname.split('.')[0]:
            same_day_lst.append(df_csv[cname].values)
        else:
            if len(same_day_lst) > 0:
                all_data_lst.append(np.array(same_day_lst).mean(axis=0))
            curr_cname = cname.split('.')[0]
            date_by_day_lst.append(curr_cname)
            same_day_lst = [df_csv[cname].values]
    # The last one:
    if len(same_day_lst) > 0:
        all_data_lst.append(np.array(same_day_lst).mean(axis=0))

    return df_csv, \
           df_csv['miflaga'].values, \
           df_csv['result'].values, \
           np.array(all_data_lst), \
           date_by_day_lst

class PollsDataDates(object):
    def __init__(self, fname = INPUT_FNAME_BY_DATES, train_percent = None, ngram_sz = NGRAM_SZ):
        # Init.
        self.ngram_sz = ngram_sz
        self.train_percent = train_percent
        self.train = DataSet()
        self.test = DataSet()
        # Read & set up.
        self.fdata, self.parties_names, self.actual_results, self.xs_by_day, self.dates = arrange_by_dates(fname)
        self.parties_num = len(self.parties_names)
        self.last_ngram = self.xs_by_day[-self.ngram_sz:]
        self.set_data()

    def get_polls_data(self, party_id):
        return self.xs_by_day[party_id]

    def get_party_name(self, party_id):
        return self.parties_names[party_id]

    def get_actual_result(self, party_id):
        return self.actual_results[party_id]

    def set_data(self):
        # Go over weeks, each 3 weeks would be a feature
        self.xs = []
        self.ys = []
        for j in range(len(self.xs_by_day) - self.ngram_sz):
            line = []
            for i in range(self.ngram_sz):
                line.append(self.xs_by_day[j + i])
            self.ys.append(self.xs_by_day[j + self.ngram_sz])
            self.xs.append(line)
        # Split to train & test.
        self.split_train_test()

    def split_train_test(self, ):
        train_mask = np.array(self.get_train_mask())
        self.test_ids = np.argwhere(train_mask==False).flatten()
        self.train = DataSet(np.array(self.xs)[train_mask], np.array(self.ys)[train_mask])
        self.test = DataSet(np.array(self.xs)[~train_mask], np.array(self.ys)[~train_mask])
        self.base_mse = np.mean([(self.test.x.values[i][-1] - self.test.y.values[i])**2 for i in range(self.test.samples_num)])


    def get_train_mask(self, ):
        train_mask = np.ones(len(self.xs), dtype=np.bool)
        if self.train_percent is not None:
            # Else take train_percent indices, randomly.
            false_indices = np.random.random_integers(low=0, high=len(self.xs) - 1, size=int((len(self.xs) - 1) * (1-self.train_percent)))
            train_mask[np.unique(false_indices)] = False  # exclude duplicates + order
        # Set train and test elections_data sets.
        train_mask[-1] = False
        return train_mask

class DataSet(object):
    def __init__(self, x_arr = np.array([]), y_arr = np.array([])):
        self.x = CoordDataSet(x_arr)
        self.y = CoordDataSet(y_arr)

    @property
    def features_num(self):
        return self.x.cols

    @property
    def samples_num(self):
        return self.x.rows


class CoordDataSet(object):
    def __init__(self, arr):
        self._values = arr
        self._values.flags.writeable = False

    @property
    def values(self):
        return self._values

    @property
    def rows(self):
        return len(self.values)

    @property
    def cols(self):
        return len(self.values[0])

    def get_record(self, idx):
        return self.values[idx]

def a2s(a):
    return " ".join([str(a[i]) for i in range(len(a))])

def validate_test_data(fdata):
    print("*************************************")
    print("*************************************")
    print("*************************************")
    print("test ids: %s" % fdata.test_ids)
    for i, tid in enumerate(fdata.test_ids):
        print("**********index: %d" % tid)
        print("all_data x: %s" % a2s(fdata.xs_by_day[tid]))
        print("all_data y: %s" % a2s(fdata.xs_by_day[tid + 3]))
        print("test x: %s" % a2s(fdata.test.x.values[i]))
        print("test y: %s" % a2s(fdata.test.y.values[i]))

def sasha(fdata):
    return np.mean([fdata.test.x.values[i][-1] - fdata.test.y.values[i] for i in range(fdata.test.samples_num)])

