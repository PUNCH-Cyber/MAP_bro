import numpy as np
import pandas as pd


def linear_val_func(vals = pd.DataFrame(index=[0],columns = [0]),weights = np.array([1,1,1]),decay = 0.9): #vals can be dataframe or series
    if vals.ndim == 2: #DataFrame
        val_tot = vals.values[:,1:] * weights[1:]					# make use of broadcasting, avoid using age column.
        decay = np.power(decay, vals.values[:,0]).reshape([-1,1]) 	# decays vals according to age and decay factor of
    # DataStore, and reshape array to prep for next step.
    elif vals.ndim == 1: #Series
        val_tot = vals.values[1:] * weights[1:]					# make use of broadcasting, avoid using age column.
        decay = np.power(decay, vals.values[0]).reshape([-1,1]) 	# decays vals according to age and decay factor of
    # DataStore, and reshape array to prep for next step.
    val_tot = np.sum(val_tot * decay, axis = 1)					# Sum all values in row for val_tot per row
    return val_tot

class DataStore(object):
    def __init__(self, size = 10, frac = 1, val_weights = [1,1,1], val_func = linear_val_func, decay = 0.9,
                 vals = pd.DataFrame([0],columns=['value_label0']), df = pd.DataFrame([0],columns=['label0'])):

        self.size = size				# Number of lines that can be stored
        self.frac = frac				# How the value of data is weighted in this datastore
        self.val_func = val_func		# Function for determining total value from various value columns
        self.vals = vals				# The various value features for each line of data
        self.vals_tot = val_func(vals, val_weights, decay)	# Total weighted value for each line of data
        self.init_vals = vals.copy()						# The initial values of the data
        self.init_vals_tot = np.copy(self.vals_tot)			# Initial total weighted value
        self.val_weights = val_weights	# The weights associated with each value column
        self.decay = decay				# Value decay coefficient for
        self.df = df					# The actual data stored

    def update(self, vals, df):
        self.vals = vals
        self.df = df

    def save(self, val, val_tot, val_arg):
        self.vals_tot[val_arg] = val_tot
        self.vals.iloc[val_arg] = val
        return

    def retrieve(self):
        return

    def delete(self):
        return

class HotStore(DataStore): #E.g. Druid
    def __init__(self):
        super().__init__(size = 10, frac = 1, val_weights = [1,1,1], val_func = linear_val_func, decay = 0.9,
                         vals = pd.DataFrame([0],columns=['value_label0']), df = pd.DataFrame([0],columns=['label0']))

class WarmStore(DataStore): #E.g. Parquet on HDD
    def __init__(self):
        super().__init__(size = 10, frac = 1, val_weights = [1,1,1], val_func = linear_val_func, decay = 0.9,
                         vals = pd.DataFrame([0],columns=['value_label0']), df = pd.DataFrame([0],columns=['label0']))

class ColdStore(DataStore): #E.g. Glacier on AWS
    def __init__(self):
        super().__init__(size = 10, frac = 1, val_weights = [1,1,1], val_func = linear_val_func, decay = 0.9,
                         vals = pd.DataFrame([0],columns=['value_label0']), df = pd.DataFrame([0],columns=['label0']))

class Architecture(object): #Object used to specify total environment
    def __init__(self):

class LambdaArch(Architecture): #2 Hot + 1 Warm + 1 Cold
    def __init__(self):

class KappaArch(Architecture): #1 Hot + 1 Warm + 1 Cold
    def __init__(self):

