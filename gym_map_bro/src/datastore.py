import numpy as np
import pandas as pd
from .data import *


def linear_val_func(vals = pd.DataFrame(index=[0],columns = [0]),weights = np.array([1,1,1]),decay = 0.9): #vals can be dataframe or series
    if isinstance(vals,pd.DataFrame): #DataFrame
        val_tot = vals.fillna(0).values[:,1:] * weights[1:]					# make use of broadcasting, avoid using age column.
        decay = np.power(decay, vals.fillna(0).values[:,0]).reshape([-1,1]) 	# decays vals according to age and decay factor of
    # DataStore, and reshape array to prep for next step.
    elif isinstance(vals,pd.Series): #Series
        val_tot = vals.fillna(0).values[1:] * weights[1:]					# make use of broadcasting, avoid using age column.
        decay = np.power(decay, vals.fillna(0).values[0]).reshape([-1,1]) 	# decays vals according to age and decay factor of
    elif isinstance(vals, np.ndarray):
        val_tot = np.nan_to_num(vals[1:]) * weights[1:]					# make use of broadcasting, avoid using age column.
        decay = np.power(decay, np.nan_to_num(vals[0])).reshape([-1,1]) 	# decays vals according to age and decay factor of
    elif isintance(vals, list):
        print('ERROR')
    # DataStore, and reshape array to prep for next step.
    val_tot = np.sum(val_tot * decay, axis = 1)					# Sum all values in row for val_tot per row
    return val_tot

def squared_val_func(vals = pd.DataFrame(index=[0],columns = [0]),weights = np.array([1,1,1]),decay = 0.9):
    pass

def max_val_func(vals = pd.DataFrame(index=[0],columns = [0]),weights = np.array([1,1,1]),decay = 0.9):
    pass

class DataStore(object):
    def __init__(self, id_num = 1, size = 10, frac = 1, val_weights = np.array([1,1,1]), val_func = linear_val_func, decay = 0.9,
                 val = pd.DataFrame(index = np.arange(10),columns=['value_label0']), rplan = np.mgrid[0:10, 1:4][1],
                 ind = np.zeros(10),expir = np.ones(10)*20,data = pd.DataFrame([0],columns=['label0'])):


        self.id_num = id_num            # Identification number of datastore
        self.size = size				# Number of lines that can be stored
        self.frac = frac				# How the value of data is weighted in this datastore

        self.val_tot = val_func(val, val_weights, decay)	# Total weighted value for each line of data
        self.init_val = val.copy()						# The initial values of the data
        self.init_val_tot = np.copy(self.val_tot)			# Initial total weighted value
        self.val_weights = val_weights	# The weights associated with each value column
        self.decay = decay				# Value decay coefficient for time decay
        self.expir = expir              # Expiration times associated with data retention plan
        self.val_func = lambda val: val_func(val,self.val_weights,self.decay)		# Function for determining total value from various value columns

        self.dataBatch = dataBatch(data,val,self.val_tot,ind,rplan) #Data and metadata stored in this dataStore

    def evaluate(self, val,val_arg, all_ds,names):

        if val_arg == -1: # Let's make -1 the flag that the data is decaying?
            pass
            curr_rplan_arg = np.argwhere(self.rplan == self.id_num)
            next_ds_id = self.rplan[val_arg][curr_rplan_arg+1] # Find the next DataStore in this data's retention plan
            val_tot = self.val_func(val,self.val_weights,self.decay)
            if next_ds_id == 0: #Next step is deletion
                reward = -val_tot
            else: #Next step is another DataStore
                next_ds = all_ds[names[next_ds_id]] # Grab DataStore associated with action
                next_val_arg = np.argmin(next_ds.val_tot, axis=0)
                low_val = next_ds.val[next_val_arg]

                if low_val != 0: # If next_ds is full. this might not be 100% fool-proof though
                    unweighted_low_val = low_val/next_ds.frac #Need to remove old frac
                    reward = self.frac*val_tot + next_ds.evaluate(unweighted_low_val, next_val_arg,all_ds,names)
                else:
                    reward = self.frac*val_tot + next_ds.save(val)

            # Reward is the new value plus the cascade of rewards caused by transferring the old value
            # New value replaces old value
            # Wait should this be happening in the evaluate step? BIG QUESTION!!!!!
            self.val_tot[val_arg] = val_tot
            self.val.iloc[val_arg] = val

        else: # Current DataStore is full! (currently same as decay version, but ultimately they will be different)
            curr_rplan_arg = np.argwhere(self.dataBatch.batch[val_arg].rplan == self.id_num)
            next_ds_id = int(self.dataBatch.batch[val_arg].rplan[curr_rplan_arg+1]) # Find the next DataStore in this data's retention plan
            val_tot = self.val_func(val)
            if next_ds_id == 0: #Next step is deletion
                reward = -val_tot
            else: #Next step is another DataStore
                next_ds = all_ds[names[next_ds_id]] # Grab DataStore associated with action
                reward = next_ds.frac*next_ds.val_func(val) #Reward for saving current metaData to dataStore

                next_val_arg = np.argmin(next_ds.dataBatch.get('val_tot',1), axis=0)
                low_val_tot = next_ds.dataBatch.batch[next_val_arg].metaData.val_tot

                if not np.isnan(low_val_tot): # If dataStore is full. this might not be 100% fool-proof though
                    low_val = next_ds.dataBatch.batch[next_val_arg].metaData.val #low_val in dataStore
                    unweighted_low_val = low_val/next_ds.frac #Need to remove old frac
                    reward +=  next_ds.evaluate(unweighted_low_val, next_val_arg,all_ds,names)

                # Reward is the new value plus the cascade of rewards caused by transferring the old value
                # New value replaces old value
                next_ds.dataBatch.batch[next_val_arg].metaData.val = val
                next_ds.dataBatch.batch[next_val_arg].metaData.val_tot = val_tot
                next_ds.dataBatch.batch[next_val_arg].metaData.ind = curr_rplan_arg + 1
        print('reward',reward )
        return reward


    def save(self, val): #Probably a better way to incorporate this behavior. Could do a lambda function of evaluate and just pass a special flag, like -2.
        val_arg = np.argmin(self.va_tot, axis=0)
        old_val_tot = self.val_tot[val_arg]
        # Reward is the new value minus the old value
        # New value replaces old value
        val_tot = self.frac * self.val_func(val,self.val_weights,self.decay)
        reward = val_tot - old_val_tot
        self.val_tot[val_arg] = val_tot
        self.val.iloc[val_arg] = self.frac*val

        return reward

    def get_expir(self):
        val = self.dataBatch.get('val')
        val_tot = self.dataBatch.get('val_tot')
        rplan = self.dataBatch.get('rplan')
        expired_data = val.loc[val['Age']>self.expir]
        expired_values = val_tot[val['Age']>self.expir]
        expired_rplan = rplan.loc[val['Age']>self.expir]
        return expired_data, expired_values, expired_rplan

class HotStore(DataStore): #E.g. Druid
    def __init__(self):
        super().__init__(size = 10, frac = 1, val_weights = np.array([1,1,1]), val_func = linear_val_func, decay = 0.9,
                         vals = pd.DataFrame([0],columns=['value_label0']), df = pd.DataFrame([0],columns=['label0']))

class WarmStore(DataStore): #E.g. Parquet on HDD
    def __init__(self):
        super().__init__(size = 10, frac = 1, val_weights = np.array([1,1,1]), val_func = linear_val_func, decay = 0.9,
                         vals = pd.DataFrame([0],columns=['value_label0']), df = pd.DataFrame([0],columns=['label0']))

class ColdStore(DataStore): #E.g. Glacier on AWS
    def __init__(self):
        super().__init__(size = 10, frac = 1, val_weights = np.array([1,1,1]), val_func = linear_val_func, decay = 0.9,
                         vals = pd.DataFrame([0],columns=['value_label0']), df = pd.DataFrame([0],columns=['label0']))

class Architecture(object): #Object used to specify total environment
    def __init__(self):
        pass

class LambdaArch(Architecture): #2 Hot + 1 Warm + 1 Cold
    def __init__(self):
        super().__init__()
        pass

class KappaArch(Architecture): #1 Hot + 1 Warm + 1 Cold
    def __init__(self):
        super().__init__()
        pass

