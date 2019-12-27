import numpy as np
import pandas as pd
from .data import *
from itertools import compress
import copy


def linear_val_func(vals = pd.DataFrame(index=[0],columns = [0]),weights = np.array([1,1,1]),decay = 0.9): #vals can be dataframe or series
    if isinstance(vals,pd.DataFrame): #DataFrame
        val_tot = vals.fillna(0).values[:,1:] * weights[1:]					# make use of broadcasting, avoid using age column.
        decay = np.power(decay, vals.fillna(0).values[:,0]).reshape([-1,1]) 	# decays vals according to age and decay factor of
                                                                            # DataStore, and reshape array to prep for next step.
        val_tot = np.sum(val_tot * decay, axis = 1)
    elif isinstance(vals,pd.Series): #Series
        val_tot = vals.fillna(0).values[1:].reshape(-1,1) * weights[1:]					# make use of broadcasting, avoid using age column.
        decay = np.power(decay, vals.fillna(0).values[0]).reshape([-1,1]) 	# decays vals according to age and decay factor of
        val_tot = np.sum(val_tot * decay, axis = 1)[0]
    elif isinstance(vals, np.ndarray):
        tmp = np.array(vals, dtype=np.float64)
        val_tot = np.nan_to_num(tmp[1:]).reshape(-1,1) * weights[1:]					# make use of broadcasting, avoid using age column.
        decay = np.power(decay, np.nan_to_num(tmp[0])).reshape([-1,1]) 	# decays vals according to age and decay factor of
        val_tot = np.sum(val_tot * decay, axis = 1)[0]
    elif isinstance(vals, list):
        print('ERROR')

    return val_tot

def single_linear_val_func(vals = pd.DataFrame(index=[0],columns = [0]),val_ind = 1,decay = 0.9): #vals can be dataframe or series
    if isinstance(vals,pd.DataFrame): #DataFrame
        val_tot = vals.fillna(0).values[:,val_ind]
        decay = 1#np.power(decay, vals.fillna(0).values[:,0]).reshape([-1,1]) 	# decays vals according to age and decay factor of
        # DataStore, and reshape array to prep for next step.
        val_tot = np.sum(val_tot * decay, axis = 1)
    elif isinstance(vals,pd.Series): #Series
        val_tot = vals.fillna(0).values[val_ind].reshape(-1,1)					# make use of broadcasting, avoid using age column.
        #print('serser', vals.fillna(0).values)
        decay = 1#np.power(decay, vals.fillna(0).values[0]).reshape([-1,1]) 	# decays vals according to age and decay factor of
        #print('sersum',val_tot,np.sum(val_tot * decay, axis = 1))
        val_tot = np.sum(val_tot * decay, axis = 1)[0]
    elif isinstance(vals, np.ndarray):
        tmp = np.array(vals, dtype=np.float64) #might need to change this to deepcopy
        #print('listlist', tmp)
        val_tot = np.nan_to_num(tmp[val_ind]).reshape(-1,1)					# make use of broadcasting, avoid using age column.
        decay = 1#np.power(decay, np.nan_to_num(tmp[0])).reshape([-1,1]) 	# decays vals according to age and decay factor of
        val_tot = np.sum(val_tot * decay, axis = 1)[0]
        #print('listsum',val_tot)
    elif isinstance(vals, list):
        print('ERROR')

    return val_tot

def squared_val_func(vals = pd.DataFrame(index=[0],columns = [0]),weights = np.array([1,1,1]),decay = 0.9):
    pass

def max_val_func(vals = pd.DataFrame(index=[0],columns = [0]),weights = np.array([1,1,1]),decay = 0.9):
    pass

class DataStore(object):
    def __init__(self, id_num = 1, size = 10, frac = 1, val_weights = np.array([1,1,1]), val_func = linear_val_func, decay = 0.9,
                 val = pd.DataFrame(index = np.arange(10),columns=['value_label0']), rplan = np.mgrid[0:10, 1:4][1].astype(int),
                 ind = np.zeros(10).astype(int),expir = 20,data = pd.DataFrame([0],columns=['label0'])):


        self.id_num = id_num            # Identification number of datastore
        self.size = size				# Number of lines that can be stored
        self.frac = frac				# How the value of data is weighted in this datastore

        self.val_tot = [np.nan for x in range(self.size)]	# Total weighted value for each line of data
        self.init_val = val.copy()						# The initial values of the data
        self.init_val_tot = np.copy(self.val_tot)			# Initial total weighted value
        self.val_weights = val_weights	# The weights associated with each value column
        self.decay = decay				# Value decay coefficient for time decay
        self.expir = expir              # Expiration times associated with data retention plan
        self.val_func = lambda val: self.frac*val_func(val,self.val_weights,self.decay)		# Function for determining total value from various value columns -->>>> for single_linear_val_func lambda val: self.frac*val_func(val,1,self.decay)#

        self.dataBatch = dataBatch(data,val,self.val_tot,ind,rplan) #Data and metadata stored in this dataStore

    def evaluate(self, val,val_arg, all_ds,names):

        # Current DataStore is full!
        curr_rplan_arg = [x for x in range(len(self.dataBatch.batch[val_arg].rplan)) if self.dataBatch.batch[val_arg].rplan[x] == self.id_num][0]
        next_ds_id = self.dataBatch.batch[val_arg].rplan[curr_rplan_arg+1] # Find the next DataStore in this data's retention plan
        val_tot = self.val_func(val)
        if next_ds_id == 0: #Next step is deletion
            reward = 0#-val_tot
        else: #Next step is another DataStore
            next_ds = all_ds[names[next_ds_id]] # Grab DataStore associated with action
            reward = next_ds.val_func(val) #Reward for saving current metaData to dataStore

            next_val_arg = np.argmin(next_ds.dataBatch.get('val_tot',1), axis=0)
            low_val_tot = next_ds.dataBatch.batch[next_val_arg].metaData.val_tot

            if not np.isnan(low_val_tot): # If dataStore is full. this might not be 100% fool-proof though
                low_val = next_ds.dataBatch.batch[next_val_arg].metaData.val #low_val in dataStore
                unweighted_low_val = low_val/next_ds.frac #Need to remove old frac
                reward +=  -low_val_tot + next_ds.evaluate(unweighted_low_val, next_val_arg,all_ds,names)

            # Reward is the new value plus the cascade of rewards caused by transferring the old value
            # New value replaces old value
            next_ds.dataBatch.batch[next_val_arg].metaData.val = val
            next_ds.dataBatch.batch[next_val_arg].metaData.val_tot = val_tot #MIGHT NEED TO CHANGE THIS as this val_tot might be old - 12/27/19
            next_ds.dataBatch.batch[next_val_arg].metaData.ind = curr_rplan_arg + 1
        return reward

    def get_expir(self):
        val = self.dataBatch.get('val')
        expired = (val['Age']>=self.expir) & (val['Age'].notna())
        expired_dis = list(compress(self.dataBatch.batch, expired.values))
        #print(f'{len(expired_dis)} rows expired in {self.id_num}',val)
        data = pd.DataFrame(index = np.arange(2), columns = val.columns) # Empty dataframe uesed to empty row that has decayed out
        for i in np.arange(len(self.dataBatch.batch)): #Seems slow. find a better implementation
            if expired.values[i]:
                self.dataBatch.batch[i] = dataItem(data.iloc[0],data.iloc[0],np.nan,0,[1,2,3,0])

        return expired_dis

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

