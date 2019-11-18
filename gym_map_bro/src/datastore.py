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

def squared_val_func(vals = pd.DataFrame(index=[0],columns = [0]),weights = np.array([1,1,1]),decay = 0.9):
    pass

def max_val_func(vals = pd.DataFrame(index=[0],columns = [0]),weights = np.array([1,1,1]),decay = 0.9):
    pass

class DataStore(object):
    def __init__(self, id_num = 1, size = 10, frac = 1, val_weights = np.array([1,1,1]), val_func = linear_val_func, decay = 0.9,
                 vals = pd.DataFrame([np.zeros(10)],columns=['value_label0']), policy = np.mgrid[0:10, 1:4][1],
                 expir = np.ones(10)*20,df = pd.DataFrame([0],columns=['label0'])):

        self.id_num = id_num            # Identification number of datastore
        self.size = size				# Number of lines that can be stored
        self.frac = frac				# How the value of data is weighted in this datastore
        self.val_func = val_func		# Function for determining total value from various value columns
        self.vals = vals				# The various value features for each line of data
        self.vals_tot = val_func(vals, val_weights, decay)	# Total weighted value for each line of data
        self.init_vals = vals.copy()						# The initial values of the data
        self.init_vals_tot = np.copy(self.vals_tot)			# Initial total weighted value
        self.val_weights = val_weights	# The weights associated with each value column
        self.decay = decay				# Value decay coefficient for time decay
        self.policy = policy            # Policy for where data moves to
        self.expir = expir              # Expiration times associated with data policy
        self.df = df					# The actual data stored

    def evaluate(self, val,val_arg, all_ds,names):

        if val_arg = -1: # Let's make -1 the sign that the data is decaying?
            curr_policy_arg = np.argwhere(self.policy == self.id_num)
            next_ds_id = self.policy[val_arg][curr_policy_arg+1] # Find the next DataStore in this data's policy
            val_tot = self.val_func(val,self.val_weights,self.decay)
            if next_ds_id == 0: #Next step is deletion
                reward = -val_tot
            else: #Next step is another DataStore
                next_ds = all_ds[names[next_ds_id]] # Grab DataStore associated with action
                next_val_arg = np.argmin(next_ds.vals_tot, axis=0)
                low_val_tot = next_ds.vals_tot[next_val_arg]
                low_val = next_ds.vals[next_val_arg]

                if low_val != 0: # If next_ds is full. this might not be 100% fool-proof though
                    unweighted_low_val = low_val/next_ds.frac #Need to remove old frac
                    reward = self.frac*val_tot + next_ds.evaluate(unweighted_low_val, next_val_arg,all_ds,names)
                else:
                    reward = self.frac*val_tot + next_ds.save(val)

            # Reward is the new value plus the cascade of rewards caused by transferring the old value
            # New value replaces old value
            self.vals_tot[val_arg] = val_tot
            self.vals.iloc[val_arg] = val

        else: # Current DataStore is full! (currently same as decay version, but ultimately they will be different)
            curr_policy_arg = np.argwhere(self.policy == self.id_num)
            next_ds_id = self.policy[val_arg][curr_policy_arg+1] # Find the next DataStore in this data's policy
            val_tot = self.val_func(val,self.val_weights,self.decay)
            if next_ds_id == 0: #Next step is deletion
                reward = -val_tot
            else: #Next step is another DataStore
                next_ds = all_ds[names[next_ds_id]] # Grab DataStore associated with action
                next_val_arg = np.argmin(next_ds.vals_tot, axis=0)
                low_val_tot = next_ds.vals_tot[next_val_arg]
                low_val = next_ds.vals[next_val_arg]

                if low_val != 0: # If next_ds is full. this might not be 100% fool-proof though
                    unweighted_low_val = low_val/next_ds.frac #Need to remove old frac
                    reward = self.frac*val_tot + next_ds.evaluate(unweighted_low_val, next_val_arg,all_ds,names)
                else:
                    reward = self.frac*val_tot + next_ds.save(low_val)

            # Reward is the new value plus the cascade of rewards caused by transferring the old value
            # New value replaces old value
            self.vals_tot[val_arg] = val_tot
            self.vals.iloc[val_arg] = val

        return reward


    def save(self, val): #Probably a better way to incorporate this behavior. Could do a lambda function of evaluate and just pass a special flag, like -2.
        val_arg = np.argmin(self.vals_tot, axis=0)
        old_val_tot = self.vals_tot[val_arg]
        # Reward is the new value minus the old value
        # New value replaces old value
        val_tot = self.frac * self.val_func(val,self.val_weights,self.decay)
        reward = val_tot - old_val_tot
        self.vals_tot[val_arg] = val_tot
        self.vals.iloc[val_arg] = self.frac*val

        return reward

    def retrieve(self):
        return

    def delete(self):
        return

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

