import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import matplotlib as plt
import pandas

class broEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    # __init__ is essentially pointless, the program is designed to be run from __myinit__
    def __init__(self):
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Discrete(3)
        N_database = 1
        N_batch = 1
        self.N_database = N_database
        self.N_batch = N_batch
        self.index = np.arange(N_database)
        self.col = pandas.read_csv("dns.col")
        self.df0 = pandas.DataFrame(index=self.index, columns=self.col.columns)
        self.values0 = np.zeros((N_database,2))
        self.values0_init = np.zeros((N_database,2))
        self.step_num = 0
        self.observation_space = spaces.Discrete(N_batch)
        pass
    
    def __myinit__(self,    N_database = 10,        # Size of the database
                            columns = "dns.col",    # File with column labels
                            N_batch = 5):           # Number of new lines to try to add to the database
        # Actions #
        # 0 = Save
        # 1 = Delete
        self.action_space = spaces.Discrete(2)
        
        # Define size variables
        self.N_database = N_database
        self.N_batch = N_batch

        # Define the (blank) database
        # N_database blank rows
        index = np.arange(N_database)
        self.index = index
        # Read the column names from dns.col
        col = pandas.read_csv("dns.col")
        self.col = col
        # Define the blank DataFrame
        self.df0 = pandas.DataFrame(index=index, columns=col.columns)

        # Define value table for the database
        # 0: Time since row was added
        # 1: Positive reward for adding the row to the table
        self.values0 = np.zeros((N_database,2))
        # Make copy of initial value table before playing the game
        self.values0_init = np.zeros((N_database,2))

        # Steps/Observations #
        # A single step is trying to save/delete/etc. a single line from the batch
        self.step_num = 0
        self.observation_space = spaces.Discrete(N_batch)
        pass
        
    # Full reset. Reset steps, database, and values
    def reset(self):
        self.step_num = 0
        self.df0 = pandas.DataFrame(index=self.index, columns=self.col.columns)
        self.values0 = np.zeros((self.N_database,2))
        return self.step_num
    
    # Batch reset. Used to start trying to save a new batch of lines
    def batch_reset(self):
        self.step_num = 0
        self.values0 = self.values0_init
        return self.step_num
    
    # An action is defined by:
    # Save: takie the value of a single bro line and try to replace the lowest (decayed) value from the value table
    # Delete: do nothing to the value table, lose value of deleted line as negative reward
    def _take_action(self, action, value):
        reward = 0
        
        if(action == 0):
            # Find the lowest value of the value table
            # axis=0 minimizes over columns, [1] is the column of values
            val_arg = np.argmin(self.values0, axis=0)[1]
            old_val = self.values0[val_arg][1]
            # Reward is the new value minus the old value
            # New value replaces old value
            reward = value - old_val
            self.values0[val_arg][1] = value
        elif(action == 1):
            reward = -value
        
        return reward
    
    # The step is doing the desired action and generating the RL variables
    def step(self, action, value):
        reward = self._take_action(action, value)
        self.step_num += 1

        obs = self.step_num
        done = self.step_num == self.N_batch
        return obs, reward, done, {}
    
    # Decay function for valeus
    # The value table will contain the current
    def decay_step(self, val):
        return val*0.9
    def inv_decay(self, val, n):
        return val*0.9**(-n)
    
    # The time step is where the time labels are increased by 1 and the values are decayed
    def time_step(self):
        self.values0[:,1] = self.decay_step(self.values0[:,1])
        self.values0[:,0] += 1

    def render(self, mode='human', close=False):
        time = self.values0[:,0]
        value = self.inv_decay(self.values0[:,1],self.values0[0])
        plt.scatter(time, value, alpha=1.0)
        plt.title('Age vs Value')
        plt.xlabel('Age')
        plt.ylabel('Value')
        plt.show()
        return 0