import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
import pandas
from gym import error, spaces, utils
from gym.utils import seeding

class broEnv(gym.Env):
	metadata = {'render.modes': ['human']}
	
	# __init__ is essentially pointless, the program is designed to be run from __myinit__
	def __init__(self):
		self.action_space = spaces.Discrete(3)
		self.compress_frac = 0.5
		self.observation_space = spaces.Discrete(3)
		N_database = 1
		N_batch = 1
		self.N_database = N_database
		self.N_batch = N_batch
		self.index = np.arange(N_database)
		self.col = pandas.read_csv("dns.col")
		self.df0 = pandas.DataFrame(index=self.index, columns=self.col.columns)
		self.df1 = pandas.DataFrame(index=self.index, columns=self.col.columns)
		self.values0 = np.zeros((N_database,2))
		self.values0_init = np.zeros((N_database,2))
		self.step_num = 0
		self.observation_space = spaces.Discrete(N_batch)
		pass
	
	def __myinit__(self,	N_database = 10,		# Size of the database
							columns = "dns.col",	# File with column labels
							N_batch = 5,			# Number of new lines to try to add to the database
							init_s = np.array([])):	# Initial database values (needs database initialization)
		# Actions #
		# 0 = Save
		# 1 = Compress
		# 2 = Delete
		self.action_space = spaces.Discrete(3)

		# Compression value fraction
		self.compress_frac = 0.5
		
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
		# Define the blank DataFrames
		# df0 is for "saving" while df1 is for "compressing"
		self.df0 = pandas.DataFrame(index=index, columns=col.columns)
		self.df1 = pandas.DataFrame(index=index, columns=col.columns)
		
		# Define value table for the database
		# 0: Time since row was added
		# 1: Positive reward for adding the row to the table
		if len(init_s) == 0:
			self.values0 = np.zeros((N_database,2))
			self.values1 = np.zeros((N_database,2))

			# Make copy of initial value table before playing the game
			self.values0_init = np.zeros((N_database,2))
			self.values1_init = np.zeros((N_database,2))
		else:
			self.values0 = init_s[0]
			self.values1 = init_s[1]
			
			self.values0_init = init_s[0]
			self.values1_init = init_s[1]

		# Steps/Observations #
		# A single step is trying to save/delete/etc. a single line from the batch
		self.step_num = 0
		self.observation_space = spaces.Discrete(N_batch)
		pass
		
	# Full reset. Reset steps, database, and values
	def reset(self):
		self.step_num = 0

		self.df0 = pandas.DataFrame(index=self.index, columns=self.col.columns)
		self.df1 = pandas.DataFrame(index=self.index, columns=self.col.columns)

		self.values0 = np.zeros((self.N_database,2))
		self.values1 = np.zeros((self.N_database,2))

		self.values0_init = np.zeros((self.N_database,2))
		self.values1_init = np.zeros((self.N_database,2))

		return self.step_num
	
	# Batch reset. Used to start trying to save a new batch of lines
	def batch_reset(self):
		self.step_num = 0
		self.values0 = np.copy(self.values0_init)
		self.values1 = np.copy(self.values1_init)
		return self.step_num
	
	# An action is defined by:
	# Save: take the value of a single bro line and try to replace the lowest (decayed) value from the value table
	# Delete: do nothing to the value table, lose value of deleted line as negative reward
	def _take_action(self, action, value):
		reward = 0
		if(action == 0):	# Save to df0 with full reward
			# Find the lowest value of the value table
			# axis=0 minimizes over columns, [1] is the column of values
			val_arg = np.argmin(self.values0, axis=0)[1]
			old_val = self.values0[val_arg][1]

			# Reward is the new value minus the old value
			# New value replaces old value
			reward = value - old_val
			self.values0[val_arg][1] = value
		if(action == 1):	# Save to df1 with discounted reward
			# Find the lowest value of the value table
			# axis=0 minimizes over columns, [1] is the column of values
			val_arg = np.argmin(self.values1, axis=0)[1]
			old_val = self.values1[val_arg][1]

			# Reward is the new value minus the old value
			# New value replaces old value
			discount_value = self.compress_frac*value
			reward = discount_value - old_val
			self.values1[val_arg][1] = discount_value
		elif(action == 2):	# Delete with negative value reward
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
	def decay_step(self, val, rate):
		val = val*rate
	def inv_decay(self, val, n, rate):
		return val*rate**(-n)
	
	# The time step is where we complete the actions recommended by the agent.
	# 1. Loop through recommendations and apply to database
	# 2. Time labels are increased by 1 and the values are decayed
	# 3. Values are applied to initial for next batch
	def time_step(self, batch, values, actions):
		#self.batch_reset()
		
		for i in range(0,self.N_batch):
			if(actions[i] == 0):
				# Find the row we want to replace
				rep_row = np.argmin(self.values0_init, axis=0)[1]
				
				# Replace the value row
				self.values0_init[rep_row, 0] = 0
				self.values0_init[rep_row, 1] = values[i]

				# Replace the database row
				dns_batch = pandas.read_csv("dns.log")
				dns_line = dns_batch.values[i]
				self.df0.loc[rep_row] = dns_line
			if(actions[i] == 1):
				# Find the row we want to replace
				rep_row = np.argmin(self.values1_init, axis=0)[1]
				
				# Replace the value row
				self.values1_init[rep_row, 0] = 0
				self.values1_init[rep_row, 1] = self.compress_frac*values[i]

				# Replace the database row
				dns_batch = pandas.read_csv("dns.log")
				dns_line = dns_batch.values[i]
				self.df1.loc[rep_row] = dns_line
		
		self.decay_step(self.values0_init[:,1], 0.9)
		self.decay_step(self.values1_init[:,1], 0.95)

		self.values0_init[:,0] += 1
		self.values1_init[:,0] += 1
		
		#new_values = self.values0
		#self.values0_init = new_values
	
	def render(self, mode='human', close=True):
		time0 = self.values0_init[:,0]
		value0 = self.inv_decay(self.values0_init[:,1],self.values0_init[:,0], 0.9)
		sub = plt.subplot()
		sub.scatter(time0, value0, color='b', alpha=1.0, label="Uncompressed")

		time1 = self.values1_init[:,0]
		value1 = self.inv_decay(self.values1_init[:,1],self.values1_init[:,0], 0.9)
		sub.scatter(time1, value1, color='r', alpha=1.0, label="Compressed")

		sub.set_title('Age vs Initial Value')
		sub.set_xlabel('Age')
		sub.set_ylabel('Value')
		sub.legend(loc=2)
		plt.show()
		plt.close()

		print_df = self.df0.copy()
		print_df['age'] = time0
		print_df['value'] = value0
		print("Uncompressed Database:")
		print(print_df[['uid', 'src', 'sport', 'dst', 'age', 'value']])

		print_df = self.df1.copy()
		print_df['age'] = time1
		print_df['value'] = value1
		print("Compressed Database:")
		print(print_df[['uid', 'src', 'sport', 'dst', 'age', 'value']])
		return 0
