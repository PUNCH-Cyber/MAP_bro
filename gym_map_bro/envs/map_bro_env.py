import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from gym import error, spaces, utils
from gym.utils import seeding
from gym_map_bro.src.datastore import *

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
		self.index0 = np.arange(N_database)
		self.index1 = np.arange(2*N_database)
		self.col = pd.read_csv("dns.col")
		self.df0 = pd.DataFrame(index=self.index0, columns=self.col.columns)
		self.df1 = pd.DataFrame(index=self.index1, columns=self.col.columns)
		self.values0 = np.zeros((N_database,2))
		self.values0_init = np.zeros((2*N_database,2))
		self.values1 = np.zeros((N_database,2))
		self.values1_init = np.zeros((2*N_database,2))
		self.step_num = 0
		self.observation_space = spaces.Discrete(N_batch)
		pass

	def __myinit__(self, env_config = #Eventually switch this with just None
	{
		"col" : "dns.col",
		"N_batch": 5,										# Number of new lines to try to add to the datastores each epoch
		"batch_stocahsitic": False,							# Whether or not the number of lines in each batch is constant (False) or not (True)
		"name": ['deletion','database','compressed','deep'],			# Names to identify different storage formats
		"link": [[],[],[],[]],
		"ds_size": [10, 20, 40],							# Number of lines in each datastore
		"ds_frac": [1, 0.5, 0.25],							# Value coefficient associated with each storage option
		"val_weight": [1,1,1],								# Weights applied to each value column
		"val_func": lambda val: linear_val_func(val,np.array[1,1,1],decay=1),# function for determining total value from various value columns
		"ds_decay": [0.9, 0.95, 0.99],						# Rate at which Value decays in each DataStore
		"vals": [pd.DataFrame(np.zeros((10,3)),columns=['Age','Key Terrain','Queries']),		# Values associated with each line of data
				   pd.DataFrame(np.zeros((20,3)),columns=['Age','Key Terrain','Queries']),
				   pd.DataFrame(np.zeros((40,3)),columns=['Age','Key Terrain','Queries'])],
		"init_policy": [np.hstack((np.mgrid[0:20, 1:4][1],np.zeros(20).reshape(-1,1))),
						np.hstack((np.mgrid[0:20, 1:4][1],np.zeros(20).reshape(-1,1))),
						np.hstack((np.mgrid[0:20, 1:4][1],np.zeros(20).reshape(-1,1)))], #Initially start with a hot to cold policy for data
		"init_expir": [np.ones((10,3))*20,np.ones((20,3))*20,np.ones((40,3))*20], #Data 20 time steps old must be re-evaluated
		"df": [pd.DataFrame(index = np.arange(10),columns=['label0']),		# Dataframes that hold actual datastore contents
			   pd.DataFrame(index = np.arange(20),columns=['label0']),
			   pd.DataFrame(index = np.arange(40),columns=['label0'])]
	} ):	# Initial database values (needs database initialization)

		# Actions #
		# 0 = Delete
		# 1 = Save to 1st DataStore
		# 2 = Save to 2nd DataStore
		# N = Save to Nth DataStore

		# Most of this doesn't need to be carried around in self. ok for now.
		self.col = pd.read_csv(env_config.get("col","dns.col"))
		# Define size variables
		self.N_batch = env_config.get("N_batch", 5)
		self.num_ds = len(self.ds_size)

		self.action_space = env_config.get("action_space", spaces.Discrete(self.num_ds))
		# Observations #
		self.observation_space = env_config.get("observation_space",spaces.Discrete(self.N_batch))
		# Size of the data storage options
		self.ds_size = env_config.get("ds_size",[10, 20, 40])
		self.ds_frac = env_config.get("ds_frac",[1, 0.5, 0.25])
		self.val_weight = env_config.get("val_weight",[1,1,1])
		self.val_func = env_config.get("val_func", lambda val: linear_val_func(val,np.array[1,1,1],decay=1)) # Note this is different from DataStore implementation
		self.ds_decay = env_config.get("ds_decay",[0.9, 0.95, 0.99])
		self.vals = env_config.get('vals',[pd.DataFrame(np.zeros((10,3)),columns=['Age','Key Terrain','Queries']),
											   pd.DataFrame(np.zeros((20,3)),columns=['Age','Key Terrain','Queries']),
											   pd.DataFrame(np.zeros((40,3)),columns=['Age','Key Terrain','Queries'])])
		self.df = env_config.get('df', [pd.DataFrame(index = np.arange(10),columns=self.col),
										pd.DataFrame(index = np.arange(20),columns=self.col),
										pd.DataFrame(index = np.arange(40),columns=self.col)])
		self.init_policy = env_config.get("init_policy",[np.mgrid[0:10, 1:4][1],np.mgrid[0:20, 1:4][1],np.mgrid[0:40, 1:4][1]])
		self.init_expir = env_config.get("init_expir",[np.ones(10)*20,np.ones(20)*20,np.ones(40)*20])
		self.ds = {}
		self.names = env_config.get("name",['deletion','database','compressed','deep'])
		for i in np.arange(self.num_ds):
			addDataStore(name[i+1], i, self.ds_size[i], self.ds_frac[i], self.val_weight, self.val_func, # i+1 to skip deletion name
						  self.ds_decay[i], self.vals[i], self.init_policy[i], self.init_expir[i], self.df[i])

		# Steps/Observations #
		# A single step is trying to save/delete/etc. a single line from the batch
		self.step_num = 0

		self.deleted = pd.DataFrame(index=[], columns=col.columns)
		self.del_val = []
		pass


	def addDataStore(self, name, id_num, size, frac, vals, policy, expir df):
		self.ds[name] = DataStore(id_num, size, frac, vals, policy, expir, df)


	# Batch reset. Used to start trying to save a new batch of lines
	def batch_reset(self):
		self.step_num = 0
		for i in np.arange(self.num_ds):
			self.values0 = np.copy(self.values0_init) #Is this necessary given DataStore.update?
		return self.step_num
	
	# An action is defined by:
	# Save: take the value of a single bro line and try to replace the lowest (decayed) value from the value table
	# Delete: do nothing to the value table, lose value of deleted line as negative reward
	def _take_action(self, action, val): # I think this is just evaluate with a val_arg of -1 case
		reward = 0
		if action == 0:
			reward = -self.val_func(val) #env val_func only takes val
		else:
			current_ds = self.ds[self.names[action]] # Grab DataStore associated with action
			val_arg = np.argmin(current_ds.vals_tot, axis=0)
			low_val_tot = current_ds.vals_tot[val_arg]
			low_val = current_ds.vals[val_arg]

			if low_val != 0: # If current_ds is full. this might not be 100% fool-proof though
				unweighted_low_val = low_val/current_ds.frac #Need to remove old frac
				reward = current_ds.evaluate(unweighted_low_val, val_arg,self.ds,self.names)
			else:
				reward = current_ds.save(val)

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
				
				if(self.values0_init[rep_row, 0] != 0):
					self.del_val.append(self.values0_init[rep_row])

				# Replace the value row
				self.values0_init[rep_row, 0] = 0
				self.values0_init[rep_row, 1] = values[i]

				# Replace the database row
				#dns_batch = pd.read_csv("dns.log")
				dns_line = batch.values[i]
				self.df0.loc[rep_row] = dns_line
			if(actions[i] == 1):
				# Find the row we want to replace
				rep_row = np.argmin(self.values1_init, axis=0)[1]

				if(self.values1_init[rep_row, 0] != 0):
					self.del_val.append(self.values1_init[rep_row])

				# Replace the value row
				self.values1_init[rep_row, 0] = 0
				self.values1_init[rep_row, 1] = self.compress_frac*values[i]

				# Replace the database row
				#dns_batch = pd.read_csv("dns.log")
				dns_line = batch.values[i]
				self.df1.loc[rep_row] = dns_line
			if(actions[i] == 2):
				# Find the rows in each table we work with
				rep_row0 = np.argmin(self.values0_init, axis=0)[1]
				rep_row1 = np.argmin(self.values1_init, axis=0)[1]

				if(self.values1_init[rep_row1, 0] != 0):
					self.del_val.append(self.values1_init[rep_row1])
				
				# Compress the firt sentry
				self.values1_init[rep_row1, 0] = self.values0_init[rep_row0, 0]
				self.values1_init[rep_row1, 1] = self.compress_frac*self.values0_init[rep_row0, 1]

				# Replace the first entry
				self.values0_init[rep_row0, 0] = 0
				self.values0_init[rep_row0, 1] = values[i]

				# Replace the compressed database row
				self.df1.loc[rep_row1] = self.df0.loc[rep_row0]

				# Replace the database row
				#dns_batch = pd.read_csv("dns.log")
				dns_line = batch.values[i]
				self.df0.loc[rep_row0] = dns_line
			#else:
				#dns_line = batch.values[i]
				#self.deleted.append(dns_line)
		self.decay_step(self.values0_init[:,1], 0.9)
		self.decay_step(self.values1_init[:,1], 0.95)

		self.values0_init[:,0] += 1
		self.values1_init[:,0] += 1
		
		#new_values = self.values0
		#self.values0_init = new_values
	
	def render(self, mode='human', out=0, close=True):
		time0 = self.values0_init[:,0]
		value0 = self.inv_decay(self.values0_init[:,1],self.values0_init[:,0], 0.9)
		time1 = self.values1_init[:,0]
		value1 = self.inv_decay(self.values1_init[:,1],self.values1_init[:,0], 0.9)

		if(out == 0):
			sub = plt.subplot()
			sub.scatter(time0, value0, color='b', alpha=1.0, label="Uncompressed")
			sub.scatter(time1, value1, color='r', alpha=1.0, label="Compressed")
			x_val = [x[0] for x in self.del_val]
			y_val = [x[1] for x in self.del_val]
			sub.scatter(np.array(x_val),-np.array(y_val),color="g", label="Deleted")

			sub.set_title('Age vs Initial Value')
			sub.set_xlabel('Age')
			sub.set_ylabel('Value')
			sub.legend(loc=2)
			plt.show()
			plt.close()
		elif(out == 1):
			print_df = self.df0.copy()
			print_df['age'] = time0
			print_df['value'] = value0
			print("Uncompressed Database:")
			print(print_df[['uid', 'src', 'sport', 'age', 'value']])

			print_df = self.df1.copy()
			print_df['age'] = time1
			print_df['value'] = value1
			print("Compressed Database:")
			print(print_df[['uid', 'src', 'sport', 'age', 'value']])
		return 0
