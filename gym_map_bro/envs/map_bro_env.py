import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from gym import error, spaces, utils
from gym.utils import seeding
from gym_map_bro.src.datastore import *
from gym_map_bro.src.data import *

class broEnv(gym.Env):
	metadata = {'render.modes': ['human']}
	
	# __init__ is essentially pointless, the program is designed to be run from __myinit__
	def __init__(self):
		#self.action_space = spaces.Discrete(3)
		#self.compress_frac = 0.5
		#self.observation_space = spaces.Discrete(3)
		#N_database = 1
		#N_batch = 1
		#self.N_database = N_database
		#self.N_batch = N_batch
		#self.index0 = np.arange(N_database)
		#self.index1 = np.arange(2*N_database)
		#self.col = pd.read_csv("dns.col")
		#self.df0 = pd.DataFrame(index=self.index0, columns=self.col.columns)
		#self.df1 = pd.DataFrame(index=self.index1, columns=self.col.columns)
		#self.values0 = np.zeros((N_database,2))
		#self.values0_init = np.zeros((2*N_database,2))
		#self.values1 = np.zeros((N_database,2))
		#self.values1_init = np.zeros((2*N_database,2))
		#self.step_num = 0
		#self.observation_space = spaces.Discrete(N_batch)
		pass

	def __myinit__(self, env_config = #Eventually switch this with just None
	{
		"col" : "dns.col",
		"N_batch": 5,										# Number of new lines to try to add to the datastores each epoch
		"batch_stocahsitic": False,							# Whether or not the number of lines in each batch is constant (False) or not (True)
		"name": ['deletion','Hot','Warm','Cold'],			# Names to identify different storage formats
		"ds_size": [10, 20, 40],							# Number of lines in each datastore
		"ds_frac": [1, 0.5, 0.25],							# Value coefficient associated with each storage option
		"val_weight": [np.array([1,1,1]),np.array([1,1,1]),np.array([1,1,1])],								# Weights applied to each value column
		"val_func": linear_val_func,# function for determining total value from various value columns
		"ds_decay": [0.9, 0.95, 0.99],						# Rate at which Value decays in each DataStore
		"vals": [pd.DataFrame(index = np.arange(10),columns=['Age','Key Terrain','Queries']),		# Values associated with each line of data
				   pd.DataFrame(index = np.arange(20),columns=['Age','Key Terrain','Queries']),
				   pd.DataFrame(index = np.arange(40),columns=['Age','Key Terrain','Queries'])],
		"init_rplan": [np.hstack((np.mgrid[0:10, 1:4][1].astype(int),np.zeros(10).reshape(-1,1).astype(int))),
						np.hstack((np.mgrid[0:20, 1:4][1].astype(int),np.zeros(20).reshape(-1,1).astype(int))),
						np.hstack((np.mgrid[0:40, 1:4][1].astype(int),np.zeros(40).reshape(-1,1).astype(int)))], #Initially start with a hot to cold retention plan for data
		"ind": [np.zeros(10).astype(int),np.zeros(20).astype(int),np.zeros(40).astype(int)], #All data is initialized to the first step of it's rplan
		"init_expir": [20,20,20], #Data 20 time steps old must be re-evaluated
		"df": [pd.DataFrame(index = np.arange(10),columns=['Age','Key Terrain','Queries']),		# Dataframes that hold actual datastore contents
			   pd.DataFrame(index = np.arange(20),columns=['Age','Key Terrain','Queries']),
			   pd.DataFrame(index = np.arange(40),columns=['Age','Key Terrain','Queries'])]
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
		self.ds_size = env_config.get("ds_size",[10, 20, 40])
		self.num_ds = len(self.ds_size)
		self.action_space = env_config.get("action_space", spaces.Discrete(self.num_ds+1)) 	#Actions are now proceed with current retention plan or
																				# move to next step in retention plan
		# Observations #
		self.observation_space = env_config.get("observation_space",spaces.Discrete(self.N_batch))
		# Size of the data storage options
		self.ds_frac = env_config.get("ds_frac",[1, 0.5, 0.25])
		self.val_weight = env_config.get("val_weight",[np.array([1,1,1]),np.array([1,1,1]),np.array([1,1,1])])
		self.val_func = env_config.get("val_func", linear_val_func) # Note this is different from DataStore implementation
		self.ds_decay = env_config.get("ds_decay",[0.9, 0.95, 0.99])
		self.vals = env_config.get('vals',[pd.DataFrame(index = np.arange(10),columns=['Age','Key Terrain','Queries']),
											   pd.DataFrame(index = np.arange(20),columns=['Age','Key Terrain','Queries']),
											   pd.DataFrame(index = np.arange(40),columns=['Age','Key Terrain','Queries'])])
		self.df = env_config.get('df', [pd.DataFrame(index = np.arange(10),columns=self.col),
										pd.DataFrame(index = np.arange(20),columns=self.col),
										pd.DataFrame(index = np.arange(40),columns=self.col)])
		self.init_rplan = env_config.get("init_rplan",[np.mgrid[0:10, 1:4][1].astype(int),np.mgrid[0:20, 1:4][1].astype(int),np.mgrid[0:40, 1:4][1].astype(int)])
		self.ind = env_config.get("ind", [np.zeros(10).astype(int),np.zeros(20).astype(int),np.zeros(40).astype(int)])
		self.init_expir = env_config.get("init_expir",[20,20,20])
		self.ds = {}
		self.names = env_config.get("name",['deletion','Hot','Warm','Cold'])
		for i in np.arange(self.num_ds):
			self.addDataStore(self.names[i+1], i+1, self.ds_size[i], self.ds_frac[i], self.val_weight[i], self.val_func, # i+1 to skip deletion name
						  self.ds_decay[i], self.vals[i], self.init_rplan[i], self.ind[i], self.init_expir[i], self.df[i])

		# Steps/Observations #
		# A single step is trying to save/delete/etc. a single line from the batch
		self.step_num = 0

		self.deleted = pd.DataFrame(index=[], columns=self.col)
		self.del_val = []
		pass


	def addDataStore(self, name, id_num, size, frac, val_weights,val_func, decay, val, rplan, ind, expir, data):
		self.ds[name] = DataStore(id_num, size, frac, val_weights,val_func, decay, val, rplan, ind, expir, data)

	# Batch reset. Used to start trying to save a new batch of lines
	def batch_reset(self):
		self.step_num = 0
		for i in np.arange(self.num_ds):
			ds = self.ds[self.names[i+1]]
			for j in np.arange(ds.dataBatch.size):
				ds.dataBatch.batch[j].metaData.val = np.copy(ds.dataBatch.batch[j].val) #Refresh metaData of all DataStores
				ds.dataBatch.batch[j].metaData.val_tot = np.copy(ds.dataBatch.batch[j].val_tot)
		return self.step_num
	
	# An action is defined by:
	# Use current step in retention plan
	# Use next step in retention plan
	def _take_action(self, action, md): # Might be able to make this a special case of evaluate
		if action == 0: #Delete this data
			reward = -self.val_func(md.val, self.val_weight, 1) #env val_func only takes val
		else:
			md_ds_arg = md.rplan[md.ind] #which dataStore is the data currently in
			ds = self.ds[self.names[action]] # Grab DataStore associated with action

			if ds.id_num < md_ds_arg:
				print('OOOOPS')
				reward = -10 #negative reward associated with choosing a previous dataStore in retention plan
			else:
				reward = ds.val_func(md.val) #Reward for saving current metaData to dataStore
			val_arg = np.argmin(ds.dataBatch.get('val_tot',1), axis=0) #arg of the min value in dataStore

			low_val_tot = ds.dataBatch.batch[val_arg].metaData.val_tot	#val_tot of low_val

			if not np.isnan(low_val_tot): # If dataStore is full. this might not be 100% fool-proof though
				low_val = ds.dataBatch.batch[val_arg].metaData.val #low_val in dataStore
				unweighted_low_val = low_val/ds.frac #Need to remove old frac
				reward += ds.evaluate(unweighted_low_val, val_arg,self.ds,self.names)

			ds.dataBatch.batch[val_arg].metaData.val = md.val
			ds.dataBatch.batch[val_arg].metaData.val_tot = md.val_tot
			ds.dataBatch.batch[val_arg].metaData.ind = [x for x in range(len(ds.dataBatch.batch[val_arg].metaData.rplan))
														if ds.dataBatch.batch[val_arg].metaData.rplan[x] == action][0] # np.argwhere(ds.dataBatch.batch[val_arg].metaData.rplan == action)
		return reward
	
	# The step is doing the desired action and generating the RL variables
	def step(self, action, dl):
		reward = self._take_action(action, dl.metaData)
		self.step_num += 1

		obs = self.step_num
		done = self.step_num == self.observation_space.n
		return obs, reward, done, {}
	
	# Decay function for values
	# The value table will contain the current

	def inv_decay(self, val, n, rate):
		return val*rate**(-n)


	
	# The time step is where we complete the actions recommended by the agent.
	# 1. Loop through recommendations and apply to database
	# 2. Time labels are increased by 1 and the values are decayed
	# 3. Values are applied to initial for next batch
	def time_step(self, db, actions):

		for i in range(0,db.size):
			di = db.batch[i] #dataItem
			if actions[i] == 0:
				#print('BOOM', di.val_tot)
				self.del_val.append([np.nan_to_num(di.val.values[0]),di.val_tot])
			else:
				ds = self.ds[self.names[actions[i]]] # Grab DataStore associated with action
				val_arg = np.argmin(ds.dataBatch.get('val_tot'), axis=0) #arg of the min value in dataStore
				next_di = ds.dataBatch.batch[val_arg]
				low_val_tot = next_di.val_tot	#val_tot of low_val
				ds.dataBatch.save(di,val_arg,ds.val_func,actions[i]) # save new dataItem

				j_arg = [x for x in range(len(next_di.rplan)) if next_di.rplan[x] == ds.id_num][0] + 1 #actions[i]+1
				j = next_di.rplan[j_arg]
				#print(f'{self.names[actions[i]]} is full. kicking out {low_val_tot}')
				#print(f'{self.names[actions[i]]}s val_tots are {ds.dataBatch.get("val_tot")}')
				if not np.isnan(low_val_tot) and j == 0: # Cold is full! So kicked out data is deleted.
					val = ds.dataBatch.get('val')
					self.del_val.append([np.nan_to_num(next_di.val.values[0]),next_di.val_tot])
					data = pd.DataFrame(index = np.arange(2), columns = val.columns) # Empty dataframe uesed to empty row that has decayed out
					ds.dataBatch.batch[val_arg] = dataItem(data.iloc[0],data.iloc[0],np.nan,0,[1,2,3,0])
				bb = 0
				while not np.isnan(low_val_tot) and j != 0: # If dataStore is full. this might not be 100% fool-proof though.
																	# Keep moving down the line until you reach a dataStore that has space or deletion
					current_di = next_di
					next_ds = self.ds[self.names[j]]
					next_val_arg = np.argmin(next_ds.dataBatch.get('val_tot'), axis=0) #arg of the min value in dataStore
					low_val_tot = next_ds.dataBatch.batch[next_val_arg].val_tot	#val_tot of low_val
					print(f'{self.names[j]}s lowest value is {low_val_tot}, BOUNCE {bb}')
					next_di = next_ds.dataBatch.batch[next_val_arg]
					j_arg = [x for x in range(len(next_di.rplan)) if next_di.rplan[x] == next_ds.id_num][0] + 1# += 1
					j = next_di.rplan[j_arg]
					if not np.isnan(next_ds.dataBatch.batch[next_val_arg].val.values[0]) and j == 0: # Cold is full! So kicked out data is deleted.
						val = next_ds.dataBatch.get('val')
						self.del_val.append([next_ds.dataBatch.batch[next_val_arg].val.values[0],low_val_tot])
						data = pd.DataFrame(index = np.arange(2), columns = val.columns) # Empty dataframe uesed to empty row that has decayed out
						next_ds.dataBatch.batch[next_val_arg] = dataItem(data.iloc[0],data.iloc[0],np.nan,0,[1,2,3,0])
					next_ds.dataBatch.save(current_di,next_val_arg,next_ds.val_func,next_ds.id_num)
					bb+=1

		for i in np.arange(self.num_ds): #Age all dataItems in dataStores by 1 timestep
			#print('before',self.names[i+1],self.ds[self.names[i+1]].dataBatch.get('val')['Age'])
			self.ds[self.names[i+1]].dataBatch.age_step(self.ds[self.names[i+1]].val_func)
			#print('after',self.names[i+1],self.ds[self.names[i+1]].size,self.ds[self.names[i+1]].dataBatch.get('val')['Age'])
	
	def render(self, mode='human', out=0, close=True):
		time = {}
		value = {}
		for i in np.arange(self.num_ds):
			ds = self.ds[self.names[i+1]]
			print(self.names[i+1],ds.dataBatch.get('val'))
			time[self.names[i+1]] = ds.dataBatch.get('val')['Age'].values
			value[self.names[i+1]] = self.inv_decay(ds.dataBatch.get('val_tot'),time[self.names[i+1]],ds.decay)

		if(out == 0):
			sub = plt.subplot()
			clr = ['r','k','b']
			for i in np.arange(self.num_ds):
				sub.scatter(time[self.names[i+1]], value[self.names[i+1]], color=clr[i], alpha=1.0, label=self.names[i+1])

			print('deldel',self.del_val, len(self.del_val))
			x_val = [x[0] for x in self.del_val]
			y_val = [x[1] for x in self.del_val]
			sub.scatter(np.array(x_val),-np.array(y_val),color="g", label="Deleted")

			sub.set_title('Age vs Initial Value')
			sub.set_xlabel('Age')
			sub.set_ylabel('Value')
			sub.legend(loc='best')
			plt.show()
			plt.close()
		elif(out == 1):
			for i in np.arange(self.num_ds):
				print_df = self.ds[self.names[i+1]].dataBatch.get('data')
				#print_df['age'] = time[self.names[i+1]]
				#print_df['value'] = value[self.names[i+1]]
				print(f"{self.names[i+1]} Database:")
				print(print_df)

		return 0
