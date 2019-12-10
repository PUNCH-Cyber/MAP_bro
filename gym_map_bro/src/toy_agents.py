import gym
import numpy as np
from gym import spaces
import copy

env = gym.make('map-bro-v0')

def recommend(files, Q):
    action = ['save', 'compress', 'delete']
    for i in range(0, Q.shape[0]-1):
        row = Q[i]
        if(np.max(row) < 0.1):
            print("File {}: {} recommendation: {}". format(i, files[i], action[2]))
        else:
            r = np.argmax(row)
            print("File {}: {} recommendation: {}". format(i, files[i], action[r]))

def delayed_reward_agent(env, db, lr, y, num_episodes):

	env.observation_space = spaces.Discrete(len(db.batch))
	Q = np.zeros([env.observation_space.n+1,env.action_space.n])#np.zeros([env.observation_space.n+1,env.action_space.n])
	# Set learning parameters
	#create lists to contain total rewards and steps per episode
	rList = []
	for i in range(num_episodes):
		# Soft Reset environment to preserve value table
		s = env.batch_reset()
		rAll = 0
		d = False
		j = 0
		#The Q-Table learning algorithm
		while j < 99:
			j+=1
			#Choose an action by greedily (with noise) picking from Q table
			epsilon = 1./np.log10(i+1.)
			rand = np.random.uniform(5.0)
			if(rand > epsilon):
				a = np.argmax(Q[s,:])
			else:
				print('RANDOM',i)
				a = np.random.choice(env.action_space.n)
			#Get new state and reward from environment
			s1,r,d,_ = env.step(a, db.batch[s])
			#Update Q-Table with new knowledge
			Q[s,a] = Q[s,a] + lr*(r + y*np.max(Q[s1,:]) - Q[s,a])
			rAll += r
			s = s1
			if d == True:
				break
		rList.append(rAll)
	return Q, rList

def greedy_agent(env, db, lr, y, num_episodes):
	Q = np.zeros([env.observation_space.n,env.action_space.n])
	# Set learning parameters
	#create lists to contain total rewards and steps per episode
	#jList = []
	rList = []
	for i in range(num_episodes):
		#Reset environment and get first new observation
		s = env.batch_reset()
		rAll = 0
		d = False
		j = 0
		#The Q-Table learning algorithm
		while j < 99:
			j+=1
			#Choose an action by greedily (with noise) picking from Q table
			epsilon = 1./(i+1.)
			rand = np.random.uniform(1.0)
			if(rand > epsilon):
				a = np.argmax(Q[s,:])
			else:
				a = np.random.choice(env.action_space.n)
			#print(s, a)
			#Get new state and reward from environment
			s1,r,d,_ = env.step(a, db[s])
			#Update Q-Table with new knowledge
			Q[s,a] = Q[s,a] + lr*r
			rAll += r
			s = s1
			if d == True:
				break
		#env.render()
		#jList.append(j)
		rList.append(rAll)
	return Q, rList


# Train the agent on a new batch of values, print the final results
def batch_load(env, db, num_episodes):
	#Need to check if any of the data has expired
	lr = .9
	y = .95
	for i in env.names[1:]:
		expir_dis = env.ds[i].get_expir()
		db.add(expir_dis)

	#Training on incoming data
	Q, rList = delayed_reward_agent(env, db, lr, y, num_episodes)
	#Q, rList = greedy_agent(env, values, lr, y, num_episodes)
	print(f'Q is {Q}')

	# Determine best actions for the batch
	batch_actions = np.argmax(Q, axis=1)

	# Perform the recommended actions
	env.time_step(db, batch_actions[0:5])

def batch_load_static(env, db, num_episodes):
	#Need to check if any of the data has expired
	lr = .9
	y = .95
	for i in np.arange(env.num_ds)[::-1]+1: #Go from cold to hot DataStores
		expir_dis = env.ds[env.names[i]].get_expir() #Get expired dataItems
		for j in np.arange(len(expir_dis)):
			next_id = expir_dis[j].rplan[expir_dis[j].ind+1] #Get next step in rplan
			if next_id != 0:
				next_ds = env.ds[env.names[next_id]] # Grab DataStore associated with action
				val_arg = np.argmin(next_ds.dataBatch.get('val_tot'), axis=0) #arg of the min value in dataStore
				next_di = next_ds.dataBatch.batch[val_arg]
				low_val_tot = next_di.val_tot
				next_ds.dataBatch.save(expir_dis[j],val_arg,next_ds.val_func,next_id)
				next_id = next_di.rplan[next_di.ind+1]

				while not np.isnan(low_val_tot) and next_id != 0: # Continue cascade until you reach empty dataStore or deletion
					next_ds = env.ds[env.names[next_id]]
					next_val_arg = np.argmin(next_ds.dataBatch.get('val_tot'), axis=0) #arg of the min value in dataStore
					low_val_tot = next_ds.dataBatch.batch[next_val_arg].val_tot	#val_tot of low_val
					tmp = copy.deepcopy(next_ds.dataBatch.batch[next_val_arg])
					next_ds.dataBatch.save(next_di,next_val_arg,next_ds.val_func,next_id)
					next_di = tmp
					next_id = next_di.rplan[next_di.ind+1]
					if next_id == 0:
						env.del_val.append([next_ds.dataBatch.batch[next_val_arg].val.values[0],low_val_tot])
			else:
				env.del_val.append([expir_dis[j].val.values[0],expir_dis[j].val_tot])


	#Training on incoming data
	Q, rList = delayed_reward_agent(env, db, lr, y, num_episodes)

	# Determine best actions for the batch
	batch_actions = np.argmax(Q, axis=1)

	# Perform the recommended actions
	env.time_step(db, batch_actions[0:5])
