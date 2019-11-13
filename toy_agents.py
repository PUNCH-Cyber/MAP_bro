import gym
import numpy as np
import gym_map_bro

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

def delayed_reward_agent(env, batch_values, lr, y, num_episodes):
	Q = np.zeros([env.observation_space.n+1,env.action_space.n])
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
			epsilon = 1./(i+1.)
			rand = np.random.uniform(1.0)
			if(rand > epsilon):
				a = np.argmax(Q[s,:])
			else:
				a = np.random.choice(env.action_space.n)
			#print(s, a)
			#Get new state and reward from environment
			s1,r,d,_ = env.step(a, batch_values[s])
			#Update Q-Table with new knowledge
			Q[s,a] = Q[s,a] + lr*(r + y*np.max(Q[s1,:]) - Q[s,a])
			rAll += r
			s = s1
			if d == True:
				break
		#env.render()
		#jList.append(j)
		rList.append(rAll)
	return Q, rList

def greedy_agent(env, batch_values, lr, y, num_episodes):
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
			s1,r,d,_ = env.step(a, batch_values[s])
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
def batch_load(env, batch, values, num_episodes):
	lr = .9
	y = .95
	Q, rList = delayed_reward_agent(env, values, lr, y, num_episodes)
	#Q, rList = greedy_agent(env, values, lr, y, num_episodes)
	#print(Q)

	# Determine best actions for the batch
	batch_actions = np.argmax(Q, axis=1)
	#print(batch_actions[0:5])

	# Perform the recommended actions
	env.time_step(batch, values, batch_actions[0:5])