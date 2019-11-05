import gym
import numpy as np
import gym_map

env = gym.make('map-hd-v0')

def recommend(files, Q):
    action = ['save', 'compress', 'delete']
    for i in range(0, Q.shape[0]-1):
        row = Q[i]
        if(np.max(row) < 0.1):
            print("File {}: {} recommendation: {}". format(i, files[i], action[2]))
        else:
            r = np.argmax(row)
            print("File {}: {} recommendation: {}". format(i, files[i], action[r]))

def delayed_reward_agent(env, lr, y, num_episodes):
	Q = np.zeros([env.observation_space.n,env.action_space.n])
	# Set learning parameters
	#create lists to contain total rewards and steps per episode
	#jList = []
	rList = []
	for i in range(num_episodes):
		#Reset environment and get first new observation
		s = env.reset()
		rAll = 0
		d = False
		j = 0
		#The Q-Table learning algorithm
		while j < 99:
			j+=1
			#Choose an action by greedily (with noise) picking from Q table
			a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
			#print(s, a)
			#Get new state and reward from environment
			s1,r,d,_ = env.step(a)
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

def greedy_agent(env, lr, y, num_episodes):
	Q = np.zeros([env.observation_space.n,env.action_space.n])
	# Set learning parameters
	#create lists to contain total rewards and steps per episode
	#jList = []
	rList = []
	for i in range(num_episodes):
		#Reset environment and get first new observation
		s = env.reset()
		rAll = 0
		d = False
		j = 0
		#The Q-Table learning algorithm
		while j < 99:
			j+=1
			#Choose an action by greedily (with noise) picking from Q table
			a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
			#print(s, a)
			#Get new state and reward from environment
			s1,r,d,_ = env.step(a)
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