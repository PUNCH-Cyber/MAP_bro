# Tensorflow Q-Network implementation
import gym
import gym_map
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt

env = gym.make('map-v0')

# Example 7
# Solving example 6 with a neural network instead of a Q table
# Two high reward files, two large files, and two normal files

# Set up the problem
hd_size = 4.1
file_array = np.array([ [1.0, 2.0], [1.0, 2.0],
						[2.0, 1.0], [2.0, 1.0],
						[1.0, 1.0], [1.0, 1.0]],np.float)
file_num = 6
compress_ratio = 0.5
def reward_function(self, size, reward, action):
	if(action == 0):
		return reward
	elif(action == 1):
		return reward*compress_ratio
	else:
		return 0

env.__myinit__(hd_size, file_num, file_array, compress_ratio, reward_function)

# Define the neural network
tf.reset_default_graph()

#These lines establish the feed-forward part of the network used to choose actions
inputs1 = tf.placeholder(shape=[1,7],dtype=tf.float32)
W = tf.Variable(tf.random_uniform([7,3],0,0.01))
Qout = tf.matmul(inputs1,W)
predict = tf.argmax(Qout,1)

#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
nextQ = tf.placeholder(shape=[1,3],dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
updateModel = trainer.minimize(loss)

init = tf.initialize_all_variables()

# Set learning parameters
y = .99
e = 0.2
num_episodes = 2000
#create lists to contain total rewards and steps per episode
jList = []
rList = []
with tf.Session() as sess:
	sess.run(init)
	for i in range(num_episodes):
		#Reset environment and get first new observation
		s = env.reset()
		rAll = 0
		d = False
		j = 0
		a_list = np.zeros((6))
		#The Q-Network
		while j < 99:
			j+=1
			#Choose an action by greedily (with e chance of random action) from the Q-network
			a,allQ = sess.run([predict,Qout],feed_dict={inputs1:np.identity(7)[s:s+1]})
			if np.random.rand(1) < e:
				a[0] = env.action_space.sample()
			a_list[j-1] = a[0]
			#Get new state and reward from environment
			s1,r,d,_ = env.step(a[0])
			if(r <= 0.01):
				a_list[j-1] = 2
			#Obtain the Q' values by feeding the new state through our network
			Q1 = sess.run(Qout,feed_dict={inputs1:np.identity(7)[s1:s1+1]})
			#Obtain maxQ' and set our target value for chosen action.
			maxQ1 = np.max(Q1)
			targetQ = allQ
			targetQ[0,a[0]] = r + y*maxQ1
			#Train our network using target and predicted Q values
			_,W1 = sess.run([updateModel,W],feed_dict={inputs1:np.identity(7)[s:s+1],nextQ:targetQ})
			rAll += r
			s = s1
			if d == True:
				#Reduce chance of random action as we train the model.
				e = 1./((i/50) + 10)
				break
		if(i == 1):
			actions = np.array(["Save", "Compress", "Delete"])
			print("New high score! Total reward = {}".format(rAll))
			print(actions[np.int_(a_list)])
		if(i>1):
			if(rAll > np.max(rList)):
				actions = np.array(["Save", "Compress", "Delete"])
				print("New high score! Total reward = {}".format(rAll))
				print(actions[np.int_(a_list)])
		jList.append(j)
		rList.append(rAll)
#print("Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%")

plt.plot(rList)
plt.show()
plt.close()