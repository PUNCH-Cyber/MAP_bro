import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

class broEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def _default_r_func(self, size, reward, action):
        if(action == 0 or action == 1):
            return reward
        else:
            return 0
    
    # HD0Env is a toy "hard drive" for use with a model reinforcement learning agent
    # The hard drive is initialized with a size and a 2d array for files
    # The files  array is a list of file sizes and rewards for storing each file
    def __init__(self):
        # Actions #
        # 0 = Save
        # 1 = Compress
        # 2 = Delete
        self.action_space = spaces.Discrete(3)
        # Define the files as pairs of size, reward for saving
        files = np.array([[1.0, 1.0], [1.0,1.0], [1.0,1.0]], np.float)
        self.files = files
        # Observations #
        # N observations for N files, advance for each file saved
        N = 3
        self.N = N
        self.observation_space = spaces.Discrete(N+1)
        # Size of the hard drive
        self.size = 3.1
        # Set the compression fraction for compressing
        self.frac = 0.5
        # Set the reward function
        self.r_func = self._default_r_func
        pass
    
    def __myinit__(self, size, N, files, redu, r_func=_default_r_func):
        # Actions #
        # 0 = Save
        # 1 = Compress
        # 2 = Delete
        self.action_space = spaces.Discrete(3)

        # Define the files as pairs of size, reward for saving
        self.files = files

        # Observations #
        # N observations for N files, advance for each file saved
        self.N = N
        self.observation_space = spaces.Discrete(N+1)

        # Size of the hard drive
        self.size = size
        # Set the compression fraction for compressing
        self.frac = redu
        
        # Define the reward function
        self.r_func = r_func
        pass
        
    # Reset number of files stored to 0 and the amount of data stored to 0
    def reset(self):
        self.step_num = 0
        self.num_stored = 0
        self.size_stored = 0.0
        return self.step_num

    def _take_action(self, action):
        # Check how big the file will be on the drive
        file_size0 = self.files[self.step_num][0]
        reward0 = self.files[self.step_num][1]

        file_size = 0.0
        if(action == 0):
            file_size = file_size0
        elif(action == 1):
            file_size = file_size0*self.frac
        
        # Try to store the file
        if(action == 0 or action == 1):
            combined_size = self.size_stored + file_size
            if(combined_size > self.size): # Delete the file because the hard drive is full
                reward = self.r_func(self, file_size0, reward0, 2)
            else:
                self.num_stored += 1
                self.size_stored += file_size
                reward = self.r_func(self, file_size0, reward0, action)
        else:   # Do nothing
            reward = self.r_func(self, file_size0, reward0, action)

        # Return the reward
        return reward
    
    def step(self, action):
        reward = self._take_action(action)
        self.step_num += 1

        obs = self.step_num
        done = (self.step_num == self.N)
        return obs, reward, done, {}
    
    def render(self, mode='human', close=False):
        print("Stored {} of {} files".format(self.num_stored, self.step_num))
        print("Using {} of {} total storage".format(self.size_stored, self.size))
        return 0

class HDEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def _default_r_func(self, size, reward, action):
        if(action == 0 or action == 1):
            return reward
        else:
            return 0
    
    # HD0Env is a toy "hard drive" for use with a model reinforcement learning agent
    # The hard drive is initialized with a size and a 2d array for files
    # The files  array is a list of file sizes and rewards for storing each file
    def __init__(self):
        # Actions #
        # 0 = Save
        # 1 = Compress
        # 2 = Delete
        self.action_space = spaces.Discrete(3)
        # Define the files as pairs of size, reward for saving
        files = np.array([[1.0, 1.0], [1.0,1.0], [1.0,1.0]], np.float)
        self.files = files
        # Observations #
        # N observations for N files, advance for each file saved
        N = 3
        self.N = N
        self.observation_space = spaces.Discrete(N+1)
        # Size of the hard drive
        self.size = 3.1
        # Set the compression fraction for compressing
        self.frac = 0.5
        # Set the reward function
        self.r_func = self._default_r_func
        pass
    
    def __myinit__(self, size, N, files, redu, r_func=_default_r_func):
        # Actions #
        # 0 = Save
        # 1 = Compress
        # 2 = Delete
        self.action_space = spaces.Discrete(3)

        # Define the files as pairs of size, reward for saving
        self.files = files

        # Observations #
        # N observations for N files, advance for each file saved
        self.N = N
        self.observation_space = spaces.Discrete(N+1)

        # Size of the hard drive
        self.size = size
        # Set the compression fraction for compressing
        self.frac = redu
        
        # Define the reward function
        self.r_func = r_func
        pass
        
    # Reset number of files stored to 0 and the amount of data stored to 0
    def reset(self):
        self.step_num = 0
        self.num_stored = 0
        self.size_stored = 0.0
        return self.step_num

    def _take_action(self, action):
        # Check how big the file will be on the drive
        file_size0 = self.files[self.step_num][0]
        reward0 = self.files[self.step_num][1]

        file_size = 0.0
        if(action == 0):
            file_size = file_size0
        elif(action == 1):
            file_size = file_size0*self.frac
        
        # Try to store the file
        if(action == 0 or action == 1):
            combined_size = self.size_stored + file_size
            if(combined_size > self.size): # Delete the file because the hard drive is full
                reward = self.r_func(self, file_size0, reward0, 2)
            else:
                self.num_stored += 1
                self.size_stored += file_size
                reward = self.r_func(self, file_size0, reward0, action)
        else:   # Do nothing
            reward = self.r_func(self, file_size0, reward0, action)

        # Return the reward
        return reward
    
    def step(self, action):
        reward = self._take_action(action)
        self.step_num += 1

        obs = self.step_num
        done = (self.step_num == self.N)
        return obs, reward, done, {}
    
    def render(self, mode='human', close=False):
        print("Stored {} of {} files".format(self.num_stored, self.step_num))
        print("Using {} of {} total storage".format(self.size_stored, self.size))
        return 0