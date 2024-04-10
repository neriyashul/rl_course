from __future__ import annotations
import gym
from matplotlib.animation import FuncAnimation
import numpy as np
from collections import deque, namedtuple
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (16, 10)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
torch.manual_seed(0)

import base64, io

# For visualization
from gym.wrappers.monitoring import video_recorder
from IPython.display import HTML
from IPython import display
import glob

# %%

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# TODO: delete this block:


if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print (x)
else:
    print ("MPS device not found.")



# ## Imports

# %%
import numpy as np
import gym
from gym import logger as gymlogger
from gym.utils import seeding
from gym import error, spaces, utils
gymlogger.set_level(40) # error only
import glob
import io
import base64
import os
import random
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
import math
import glob
from pyvirtualdisplay import Display
from IPython.display import HTML
from IPython import display as ipythondisplay
import pygame
import pyvirtualdisplay
import imageio
import IPython
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque


# ## Display utils
# The cell below contains the video display configuration. No need to make changes here.

# %%
def embed_mp4(filename):
  """Embeds an mp4 file in the notebook."""
  video = open(filename,'rb').read()
  b64 = base64.b64encode(video)
  tag = '''
  <video width="640" height="480" controls>
    <source src="data:video/mp4;base64,{0}" type="video/mp4">
  Your browser does not support the video tag.
  </video>'''.format(b64.decode())

  return IPython.display.HTML(tag)
# display = pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()


# #Env1 - Empty Env


# ## Class Env

from gym_minigrid.minigrid import COLOR_NAMES
from gym_minigrid.minigrid import Grid
from gym_minigrid.minigrid import MissionSpace
from gym_minigrid.minigrid import Door, Goal, Key, Wall, Lava, Floor
from minigrid_x import MiniGridEnv
from gym import spaces
import random

class RandomEmptyEnv_10(MiniGridEnv):
    def __init__(
        self,
        size=10, # DEFINE THE WIDTH AND HEIGHT
        agent_start_pos=(1, 1),
        agent_start_dir = 0,
        max_steps: int | None = None,
        **kwargs,
    ):

        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        agent_start_x_loc =random.randint(1, 6)
        agent_start_y_loc =random.randint(1, 6)

        self.agent_start_pos = (agent_start_x_loc,agent_start_y_loc)

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )
        self.action_space = spaces.Discrete(3)
        self.walls_init = []

        # change 21-09 for random env
        self.not_goal_loc = [(col,row) for col in range(self.unwrapped.width) for row in range(self.unwrapped.height) if row == 0 or row == (self.unwrapped.width-1) or col == 0 or col == (self.unwrapped.height-1)]
        self.not_goal_loc.append(self.agent_start_pos)
        self.goal_pos = (self.grid.width - 2, self.grid.height - 2)

    def reset(self, **kwargs):
        # Randomize start position each time environment is reset
        agent_start_x_loc = random.randint(1, 6)
        agent_start_y_loc = random.randint(1, 6)
        self.agent_start_pos = (agent_start_x_loc, agent_start_y_loc)

        self.values = [0, 1, 2, 3]
        self.agent_start_dir = random.choice(self.values)

        # Recalculate not_goal_loc as it depends on the agent's start position
        self.not_goal_loc = [(col, row) for col in range(self.unwrapped.width) for row in range(self.unwrapped.height)
                             if row == 0 or row == (self.unwrapped.width - 1) or col == 0 or col == (self.unwrapped.height - 1)]
        self.not_goal_loc.append(self.agent_start_pos)

        # Call the reset method of the parent class
        return super().reset(**kwargs)

    @staticmethod
    def _gen_mission():
        return "grand mission"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Generate verical separation wall
        # for i in range(0, height):
        #     self.grid.set(5, i, Wall())


        for column, row in self.walls_init:
          self.grid.set(column, row, Wall())

        self.key_pos = (6, 5)
        self.door_pos = (6, 7)

        self.goal_pos = random.choice([(8,1), (8,8), (1,8)])

        self.put_obj(Goal(), self.goal_pos[0], self.goal_pos[1])

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "grand mission"




# ## WRAPPER

# %%
class EMPTYRGBImgObsWrapper(gym.core.ObservationWrapper):
    """
    Wrapper to use fully observable RGB image as the only observation output,
    no language/mission. This can be used to have the agent to solve the
    gridworld in pixel space.
    """

    def __init__(self, env, tile_size=32, plot=False, preprocess= lambda x: x):
        super().__init__(env)
        self.tile_size = tile_size
        self.plot = plot
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            # #the default is (320,320,3). Tile size  = 32 X 10 (grid size)
            # TODO: BE CAREFULL TO CHANGE the shape size to be according your preprocess size/channels
            # The env information i staken from here
            shape=(1, 144, 144),
            dtype='uint8'
        )
        self.prev_door = False
        self.prev_key = False
        self.action_space = spaces.Discrete(self.action_space.n)

        #TODO: THINK AND TEST DIFFERENT PREPROCESS
        self._preprocess = preprocess
        
    

    def observation(self, obs):
      env = self.unwrapped

      # Call render without any unsupported keyword arguments
      rgb_img = env.render()  # Use the default rendering behavior
      return rgb_img

    def reset(self, seed = 0, options = None):
        self.prev_door = False
        self.prev_key = False
        obs, info = super().reset()
        if self.plot:
            plt.show()
        return self._preprocess(obs), info

    def step(self, action):
        #action = ACTION_MAP[action]
        obs, r, d, info, x = super().step(action)
        obs = self._preprocess(obs)

        # TODO: REWARD SHAPING
        #we encourage you to come up with a better reward function using  self.is_door_open() and self.is_carrying_key()
        if r <= 0:
            r = -0.001 # penalty for each step
        else:
            r *= 0.5

        return obs, r, d, info, x


# ## Random action

# %%
# env = EMPTYRGBImgObsWrapper(RandomEmptyEnv_10 (render_mode='rgb_array'))
# obs = env.reset()
# video_filename = './dl/vid.mp4'
# max_steps = 100
# # Evaluation
# os.makedirs(os.path.dirname(video_filename), exist_ok=True)
# with imageio.get_writer(video_filename, fps=10) as video:
#   obs = env.reset()
#   done = False
#   total_reward = 0
#   for step in range(max_steps):
#       action = env.action_space.sample()
#       obs, reward, done, _, _ = env.step(action)
#       next_obs = obs  # Get agent's position directly from the environment
#       video.append_data(env.render())
#       if done:
#         print("done","reward=", total_reward,"num_of_steps=",step)
#         break
# embed_mp4(video_filename)




# ### Define Policy
# Unlike value-based method, the output of policy-based method is the probability of each action. It can be represented as policy. So activation function of output layer will be softmax, not ReLU.

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# ### Network

class PolicyNetwork(nn.Module):
    def __init__(self, state_shape, action_size, seed):
        super(PolicyNetwork, self).__init__()
        # self.seed = torch.manual_seed(seed)
        print(state_shape)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # self.conv = nn.Sequential(
        #     nn.Conv2d(1, 16, kernel_size=8, stride=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(16, 32, kernel_size=3, stride=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        # )
        print(f'input shape: {state_shape}')
        dummy_input = torch.zeros(1, *state_shape)
        conv_out_size = self.conv(dummy_input).view(1, -1).size(1)
        

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_size)
        )
        
        # self.fc = nn.Sequential(
        #     nn.Linear(conv_out_size, 128),
        #     nn.ReLU(),
        #     nn.Dropout(),
        #     nn.Linear(128, action_size),
        # )

    def forward(self, state):
        conv_out = self.conv(state).view(state.size()[0], -1)
        return F.softmax(self.fc(conv_out), dim=-1)

    def run(self, state):
        # Automatically use the device of the model's parameters
        device = next(self.parameters()).device
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        
        global temp_it
        temp_it += 1
        if temp_it % 100 == 0:
            print('action:', action.item(), 'probs:', probs)
        
        return action.item(), m.log_prob(action)

temp_it = 0

# %%

# class ReplayBuffer:

#   def __init__(self, action_size, buffer_size, batch_size, seed):
#     self.action_size = action_size
#     self.memory = deque(maxlen=buffer_size)
#     self.batch_size = batch_size
#     self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
#     self.seed = random.seed(seed)

#   def add(self, state, action, reward, next_state, done):
#     e = self.experience(state, action, reward, next_state, done)
#     self.memory.append(e)

#   def sample(self):
#     experiences = random.sample(self.memory, k=self.batch_size)

#     states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
#     actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
#     rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
#     next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
#     dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

#     return (states, actions, rewards, next_states, dones)

#   def __len__(self):
#     return len(self.memory)




# ### Agent

#Agent

# class Agent():
#   def __init__(self, state_size, action_size, seed):
#     """Initialize an Agent object.

#     Params
#     ======
#         state_size (int): dimension of each state
#         action_size (int): dimension of each action
#         seed (int): random seed
#     """
#     self.state_size = state_size
#     self.action_size = action_size
#     self.seed = random.seed(seed)
#     # Q-Network
#     self.qnetwork_local = PolicyNetwork(state_size, action_size, seed).to(device)
#     self.qnetwork_target = PolicyNetwork(state_size, action_size, seed).to(device)

#     self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LEARNING_RATE)

#     # Replay memory
#     self.memory = ReplayBuffer(action_size, REPLAY_BUFFER_SIZE, BATCH_SIZE, seed)

#     self.t_step = 0

#   def step(self, state, action, reward, next_state, done):
#     """Save experience in replay memory, and use random sample from buffer to learn."""


#     # Save experience in replay memory
#     self.memory.add(state, action, reward, next_state, done)

#     # Learn every UPDATE_EVERY time steps.
#     self.t_step = (self.t_step + 1) % UPDATE_EVERY

#     if self.t_step == 0:
#       # If enough samples are available in memory, get random subset and learn
#       if len(self.memory) > BATCH_SIZE:
#         experiences = self.memory.sample()
#         self.learn(experiences, GAMMA)

#   def act(self, state, eps=0.):
#     """Returns actions for given state as per current policy.

#     Params
#     ======
#         state (array_like): current state

#         eps (float): epsilon, for epsilon-greedy action selection
#     """


#     state = torch.from_numpy(state).float().unsqueeze(0).to(device)
#     self.qnetwork_local.eval()
#     with torch.no_grad():
#       action_values = self.qnetwork_local(state)

#     self.qnetwork_local.train()

#     # e-greedy explore/exploit
#     if random.random() > eps:
#       return np.argmax(action_values.cpu().data.numpy())

#     else:
#       return random.choice(np.arange(self.action_size))


#   def learn(self, experiences, gamma):
#     """Update value parameters using given batch of experience tuples.

#     Params
#     ======
#         experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
#         gamma (float): discount factor
#     """
#     states, actions, rewards, next_states, dones = experiences
#     # Get max predicted Q values (for next states) from target model

#     q_targets_next = self.qnetwork_target(next_states.unsqueeze(dim=1)).detach().max(1)[0].unsqueeze(1)

#     # Compute Q targets for current states
#     q_targets = rewards + (gamma * q_targets_next * (1 - dones))

#     # Get expected Q values from local model
#     q_expected = self.qnetwork_local(states.unsqueeze(dim=1)).gather(1, actions)

#     loss = F.mse_loss(q_expected, q_targets)
#     # loss = nn.HuberLoss(delta=1.0)

#     # Minimize the loss
#     self.optimizer.zero_grad()
#     # negative loss to make the gradient ascent
#     (-loss).backward() 
#     self.optimizer.step()

#     # ------------------- update target network ------------------- #
#     self.soft_update(self.qnetwork_local, self.qnetwork_target, SOFT_UPDATE_RATE)

#   def soft_update(self, local_model, target_model, SOFT_UPDATE_RATE):
#     for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
#       target_param.data.copy_(SOFT_UPDATE_RATE*local_param.data + (1.0-SOFT_UPDATE_RATE)*target_param.data)

def train(policy, optimizer, env, n_episodes=1000, max_t=500, gamma=0.9, print_every=1):
    scores_deque = deque(maxlen=100)
    scores = []
    
    
    for i_episode in range(1, n_episodes+1):
        saved_log_probs = []
        rewards = []
        state, _ = env.reset()
        for t in range(max_t):
            action, log_prob = policy.run(state)
            saved_log_probs.append(log_prob)
            state, reward, done, truncated, _ = env.step(action)
            rewards.append(reward)
            if done:
                break
            
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        discounts = [gamma**i for i in range(len(rewards)+1)]
        R = sum([a*b for a,b in zip(discounts, rewards)])

        policy_loss = []
        for log_prob in saved_log_probs:
            policy_loss.append(-log_prob * R)
        policy_loss = torch.cat(policy_loss).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        
        # TODO: delete this
        # global plot_list
        # plot_list = scores_deque
        arr = list(scores_deque)[-75:]
        plt.plot(arr, color='blue')
        plt.pause(0.05)

        if i_episode % print_every == 0:
            print(f'Episode {i_episode}\tScore: {arr[-1]:.2f}\tAverage Score: {np.mean(scores_deque):.2f}')
        if np.mean(scores_deque) >= 985.0:
            print(f'Environment solved in {i_episode-100:d} episodes!\tAverage Score: {np.mean(scores_deque):.2f}')
            break
    return scores




# %%
def preprocess(obs, resize_shape=(144, 144)):
        '''
        preprocess state
        '''
        
        # Convert RGB to grayscale
        # print(f'obs.shape {obs.shape}')
        cropped_obs = obs[32:-31, 32:-31]
        # print(f'cropped obs.shape {cropped_obs.shape}')
        obs_gray = cv2.cvtColor(cropped_obs, cv2.COLOR_RGB2GRAY)
        # print(f'obs_gray.shape {obs_gray.shape}')
        # Resize image


        # increase the contrast of red and blue pixels
        red_mask = np.all(cropped_obs == [0, 0, 255], axis=-1)  # Check for red pixels (BGR format)
        blue_mask = np.all(cropped_obs == [255, 0, 0], axis=-1)  # Check for blue pixels (BGR format)
        obs_gray[red_mask] = 100  # Replace red pixels with 0
        obs_gray[blue_mask] = 255  # Replace blue pixels with 0

        obs_resized = cv2.resize(obs_gray, resize_shape)


        # print(f'obs_resized.shape {obs_resi}')
        # Normalize pixel values to [0, 1]
        # obs_resized = obs_resized / 255.0
        # print(f'm cool {obs_resized.shape}')
        # Add channel dimension to make it (H, W, 1)
        # obs_resized = np.expand_dims(obs_resized, axis=-1)
        obs_resized = np.expand_dims(obs_resized, axis=0)
        # print(obs_resized.shape)
        return obs_resized



# ### Run

# %%
# import gym




# %%
env = EMPTYRGBImgObsWrapper(RandomEmptyEnv_10 (render_mode='rgb_array'), preprocess=preprocess)
state, _ = env.reset()
print(state.shape)
screen = env.render()
# plt.imshow(screen)


state_shape = state.shape  # Replace with your actual state size
action_space = env.action_space
num_actions = action_space.n
print(f"Number of actions: {num_actions}")
print(f"State size is: {state_shape}")
# plt.imshow(obs.squeeze(0))





# env = EMPTYRGBImgObsWrapper(RandomEmptyEnv_10 (render_mode='rgb_array'))
# obs = env.reset()
# video_filename = './dl/vid.mp4'
# max_steps = 100
# # Evaluation
# os.makedirs(os.path.dirname(video_filename), exist_ok=True)
# with imageio.get_writer(video_filename, fps=10) as video:
#   obs = env.reset()
#   done = False
#   total_reward = 0
#   for step in range(max_steps):
#       action = env.action_space.sample()
#       obs, reward, done, _, _ = env.step(action)
#       next_obs = obs  # Get agent's position directly from the environment
#       video.append_data(env.render())
#       if done:
#         print("done","reward=", total_reward,"num_of_steps=",step)
#         break
# embed_mp4(video_filename)


# env = EMPTYRGBImgObsWrapper(RandomEmptyEnv_10 (render_mode='rgb_array'))


#BUFFER SIZE VERY IMPORTANT TO BE LARGE FOR CONVERGENCE
REPLAY_BUFFER_SIZE =    1000    #@param {type:"number"}
BATCH_SIZE         =    64       # minibatch size
GAMMA              =    0.9     # discount factor
SOFT_UPDATE_RATE   =    1e-3     # for soft update of target parameters
LEARNING_RATE      =    5e-4     # learning rate
UPDATE_EVERY       =    5        #@param {type:"slider", min:5, max:50}





# Environment details
state_shape = env.observation_space.shape
action_size = env.action_space.n

# Device configuration

# TODO: delete it:
mps_device = torch.device("mps")
device = mps_device # TODO: delete it:

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# Initialize policy
policy = PolicyNetwork(state_shape=state_shape, action_size=action_size, seed=42).to(device)

# Optimizer
optimizer = optim.Adam(policy.parameters(), lr=0.00001)

print('start training...')
# Run REINFORCE
scores = train(policy, optimizer, env)

print('finish training!')
print('scores:', scores)






# Initialize environment (Replace 'YourEnvHere' with your environment's id)
# env = gym.make('CartPole-v0')

# # Environment details
# state_size = env.observation_space.shape[0]
# action_size = env.action_space.n

# # Device configuration
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
# # Initialize policy
# policy = Policy2(state_size=state_size, action_size=action_size, hidden_size=128).to(device)

# # Optimizer
# optimizer = optim.Adam(policy.parameters(), lr=1e-2)

# # Run REINFORCE
# scores = reinforce(policy, optimizer, env, n_episodes=10000, print_every=100, device=device)


# %%
torch.save(policy.state_dict(), './policy_network.pth')

print(f'model saved in policy_network.pth')

# %%
policy = PolicyNetwork().to(device)
policy.load_state_dict(torch.load('/content/policy_network.pth', map_location=device))
policy.eval()


# ### Plot the learning progress

# %%
# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()


# ### Animate it with Video

# %%
# TODO
# 
# def show_video(env_name):
#     mp4list = glob.glob('content/*.mp4')
#     if len(mp4list) > 0:
#         mp4 = 'video/{}.mp4'.format(env_name)
#         video = io.open(mp4, 'r+b').read()
#         encoded = base64.b64encode(video)
#         display.display(HTML(data='''<video alt="test" autoplay
#                 loop controls style="height: 400px;">
#                 <source src="data:video/mp4;base64,{0}" type="content/mp4" />
#              </video>'''.format(encoded.decode('ascii'))))
#     else:
#         print("Could not find video")

# def show_video_of_model(policy, env_name):
#     env = gym.make(env_name)
#     vid = video_recorder.VideoRecorder(env, path="/content/{}.mp4".format(env_name))
#     state = env.reset()
#     done = False
#     for t in range(1000):
#         vid.capture_frame()
#         action, _ = policy.act(state)
#         next_state, reward, done, _ = env.step(action)
#         state = next_state
#         if done:
#             break
#     vid.close()
#     env.close()


