
import time
import gym
import numpy as np
import random
import gym_minigrid
from gym_minigrid.wrappers import *
import matplotlib.pyplot as plt
from numpy import load
from numpy import save
import gym
import numpy as np
import imageio

from environment import KeyFlatObsWrapper, RandomEmptyEnv_10, RandomKeyMEnv_10, embed_mp4


# This function converts the floating point state values into
# discrete values. This is often called binning.  We divide
# the range that the state values might occupy and assign
# each region to a bucket.
def calc_state_idx(env, goal_options=[(8,1), (8,8), (1,8)]):
    agent_x, agent_y = env.get_position()
    direction = env.get_direction()
    
    goal_loc = env.get_goal_pos()
    reduced_goal_loc = goal_options.index(goal_loc)
    
    return agent_x-1, agent_y-1, direction, reduced_goal_loc


def run_game(q_table, render, should_update):
    done = False
    state, _ = env.reset()
    success = False
    steps = 0

    while not done:
        steps += 1
        state_idx = calc_state_idx(env)
        
        # Exploit - use the q-table
        if np.random.random() > epsilon:
            action = np.argmax(q_table[state_idx])
            if steps % 300 == 0:
                print("in exploit", action)
        else:
            # Explore - take a random action
            action = np.random.randint(0, env.action_space.n)
            if steps % 300 == 0:
                print("in explore")

        # Run simulation step
        new_state, reward, done, _, _ = env.step(action)

        # Have we reached the goal position (have we won?)?
        if done: # TODO: check if done is True when the agent reaches the goal
            if steps % 300 == 0:
                print("initial reward", reward)
            success = True

        # Update q-table
        if should_update:
            new_state_idx = calc_state_idx(env)

            max_future_q = np.max(q_table[new_state_idx])
            current_q = q_table[state_idx][action]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[state_idx][action] = new_q

        state = new_state

        # if render:
        #     env.render()

    return success, steps

# (agentlc, direction, exitloc)
# (23, 2, 45) ---> line?

'''
env.get_goal_pos()
env.get_direction()


position - (1, 6)

1*8+6 = 14
6*8+1 = 49

goal - (8,8)



'''

env = KeyFlatObsWrapper(RandomEmptyEnv_10 (render_mode='rgb_array'))
obs = env.reset()

screen = env.render()
plt.imshow(screen)

print('Agent Direction:', env.get_direction())
print('Agent Position:', env.get_position())
print('Goal position: ', env.get_goal_pos())
print(f"Number of actions: {env.action_space.n}")

height = env.height - 2 # without the border
width = env.width - 2 # without the border
print('Map Height:', env.height-2)
print('Map Width:', env.width-2)

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 5000
SHOW_EVERY = 10

epsilon = 1
START_EPSILON_DECAYING = 0.5
END_EPSILON_DECAYING = EPISODES // 10
print(END_EPSILON_DECAYING)
epsilon_change = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)
num_of_actions = env.action_space.n

num_of_directions = 4
goal_options = 3

# agent_x, agent_y, agent_dir, goal_x, goal_y
q_table = np.random.uniform(low=-3, high=0, size=[width, height, num_of_directions, goal_options, num_of_actions])
print("num_of_actions =", num_of_actions)
print("Q table shape = ",q_table.shape)


success = False
episode = 0
success_count = 0
steps = 0
while episode < EPISODES:
    episode += 1
    done = False

    if episode % SHOW_EVERY == 0:
        print(f"Current episode: {episode}, Steps: {steps}, success: {success_count} ({float(success_count)/SHOW_EVERY})")
        success, steps = run_game(q_table, True, False)
        
        success_count = 0
    else:
        run_game(q_table, False, True)
        success, steps = run_game(q_table, False, True)

    # Count successes
    if success:
        success_count += 1

    # Move epsilon towards its ending value, if it still needs to move
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon = max(0, epsilon - epsilon_change)

print(success)





# create a video of the trained agent


name = "MountainCar-v0"
stepCounter = 0

env = gym.make(name)
env.reset()
done = False
video_filename = 'imageio.mp4'
discrete_state = calc_discrete_state(env.reset())
success = False
iter= 0

with imageio.get_writer(video_filename, fps=10) as video:
  while  not done:
    action = np.argmax(q_table[discrete_state])
    iter +=1
    new_state, reward, done, info = env.step(action)
    new_state_disc = calc_discrete_state(new_state)
    if new_state[0] >= env.unwrapped.goal_position:
       success = True
    discrete_state = new_state_disc
    stepCounter = stepCounter +1
    #print("iteration=",iter, " observation=",observation, " reward-", reward," done=",done," info=",info)
    video.append_data(env.render(mode='rgb_array'))

embed_mp4(video_filename)






'''
#Random action plus visualization
# env = KeyFlatObsWrapper(RandomEmptyEnv_10 (render_mode='rgb_array'))
obs = env.reset()
video_filename = './fn.mp4'
max_steps = 100

with imageio.get_writer(video_filename, fps=10) as video:
  obs = env.reset()
  done = False
  total_reward = 0
  for step in range(max_steps):
      action = random.randint(0, num_of_actions - 1)
      obs, reward, done, _, _ = env.step(action)
      next_obs = obs  # Get agent's position directly from the environment
      video.append_data(env.render())
      if done:
        print("done","reward=", total_reward,"num_of_steps=",step)
        break
embed_mp4(video_filename)



#=============== DO NOT DELETE ===============
# random.seed(42)
env = KeyFlatObsWrapper(RandomKeyMEnv_10 (render_mode='rgb_array'))
#env = KeyFlatObsWrapper(KeyMazeEnv_10(render_mode='rgb_array'))

obs = env.reset()
# ============================================
screen = env.render()
plt.imshow(screen)

action_space = env.action_space
num_of_actions = action_space.n
print(f"Number of actions: {num_of_actions}")

env = KeyFlatObsWrapper(RandomKeyMEnv_10 (render_mode='rgb_array'))
obs = env.reset()
# ============================================
screen = env.render()
plt.imshow(screen)
print('Door is opened? : ', env.is_door_open())
print('carrying Key? : ',env.is_carrying_key())
print('Agent Direction:', env.get_direction())
print('Agent Position:', env.get_position())
print('Is there a wall in from of the Agent? : ', env.is_wall_front_pos())
print('Goal position: ', env.get_goal_pos())
print('Key Position:', env.get_k_pos())
print('Door Position:', env.get_d_pos())
'''