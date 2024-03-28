
import math
import time
import imageio
from matplotlib import pyplot as plt
import numpy as np
import random
from environment import KeyFlatObsWrapper, RandomKeyMEnv_10, embed_mp4
import threading


class QTable:
    def __init__(self, *args):
        self.q_table = np.random.uniform(low=-0.0001, high=0, size=[*args])
        # self.q_table = np.zeros(shape=[*args])

    def __getitem__(self, state):
        '''
        This function converts the floating point state values into
        discrete values. This is often called binning.  We divide
        the range that the state values might occupy and assignaction_values
        each region to a bucket.
        '''
        return self.q_table[*state]
    def __setitem__(self, key, value):
        state, action = key
        self.q_table[*state, action] = value
    

def get_state(env):
    '''
    This function converts the floating point state values into
    discrete values. This is often called binning.  We divide
    the range that the state values might occupy and assign
    each region to a bucket.
    '''
    agent_x, agent_y = env.get_position()
    direction = env.get_direction()
    carring_key = int(env.is_carrying_key())
    key_x, key_y = env.get_k_pos()
    _, door_y = env.get_d_pos()
    is_door_open = int(env.is_door_open())
    # goal_options is always 1, so it's not used
    
    return (agent_x-1, agent_y-1, direction, carring_key, key_x-1, key_y-1, door_y-1, is_door_open)

class MiniGridSolver:
    LEARNING_RATE = 0.2
    DISCOUNT = 0.99
    EPISODES = 30000
    SHOW_EVERY = 10
    
    INITIAL_EPSILON = 1.
    
    START_DECAYING = EPISODES // 10
    DECAY_RATE = 1/(EPISODES-START_DECAYING)
    
    # DECAY_RATE = -0.0005
    # epsilon = math.exp(MiniGridSolver.DECAY_RATE * episode)
    ACTIONS_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 5, 5:5}

    
    def __init__(self, env):
        # random.seed(42)
        self.env = env
        self.env.reset()
        self.action_space = env.action_space
        
        self.is_cur_open = env.is_door_open()
        self.is_already_open = False
        self.is_grabbed_key = False
        self.height = self.env.height - 2 # without the border
        self.width = self.env.width - 2 # without the border
        self.epsilon = MiniGridSolver.INITIAL_EPSILON
        # print(MiniGridSolver.END_EPISODE_DECAYING)
        
        # self.epsilon_change = self.epsilon/(MiniGridSolver.END_EPISODE_DECAYING - MiniGridSolver.START_EPSISODE_DECAYING)
        self.num_of_actions = 5 # return 7 actions that 2 actions are not possible
        self.num_of_directions = 4
        print(f"Number of actions: {self.num_of_actions}")
        


        holding_key_options = 2
        key_x_options = 2
        key_y_options = self.height
        door_y_options = self.height
        door_open_options = 2
        self.q_table = QTable(self.width, self.height, self.num_of_directions, holding_key_options, 
                              key_x_options, key_y_options, door_y_options, door_open_options, self.num_of_actions)


    def get_step(self, state, epsilon_greedy=True):
        # Exploit - use the q-table
        if epsilon_greedy and np.random.random() < self.epsilon:
            return random.randint(0, self.num_of_actions-1)
        else:
            return np.argmax(self.q_table[state])
        
    def _run_game(self):
        done = False
        self.env.reset()
        success = False
        steps = 0
        episode_reward = 0
        state = get_state(self.env)
        

        while not done:
            steps += 1

            action = self.get_step(state)
            
            # Run simulation step
            _, orig_reward, done, truncated, _ = self.env.step(MiniGridSolver.ACTIONS_MAP[action])
            reward = self.calculate_reward(orig_reward)

            # Have we reached the goal position (have we won?)?
            if done:
                if steps % 300 == 0:
                    print("initial reward", reward)
                success = True
            # Update q-table
            new_state = get_state(self.env)
            max_future_q = np.max(self.q_table[new_state])
            current_q = self.q_table[state][action]
            new_q = (1 - MiniGridSolver.LEARNING_RATE) * current_q + MiniGridSolver.LEARNING_RATE * (reward + MiniGridSolver.DISCOUNT * max_future_q)
            self.q_table[state, action] = new_q
            episode_reward += reward
            state = new_state
            
            if truncated:
                break

        return success, steps, episode_reward
        
    
    def calculate_reward(self, reward):
        # TODO: move this to the environment
        x_pos, y_pos = self.env.get_position()
        

        if reward > 0:
            return reward*300000
        
        if x_pos > 2:
            print("pass the wall")
            return -0.01
        
        if self.env.is_door_open():
            return -1
        
        if self.env.is_carrying_key():
            return -100
        
        return -1000
            
            
        # new_openning_of_door = self.env.is_door_open() and not self.is_cur_open
        # if new_openning_of_door:
        #     self.is_cur_open = True
        #     return 0.5
        # new_closing_of_door = not self.env.is_door_open() and self.is_cur_open
        # if new_closing_of_door:
        #     self.is_cur_open = False
        #     return -0.59
        
        # if not self.is_grabbed_key and self.env.is_carrying_key():
        #     self.is_grabbed_key = True
        #     return 0.5
        # elif self.is_grabbed_key and not self.env.is_carrying_key():
        #     self.is_grabbed_key = False
        #     return -0.59
        
        # return  step_cost

    def train(self):
        success = False
        success_count = 0
        steps = 0
        episode_rewards = []
        episode_steps = []
        
        for episode in range(MiniGridSolver.EPISODES):
            self.is_already_open = False
            self.is_grabbed_key = False
            self.is_cur_open = False
            success, steps, episode_reward = self._run_game()
            print(f"Current episode: {episode+1}, Steps: {steps}, q-table sum: {np.sum(self.q_table.q_table)}, episode_rewards: {episode_reward}, min: {np.min(self.q_table.q_table)}, max: {np.max(self.q_table.q_table)}, epsilon: {self.epsilon}")
            episode_rewards.append(episode_reward)
            episode_steps.append(steps)
            # Count successes
            if success:
                success_count += 1
            # Move epsilon towards its ending value, if it still needs to move
            # if MiniGridSolver.END_EPISODE_DECAYING >= episode >= MiniGridSolver.START_EPSISODE_DECAYING:
                # self.epsilon = max(0, self.epsilon - self.epsilon_change)
            # self.epsilon = math.exp(MiniGridSolver.DECAY_RATE * episode)    
            if episode > MiniGridSolver.START_DECAYING:
                self.epsilon = max(0, self.epsilon - MiniGridSolver.DECAY_RATE)
        
                
    def create_video(self, video_filename):
        iter= 0
        done = False
        print("creating video...")
        self.env.reset()
        with imageio.get_writer(video_filename, fps=10) as video:
            while not done:
                state = get_state(self.env)
                action = game.get_step(state, False)
                # print(f'state:{state} Qtable values: {self.q_table[state]}')
                iter +=1
                _, reward, done, truncated, _ = self.env.step(MiniGridSolver.ACTIONS_MAP[action])
                if done or truncated:
                    done = True
                video.append_data(self.env.render(mode='rgb_array'))

        embed_mp4(video_filename)
    
    def create_image(self):
        screen = env.render()
        plt.imshow(screen)


env = KeyFlatObsWrapper(RandomKeyMEnv_10(render_mode='rgb_array'))
game = MiniGridSolver(env)
game.train()
game.create_video('imageio1.mp4')
game.create_video('imageio2.mp4')
game.create_video('imageio3.mp4')
