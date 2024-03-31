import imageio
import numpy as np
from environment import KeyFlatObsWrapper, RandomEmptyEnv_10, embed_mp4

class QTable:
    def __init__(self, *args):
        self.q_table = np.random.uniform(low=-0.0001, high=0, size=[*args])

    def __getitem__(self, state):
        '''
        This function converts the floating point state values into
        discrete values. This is often called binning.  We divide
        the range that the state values might occupy and assignaction_values
        each region to a bucket.
        '''
        return self.q_table[tuple(state)]
    def __setitem__(self, key, value):
        state, action = key
        self.q_table[tuple(state) + (action,)] = value

def get_state(env, goal_options=[(1,8),(8,8), (8,1)]):
    '''
    This function converts the floating point state values into
    discrete values. This is often called binning.  We divide
    the range that the state values might occupy and assign
    each region to a bucket.
    '''
    agent_x, agent_y = env.get_position()
    direction = env.get_direction()
    goal_loc = env.get_goal_pos()
    reduced_goal_loc = goal_options.index(goal_loc)
    
    return (agent_x-1, agent_y-1, direction, reduced_goal_loc)

class MiniGridSolver:
    LEARNING_RATE = 0.1
    DISCOUNT = 0.95
    EPISODES = 2000
    SHOW_EVERY = 10
    START_EPSILON_DECAYING = 0.2
    END_EPSILON_DECAYING = EPISODES // 10
    def __init__(self, env):
        # np.random.seed(42)
        self.env = env
        self.env.reset()
        self.action_space = env.action_space
        self.num_of_actions = self.action_space.n
        print(f"Number of actions: {self.num_of_actions}")
        self.height = self.env.height - 2 # without the border
        self.width = self.env.width - 2 # without the border
        self.epsilon = 1
        print(MiniGridSolver.END_EPSILON_DECAYING)
        self.epsilon_change = self.epsilon/(MiniGridSolver.END_EPSILON_DECAYING - MiniGridSolver.START_EPSILON_DECAYING)
        self.num_of_actions = self.env.action_space.n
        self.num_of_directions = 4
        goal_options = 3
        self.q_table = QTable(self.width, self.height, self.num_of_directions, goal_options, self.num_of_actions)


    def _run_game(self):
        done = False
        self.env.reset()
        success = False
        steps = 0
        episode_reward = 0
        state = get_state(self.env)

        while not done:
            steps += 1
            # Exploit - use the q-table
            if np.random.random() >= self.epsilon:
                action = np.argmax(self.q_table[state])
                if steps % 300 == 0:
                    print("in exploit", action)
            else:
                # Explore - take a random action
                action = np.random.randint(0, self.num_of_actions)
                if steps % 300 == 0:
                    print("in explore")

            # Run simulation step
            _, orig_reward, done, truncated, _ = self.env.step(action)
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
        if reward > 0:
            return reward
        step_cost = -0.01
        return  step_cost
    
    def train(self):
        success = False
        success_count = 0
        steps = 0
        episode_rewards = []
        episode_steps = []
        for episode in range(MiniGridSolver.EPISODES):
            success, steps, episode_reward = self._run_game()
            print(f"Current episode: {episode+1}, Steps: {steps}, q-table sum: {np.sum(self.q_table.q_table)}, episode_rewards: {episode_reward}, min: {np.min(self.q_table.q_table)}")
            episode_rewards.append(episode_reward)
            episode_steps.append(steps)
            # Count successes
            if success:
                success_count += 1
            # Move epsilon towards its ending value, if it still needs to move
            if MiniGridSolver.END_EPSILON_DECAYING >= episode >= MiniGridSolver.START_EPSILON_DECAYING:
                self.epsilon = max(0, self.epsilon - self.epsilon_change)
                
    def create_video(self, video_filename):
        iter = 0
        done = False
        print("creating video...")
        self.env.reset()
        with imageio.get_writer(video_filename, fps=10) as video:
            while not done:
                state = get_state(self.env)
                action = np.argmax(self.q_table[state])
                iter += 1
                _, reward, done, truncated, _ = self.env.step(action)
                if done or truncated:
                    done = True
                video.append_data(self.env.render(mode='rgb_array'))

        embed_mp4(video_filename)


env = KeyFlatObsWrapper(RandomEmptyEnv_10 (render_mode='rgb_array'))
game = MiniGridSolver(env)
game.train()
game.create_video('imageio.mp4')
