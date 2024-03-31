


from matplotlib import pyplot as plt

from environment import KeyFlatObsWrapper, RandomKeyMEnv_10
from q1_b import MiniGridKeySolver


for discount in [0.85, 0.90, 0.95, 0.97]:
    env = KeyFlatObsWrapper(RandomKeyMEnv_10(render_mode='rgb_array'))
    game = MiniGridKeySolver(env, 0.2, discount, 30000)
    game.train()

    averages = [sum(game.episode_rewards[i:i + 30]) / 30 for i in range(0, len(game.episode_rewards), 30)]
    game.create_video(f'imageio2_with_discount_{discount}_factor.mp4')

    plt.plot(range(len(averages)), averages)
    plt.xlabel('Episode (average of 30 episodes)')
    plt.ylabel('Rewards')
    plt.title(f'Episode Rewards Progress, discount={discount}')

    # Save the plot image with a unique nam
    plt.savefig(f'plot_q1b_rewards_discount_{discount}_factor.png')  # Save each plot with a unique name

    # Show the plot without blocking
    plt.show(block=False)


