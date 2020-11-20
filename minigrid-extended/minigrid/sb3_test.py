import gym
import os
import time
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from datetime import datetime
from stable_baselines import PPO2
from env_wrapper import FlatObsWrapper
from window import Window

# tf.get_logger().setLevel('INFO')
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def policy_rollout(env, model, num_episodes):

    for i in range(num_episodes):
        print('===================== Starting episode {} ========================='.format(i))
        window = Window(env.spec.id)
        observation = env.reset()
        done = False

        episode_reward = 0
        episode_length = 0
        images = []

        while not done:
            action = model.predict(observation)

            observation, reward, done, info = env.step(action[0])

            episode_reward += reward
            episode_length += 1

            img = env.unwrapped.render('rgb_array')

            images.append(img)
            window.set_caption(
                'Trained Policy | Timesteps:{} Reward:{} Total_reward:{}'.format(episode_length, reward,
                                                                                 episode_reward))
            time.sleep(0.1)

            window.show_img(img)

        print('Total reward: {:.5f} '.format(episode_reward))
        print('Total length: ', episode_length)
        window.close()


if __name__ == '__main__':

    train_policy = True

    env = gym.make('MiniGrid-Empty-5x5-v0')

    new_model_path = os.path.join(os.getcwd(), 'log/') + str(
        datetime.now().strftime('%H%M%S')) + '_' + env.spec.id
    exist_model_path = os.path.join(os.getcwd(), 'log/') + '145939' + '_' + env.spec.id

    wrapped_env = FlatObsWrapper(env)

    obs = wrapped_env.reset()
    model = PPO2(MlpPolicy, wrapped_env, verbose=0)

    if train_policy:
        model.learn(total_timesteps=2000)
        model.save(new_model_path)
    else:
        trained_model = model.load(exist_model_path)
        policy_rollout(wrapped_env, trained_model, 3)

    env.close()
