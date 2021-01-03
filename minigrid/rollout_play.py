import torch
import os
import imageio
import time

def updated_policy_rollout(policy, env, window, iter):
    observation = env.reset()

    done = False
    episode_reward = 0
    episode_length = 0

    while not done:
        observation = torch.tensor(observation, dtype=torch.float32)
        action = policy.act(observation)[0].data.cpu().numpy()

        observation, reward, done, info = env.step(action)

        episode_reward += reward
        episode_length += 1

        img = env.unwrapped.render('rgb_array')
        window.set_caption(
            '{}th Update | Timesteps:{} Reward:{:.4f} Total_reward:{:.4f}'.format(iter, episode_length, reward,
                                                                                  episode_reward))

        window.show_img(img)

    print('Total reward: {:.5f} '.format(episode_reward))
    print('Total length: ', episode_length)

    window.close()


def loaded_policy_rollout(env, policy, window, model_path):
    observation = env.reset()

    done = False
    episode_reward = 0
    episode_length = 0

    images = []

    print('Loading trained model')
    ckpt = torch.load(model_path)
    policy.actor.load_state_dict(ckpt)

    while not done:
        observation = torch.tensor(observation, dtype=torch.float32)
        action = policy.act(observation)[0].data.cpu().numpy()

        observation, reward, done, info = env.step(action)

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

    imageio.mimsave(os.path.join(os.path.dirname(model_path), 'trained_result.gif'), images)