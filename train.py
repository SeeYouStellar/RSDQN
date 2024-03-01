from env import Env
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
from Agent import DQN
import matplotlib.pyplot as plt
import hyperparameter as hp
# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

evaluate_rewards = []  # Record the rewards during the evaluating

# def evaluate_policy(env, agent):
#     times = 3  # Perform three evaluations and calculate the average
#     evaluate_reward = 0
#     for _ in range(times):
#
#         s = env.reset()
#         done = False
#         episode_reward = 0
#         while not done:
#             a = agent.choose_action(s, deterministic=False)
#             s_, r, done, _, _ = env.step(a, MinCost, REWARD, episode)
#             episode_reward += r
#             s = s_
#         evaluate_reward += episode_reward
#
#     return int(evaluate_reward / times)

def plot_reward(rewards):
    plt.figure(1)
    plt.clf()
    plt.title('Train')
    plt.xlabel('Episodes(*50)')
    plt.ylabel('reward')
    plt.plot(rewards)
    plt.pause(0.001)  # pause a bit so that plots are updated

if __name__ == '__main__':
    # env_name = ['CartPole-v0', 'CartPole-v1']
    # env_index = 0  # The index of the environments above
    env = Env()
    # env_evaluate = Env()  # When evaluating the policy, we need to rebuild an environment
    number = 1
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)

    state_dim = env.state_dim
    action_dim = env.action_dim
    container_num = env.containernum

    agent = DQN(state_dim, action_dim, container_num)

    max_train_episode = 1e5  # Maximum number of training steps
    evaluate_freq = 1e2  # Evaluate the policy every 'evaluate_freq' steps
    evaluate_num = 0  # Record the number of evaluations


    print("state_dim={}".format(state_dim))
    print("action_dim={}".format(action_dim))
    print('max_train_episode={}'.format(max_train_episode))
    print('evaluate_freq={}'.format(evaluate_freq))

    episode = 0
    step = 0
    # 用来临时储存每个episode的transition
    episode_s = []
    episode_r = []
    episode_s_ = []
    episode_a = []
    episode_d = []
    rewards = []

    while episode < max_train_episode:
        s = env.reset()
        done = False
        episode += 1
        while not done:
            a = agent.choose_action(s, deterministic=False)
            s_, r, done = env.step(a, episode)
            print("episode:{} \t step:{} \t action = {} -> {} \t reward:{}".format(episode, step, a % container_num, int(a / container_num), r))
            episode_s.append(s)
            episode_s_.append(s_)
            episode_a.append(a)
            episode_r.append(r)
            episode_d.append(done)
            s = s_
            step += 1
            if done:
                for i in range(len(episode_r)):
                    episode_r[i] = r
                for i in range(len(episode_r)):
                    agent.replay_buffer.push(episode_s[i], episode_a[i], episode_r[i], episode_d[i], episode_s_[i])
                rewards.append(r)
                agent.clean_settled()
            if step > hp.ReplyBuffer_size:
                bs, ba, br, bd, bs_ = agent.replay_buffer.sample(n=hp.batch_size)
                loss = agent.learn(bs, ba, br, bd, bs_)
            if step % hp.target_update_frequency == 0:
                agent.target.load_state_dict(agent.policy.state_dict())
        if episode % hp.plot_frequency == 0:
            plot_reward(rewards)



