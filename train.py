from env import Env
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
from Agent import DQN
import matplotlib.pyplot as plt
import hyperparameter as hp
import logging
from replaybuffer import ReplayBuffer
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
    plt.xlabel('Episodes')
    plt.ylabel('reward')
    plt.plot(rewards)
    plt.pause(0.001)  # pause a bit so that plots are updated

def plot_loss(losss):
    plt.figure(1)
    plt.clf()
    plt.title('Train')
    plt.xlabel('Step')
    plt.ylabel('loss')
    plt.plot(losss)
    plt.pause(0.001)  # pause a bit so that plots are updated

def plot_cost(costs):
    plt.figure(1)
    plt.clf()
    plt.title('Train')
    plt.xlabel('Episodes')
    plt.ylabel('cost')
    plt.plot(costs)
    plt.pause(0.001)  # pause a bit so that plots are updated

if __name__ == '__main__':
    # env_name = ['CartPole-v0', 'CartPole-v1']
    # env_index = 0  # The index of the environments above
    env = Env()
    # env_evaluate = Env()  # When evaluating the policy, we need to rebuild an environment
    number = 1
    seed = 0

    state_dim = env.state_dim
    action_dim = env.action_dim
    container_num = env.containernum

    agent = DQN(state_dim, action_dim, container_num)
    RB = ReplayBuffer(hp.ReplyBuffer_size)
    # logging.basicConfig(level=logging.INFO, filename='loss.log', format='%(asctime)s - %(levelname)s - %(message)s')

    episode = 0
    globalstep = 0
    # 用来临时储存每个episode的transition
    episode_s = []
    episode_r = []
    episode_s_ = []
    episode_a = []
    episode_d = []
    episode_cost = []
    rewards = []
    costs = []
    losss = []

    while episode < hp.max_train_episode:
        s = env.reset()
        done = False
        episode += 1
        step = 0
        while not done:
            a, x = agent.choose_action(s)

            s_, r, done, cost = env.step(a, episode, step)
            # print("episode:{} \t step:{} \t action = {} -> {} \t x: {} \t reward:{}".format(episode, step, a % container_num, int(a / container_num), x, r))
            episode_s.append(s)
            episode_s_.append(s_)
            episode_a.append(a)
            episode_r.append(r)
            episode_d.append(done)
            episode_cost.append(cost)
            s = s_
            step += 1
            globalstep += 1

            if done and episode != 1:
                # if r > 0:
                for i in range(len(episode_r)):  # 有些情况没有部署完就退出，因为cost<0
                    episode_r[i] = r
                for i in range(len(episode_r)):
                    RB.push(episode_s[i], episode_a[i], episode_r[i], episode_d[i], episode_s_[i])
                # else:
                #     RB.push(episode_s[-1], episode_a[-1], episode_r[-1], episode_d[-1], episode_s_[-1])
                rewards.append(r)
                costs.append(np.mean(episode_cost))
                # print("episode:{} \t cost: {}".format(episode, costs[-1]))
                agent.clean_settled()
                episode_s = []
                episode_s_ = []
                episode_a = []
                episode_r = []
                episode_d = []
                episode_cost = []
                step = 0
            elif done:
                agent.clean_settled()
                episode_s = []
                episode_s_ = []
                episode_a = []
                episode_r = []
                episode_d = []
                episode_cost = []
                step = 0
            if globalstep > hp.ReplyBuffer_size:
                bs, ba, br, bd, bs_ = RB.sample(n=hp.batch_size)
                loss = agent.learn(bs, ba, br, bd, bs_)

                losss.append(loss)
                # logging.info('%f', loss)
            if globalstep % hp.target_update_frequency == 0:
                agent.target.load_state_dict(agent.policy.state_dict())
        if episode % hp.plot_frequency == 0 and globalstep > hp.ReplyBuffer_size:
            # plot_reward(rewards)

            plot_cost(costs)

    plot_loss(losss[:2501])

