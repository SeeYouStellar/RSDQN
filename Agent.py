import torch
from model import Qnet
import numpy as np
from replaybuffer import ReplayBuffer
import hyperparameter as hp
import torch.nn.functional as F
import random

class DQN(object):
    def __init__(self, state_dim, action_dim, container_num):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = hp.epsilon

        self.policy = Qnet(state_dim, action_dim)
        self.target = Qnet(state_dim, action_dim)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=hp.lr)
        self.container_num = container_num

        self.settled = []
        for i in range(container_num):
            self.settled.append(0)

    def choose_action(self, s):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)  # 神经网络的输入一定要有batch_size这个维度，所以增加一个维度
        qvals = self.policy(s).detach().numpy().flatten()
        x = random.random()
        if x < self.epsilon:  # We use the deterministic policy during the evaluating
            a = self.argmax_action(qvals)  # Select the action with the highest probability
            self.settled[a % self.container_num] = 1
        else:  # We use the stochastic policy during the training
            a = np.random.randint(0, 40)
            while self.settled[a % self.container_num] == 1:
                a = np.random.randint(0, 40)
            self.settled[a % self.container_num] = 1
        self.epsilon = max(0.99, self.epsilon + hp.epsilon_decrement)
        return a, x

    def argmax_action(self, qvals):
        tulpe_qvals = [(index, value) for index, value in enumerate(qvals)]
        sorted_tulpe_qvals = sorted(tulpe_qvals, key=lambda x: x[1])
        sorted_tulpe_qvals = np.array(sorted_tulpe_qvals)[::-1]
        '''选择最大概率的还未部署的动作'''
        for tuple_qval in sorted_tulpe_qvals:
            action = int(tuple_qval[0])
            if self.settled[action % self.container_num] == 0:
                return action

    def learn(self, bs, ba, br, bd, bs_):

        qvals = self.policy(bs).gather(1, ba.unsqueeze(1)).squeeze()  # [batch_size, 40]
        # 使用target Q网络计算next_s_batch对应的值。
        next_qvals = self.target(bs_).detach().max(dim=1)[0]  # [batch_size, 1]
        # 使用MSE计算loss。
        # one = torch.ones_like(bs)
        loss = F.mse_loss(br + hp.GAMMA * next_qvals * (1 - bd), qvals)

        return loss.detach().numpy()

    def clean_settled(self):
        for i in range(self.container_num):
            self.settled[i] = 0