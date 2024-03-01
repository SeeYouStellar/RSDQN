import torch
from model import Qnet
import numpy as np
from replaybuffer import ReplayBuffer
import hyperparameter as hp
import torch.nn.functional as F

class DQN(object):
    def __init__(self, state_dim, action_dim, container_num):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = hp.lr  # learning rate
        self.GAMMA = hp.GAMMA  # discount factor

        self.policy = Qnet(state_dim, action_dim)
        self.target = Qnet(state_dim, action_dim)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.container_num = container_num

        self.settled = []
        for i in range(container_num):
            self.settled.append(0)
        self.replay_buffer = ReplayBuffer(hp.ReplyBuffer_size)

    def choose_action(self, s, deterministic):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)  # 神经网络的输入一定要有batch_size这个维度，所以增加一个维度
        qvals = self.policy(s).detach().numpy().flatten()
        x = np.random.randint(0,10) / 10
        if x > hp.epsilon:  # We use the deterministic policy during the evaluating
            a = np.argmax(qvals)  # Select the action with the highest probability
            self.settled[a % self.container_num] = 1
            return a
        else:  # We use the stochastic policy during the training
            a = np.random.randint(0, 40)
            while self.settled[a % self.container_num] == 1:
                a = np.random.randint(0, 40)
            self.settled[a % self.container_num] = 1
            return a

    def argmax_action(self, prob_weights):
        prob_weights = np.array(prob_weights)[::-1]
        prob_weights.sort()
        '''选择最大概率的还未部署的动作'''
        for action, _ in enumerate(prob_weights):
            if self.settled[action % self.container_num] == 0:
                return action

    def learn(self, bs, ba, br, bd, bs_):

        qvals = self.policy(bs).gather(1, ba.unsqueeze(1)).squeeze()  # [batch_size, 40]
        # 使用target Q网络计算next_s_batch对应的值。
        next_qvals = self.target(bs_).detach().max(dim=1)[0]  # [batch_size, 1]
        # 使用MSE计算loss。
        # one = torch.ones_like(bs)
        loss = F.mse_loss(br + self.GAMMA * next_qvals * (1 - bd), qvals)

        return loss.detach().numpy()

    def clean_settled(self):
        for i in range(self.container_num):
            self.settled[i] = 0