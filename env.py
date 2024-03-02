from dataSet.data import Data
import numpy as np
import hyperparameter as hp
import logging

alpha = 0.5  # reward weighting factor
beta = [0.5, 0.5]  # 资源亲和度

data = Data()
NodeNumber = data.NodeNumber
ContainerNumber = data.ContainerNumber
ServiceNumber = data.ServiceNumber
ResourceType = data.ResourceType
service_containernum = data.service_containernum  # 每个服务所需容器数列表

service_container = data.service_container  # 每个服务所启动的容器id列表
service_container_relationship = data.service_container_relationship  # 容器和微服务的映射列表
container_state1 = data.container_state_queue[:]  # 容器状态（部署位置,cpu,mem）队列

class Env():
    def __init__(self):
        # State
        self.State = []
        self.node_state = []
        self.container_state = []

        self.state_dim = 0
        self.action_dim = NodeNumber*ContainerNumber
        self.containernum = ContainerNumber
        self.nodenum = NodeNumber
        self.servicenum = ServiceNumber

        self.lambda1 = hp.lambda1
        self.lambda2 = hp.lambda2
        self.lambda3 = hp.lambda3
        self.lambda4 = hp.lambda4

        self.MinCost = 1e3
        self.REWARD = -1e3

        self.allCost = []
        for i in range(ContainerNumber):
            self.allCost.append([])

        self.prepare()
        logging.basicConfig(level=logging.INFO, filename='cost.log', format='%(asctime)s - %(levelname)s - %(message)s')

    def prepare(self):
        self.container_state = container_state1[:]
        for i in range(NodeNumber):
            '''[[1, 0, ..., 1, cpu, mem], [], []], 前面containernum个元素代表该container是否部署在该节点上，最后两个属性是cpu和mem的资源利用率'''
            for j in range(ContainerNumber + 2):
                self.node_state.append(0)
        # print(self.container_state)
        # print(self.node_state)
        self.State = self.container_state + self.node_state
        # print(self.State)
        self.action = [-1, -1]  # 当前step采取的action,每一个action只会最多将一个容器部署到节点上
        # Communication weight between microservices
        self.service_weight = data.service_weight
        # Communication distance between nodes
        self.Dist = data.Dist
        self.state_dim = len(self.State)

    def reset(self):
        self.node_state = []
        self.container_state = []
        self.prepare()
        return self.State

    def containerDis(self, i, j):
        # to calculate the distance between container i and j
        m = -1
        n = -1
        # 每个微服务部署的节点
        m = self.container_state[i * 3]
        n = self.container_state[j * 3]
        # 每个容器对应的微服务
        p = service_container_relationship[i]
        q = service_container_relationship[j]

        # 判断部署后，这两个容器是否在同一个节点上，并且这两个容器不能是同一个微服务的不同副本（不同副本间不会有通信开销，而会负载均衡）
        if self.Dist[m-1][n-1] != 0 and (p != q):
            container_dist = self.Dist[m-1][n-1]
        else:
            container_dist = 0
        return container_dist

    def serviceComCost(self, i, j):  # 某两个服务间的通信开销
        cost = 0
        interaction = self.service_weight[i][j] / (service_containernum[i] * service_containernum[j])
        # print('CalcuCost({}, {})'.format(i, j))
        # print('service_weight = {}'.format(self.service_weight[i][j]))
        # print('it = {}'.format(interaction))
        # paper eq(3) 计算每两个微服务之间的所有容器的comCost
        for k in range(len(service_container[i])):
            for l in range(len(service_container[j])):
                # paper eq(2) 计算某两个容器的comCost
                cost += self.containerDis(service_container[i][k], service_container[j][l]) * interaction
                # print('container {} container {} , cost = {}'.format(service_container[i][k], service_container[j][l], cost))
        return cost

    def ComCost(self):
        # paper eq(3) 计算整个应用中所有微服务之间的comCost
        Cost = 0
        for i in range(ServiceNumber):
            for j in range(ServiceNumber):
                Cost += self.serviceComCost(i, j)
        # paper eq(4)
        return 0.5 * Cost

    def usageVar(self):
        NodeCPU = []
        NodeMemory = []
        Var = 0
        for i in range(NodeNumber):
            '''[[1, 0, ..., 1, cpu, mem], [], []], 前面containernum个元素代表该container是否部署在该节点上，最后两个属性是cpu和mem资源占用率'''
            U = self.node_state[i * (ContainerNumber + 2) + ContainerNumber]
            M = self.node_state[i * (ContainerNumber + 2) + (ContainerNumber + 1)]
            NodeCPU.append(U)
            NodeMemory.append(M)
            if NodeCPU[i] > 1 or NodeMemory[i] > 1:
                # 0<资源利用率<1
                Var = -10
        # Variance of node load
        # paper eq(7)
        Var += beta[0] * np.var(NodeCPU) + beta[1] * np.var(NodeMemory)
        return Var

    def cost(self):
        # todo g1和g2的归一化
        re = 0
        g1 = self.ComCost()
        g1 = g1 / 371.5
        g2 = self.usageVar()
        # if g2 < 0:
        #     g2 = -100
        g2 = g2 / 6.002500000000001

        logging.info('com:%f  use:%f', g1, g2)
        re += alpha * g1 + (1 - alpha) * g2   
        return re*100, g1, g2

    def reward(self, cost, done, episode):
        if not done:
            return 0
        elif episode == 1:
            return self.lambda1
        else:
            if cost < 0:
                return self.lambda2
            elif cost > 0 and cost < self.MinCost:
                return self.REWARD+self.lambda3
            elif cost == self.MinCost:
                return self.REWARD
            else:
                return self.lambda4*(self.MinCost-cost)

    def state_update(self, container_state, node_state):
        self.State = container_state + node_state

    def update(self):
        # 根据 action 来 更新 state

        if self.action[0] >= 0 and self.action[1] >= 0:
            # self.action[1] 容器
            # self.action[0] 节点
            # update container state
            self.container_state[self.action[1] * 3] = self.action[0]  # 某个容器部署在某个节点上
            # update node state
            self.node_state[(self.action[0]) * (ContainerNumber + 2) + self.action[1]] = 1  # 代表容器该部署在该节点上
            # 把容器的资源需求加到节点的资源占用上
            self.node_state[(self.action[0]) * (ContainerNumber + 2) + ContainerNumber] += self.container_state[self.action[1] * 3 + 1]
            self.node_state[(self.action[0]) * (ContainerNumber + 2) + (ContainerNumber + 1)] += self.container_state[self.action[1] * 3 + 2]
        else:
            # print("invalid action, action = [{}, {}]".format(self.action[0], self.action[1]))
            self.node_state = []
            self.container_state = []
            self.prepare()

        self.state_update(self.container_state, self.node_state)

    def step(self, action, episode, step):

        self.action = self.index_to_act(action)  # action = containernum*nodenum+x  [节点，容器]
        self.update()
        done = 0
        cost, comm, var = self.cost()

        if self.allCost[step]:
            self.MinCost = min(self.allCost[step])

        # 第一个episode索性不要了
        if episode == 1 and step == self.containernum - 1:
            done = 1
            return self.State, 0, done, cost
        elif episode == 1:
            return self.State, 0, done, cost

        r = 0
        if cost > 0:
            if step == self.containernum - 1:
                done = 1
                if abs(self.MinCost - cost) < 1e-7:
                    r = self.REWARD
                elif (self.MinCost - cost) > 0:
                    self.REWARD = self.REWARD + self.lambda3
                    r = self.REWARD
                else:
                    r = self.lambda4 * (self.MinCost - cost)
                self.allCost[step].append(cost)
        else:
            done = 1
            r = self.lambda2

        return self.State, r, done, cost

    def index_to_act(self, index):
        act = [-1, -1]
        act[0] = int(index / ContainerNumber)  # 节点
        act[1] = index % ContainerNumber  # 容器
        return act


# env = Env()
# env.step(0)
# env.step(1)
# env.step(2)
# env.step(3)
# env.step(4)
# env.step(5)
# env.step(6)
# env.step(7)
