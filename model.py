import torch.nn as nn
import torch.nn.functional as F

class Qnet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Qnet, self).__init__()
        self.l1 = nn.Linear(state_dim, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, action_dim)

    def forward(self, s):
        s = F.relu(self.l1(s))
        s = F.relu(self.l2(s))

        out = self.l3(s)
        # out_prob = F.softmax(self.l3(s))
        return out
