import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.autograd import Variable

import random
from collections import deque
import time
import numpy as np

memory = deque(maxlen=10000)


def add(state, action, reward, next_state):
    global memory
    memory.append((state, action, reward, next_state))

def createMem(len):
    for i in range(len):
        state = (np.random.rand(48)*10).astype(int)
        state2 = (np.random.rand(48)*10).astype(int)
        add(state,random.randint(0,10),random.randint(0,10),state2)
        #print("memory : ")
        #print(memory)
        #time.sleep(0.1)

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
                nn.Linear(in_features = 48, out_features = 24),
                nn.BatchNorm1d(24),
                nn.Linear(in_features = 24, out_features = 12),
                nn.Dropout(0.5),
                nn.Linear(in_features = 12, out_features = 6)
            )
    def forward(self, s):
        s = self.fc(s)
        s = s.view(s.size(0),-1)
        return s

model = DQN()
t_model = DQN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
#loss_func = nn.functional.smooth_l1_loss()

def optimize(update=False):
    trans = np.array(memory)
    #trans = torch.FloatTensor(trans)
    #Trans = random.sample(trans, 32)
    #print(Trans)
    #Trans = torch.from_numpy(trans)

    trans_state = trans[:,0]
    trans_action = trans[:,1]
    trans_reward = trans[:,2]
    trans_next_state = trans[:,3]

    print("trans_state:",trans_state)
    print("trans_action:",trans_action)
    print("trans_reward:",trans_reward)
    print("trans_next_state:",trans_next_state)

    n = np.arange(10).tolist()
    print(n)
    pick = random.sample(n, k=2)

    trans_state = trans_state[pick].tolist()
    trans_action = trans_action[pick].tolist()
    trans_reward = trans_reward[pick].tolist()
    trans_next_state = trans_next_state[pick].tolist()
    

    trans_state = torch.FloatTensor(trans_state)
    trans_action = torch.FloatTensor(trans_action)
    trans_reward = torch.FloatTensor(trans_reward)
    trans_next_state = torch.FloatTensor(trans_next_state)

    # print("trans_state:",trans_state)
    # print("trans_action:",trans_action)
    # print("trans_reward:",trans_reward)
    # print("trans_next_state:",trans_next_state)

    # print("trans_state s:",trans_state.shape)
    # print("trans_action s:",trans_action.shape)
    # print("trans_reward s:",trans_reward.shape)
    # print("trans_next_state s",trans_next_state.shape)
    #print(trans_state.shape)
    #print(model(trans_state))
    state_action_values = model(trans_state).max(1)[0].detach().unsqueeze(1).type(torch.FloatTensor)
    next_state_values = t_model(trans_next_state).max(1)[0].detach()
    expected_state_action_values = (next_state_values * 0.999) + trans_reward
    expected_state_action_values = expected_state_action_values.unsqueeze(1).type(torch.FloatTensor)
    
    # print("=============================================================")
    # print(state_action_values.shape)
    # print(state_action_values)
    # print("=============================================================")
    # print(expected_state_action_values.shape)
    # print(expected_state_action_values)
    # print(type(expected_state_action_values))

    

    loss = nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values)

    loss = Variable(loss, requires_grad = True)

    optimizer.zero_grad()
    loss.backward()
    print("optimize")






if __name__ == '__main__':
    createMem(100)
    #memory = np.array(memory)
    #print(memory)
    #print("done")
    #print(memory[:,0])
    for i in range(10):
        if i % 10 == 0:
            optimize(True)
        else:
            optimize()
    

    
