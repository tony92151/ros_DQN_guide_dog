import copy
from collections import namedtuple
from collections import deque
from itertools import count
import math
import random
import numpy as np 
import time

import gym

from wrappers import *
from memory import ReplayMemory
from models import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T



class DQN(nn.Module):
    def __init__(self, in_channels=4, n_actions=5):

        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.head = nn.Linear(512, n_actions)
        
    def forward(self, x):
        x = x.float() / 255
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.head(x)

class ReinforceAgent():
    def __init__(self):
        #self.pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
        #self.result = Float32MultiArray()

        self.load_model = False
        self.load_episode = 0
        self.state_size = 12
        self.action_size = 5
        self.episode_step = 6000
        self.target_update = 200
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.07

        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # if torch.cuda.is_available():
        #     rospy.loginfo("Use gpu")

        self.batch_size = 32
        self.train_start = 100
        self.memory = deque(maxlen=1000)

        self.model = DQN()
        self.target_model = DQN().eval()

        self.target_model.load_state_dict(self.model.state_dict())
        self.loss_func = nn.MSELoss()

        self.devices = "cpu"
        self.q_value = 0
        # if torch.cuda.is_available():
        #     self.model.cuda()
        #     self.target_model.cuda()
        #     rospy.loginfo("model in gpu")

        self.steps_done = 0
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 200
        self.GAMMA = 0.999

        self.optimizer = optim.RMSprop(self.model.parameters(),lr = 0.0002, eps=0.9, momentum=0.9)
    


    def selectAction(self, state):
        if random.random() <= self.epsilon:
            k = random.randrange(self.action_size)
            #print(k)
            return k
        else:
            state= state.tolist()
            state = torch.FloatTensor([state])
            return self.model(state).max(1)[1].view(1, 1)

    def getQvalue(self, reward, next_target, done):
        if done:
            return reward
        else:
            return reward + self.discount_factor * np.max(next_target)

    def updateTargetModel(self):
        self.target_model.set_weights(self.model.get_weights())
            

    def appendMemory(self, state, action, reward, next_state,done):
        #print(len(self.memory))
        self.memory.append((state, action, reward, next_state,done))

    # def appendMemory(self, state, action, reward, next_state, done):
    #     self.memory.append((state, action, reward, next_state, done))

    def optimize(self,target=False):
        ############################################################
        trans = np.array(self.memory).tolist()
        #print(len(self.memory))

        pick = random.sample(trans, k=self.batch_size)

        pick = np.array(pick)

        #print(pick[0][0].shape)
        d,w,h = pick[0][0].shape
        train_X = np.empty((0, d,w,h ), dtype=np.float64)
        train_Y = np.empty((0, self.action_size), dtype=np.float64)

        for i in range(self.batch_size):
            
            #print("in batch: ",i)
            trans_state = pick[i][0]
            trans_action = pick[i][1]
            trans_reward = pick[i][2]
            trans_next_state = pick[i][3]
            trans_dones = pick[i][4]

            state = torch.FloatTensor(trans_state).unsqueeze(0)
            #print(state.shape)
            with torch.no_grad():
                q_value = self.model(state)
            q_value = q_value.cpu().numpy()
            self.q_value = q_value


            next_state = torch.FloatTensor(trans_next_state).unsqueeze(0)
            #print(next_state.shape)
            trans_next_state_values = 0
            with torch.no_grad():
                if target:
                    trans_next_state_values = self.target_model(next_state)
                else:
                    trans_next_state_values = self.model(next_state)
            trans_next_state_values = np.array(trans_next_state_values)
            trans_next_q_value = self.getQvalue(trans_reward, trans_next_state_values, trans_dones)

            #print("trans_state: ",np.array([trans_state]).shape)
            train_X = np.append(train_X,np.array([trans_state]),axis = 0)
            q_va = q_value.copy()

            q_va[0][trans_action] = q_va[0][trans_action] + self.discount_factor*(trans_next_q_value-q_va[0][trans_action])

            #q_va[0][trans_action] = trans_next_q_value + q_va[0][trans_action]*self.discount_factor

            train_Y = np.append(train_Y, np.array([q_va[0]]),axis = 0)

        #print(train_X.shape)
        #print(train_Y.shape)
        train_X = torch.FloatTensor(train_X)
        train_Y = torch.FloatTensor(train_Y)

        output = self.model(train_X)
        loss = self.loss_func(output,train_Y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        #print("GG")


def train(env, nepos):
    for i in range (nepos):
        #print
        obs = env.reset()
        for t in count():
            display(obs)

            obs, reward, done, info = env.step(action)
            if done:
                break
def display(obs):
    state = np.array(obs)
    state = state.transpose((2, 0, 1))
    state = torch.from_numpy(state)
    return state


if __name__ == '__main__':

    nepos=100

    agent = ReinforceAgent()

    env = gym.make("PongNoFrameskip-v4")
    env = make_env(env)
    
    for i in range (nepos):
        print("EP",i)
        obs = env.reset()

        global_step = 0
        for t in count():
            state = display(obs)

            env.render()
            time.sleep(1/60)

            action = agent.selectAction(state)
            obs, reward, done, info = env.step(action)

            

            if len(agent.memory) >= agent.train_start:
                if global_step <= agent.target_update:
                    agent.optimize()
                else:
                    agent.optimize(True)

            print("R:",reward,"  MEM:",len(agent.memory))
            global_step += 1
            #next_state = 0
            next_state = display(obs)
            
            agent.appendMemory(np.array(state),action,reward,np.array(next_state),done)

            if global_step % agent.target_update == 0:
                agent.target_model.load_state_dict(agent.model.state_dict())
                print("UPDATE TARGET NETWORK")

            if done:
                break


            if agent.epsilon > agent.epsilon_min:
                agent.epsilon *= agent.epsilon_decay
    env.close()