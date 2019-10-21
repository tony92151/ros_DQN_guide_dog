#!/home/tedbest/torch_gpu_ros/bin/python

import rospy
import rospkg
import os
import json
import numpy as np
import random
import time
import sys
import math
#https://github.com/dokelung/Python-QA/blob/master/questions/standard_lib/Python%20%E7%8D%B2%E5%8F%96%E6%96%87%E4%BB%B6%E8%B7%AF%E5%BE%91%E5%8F%8A%E6%96%87%E4%BB%B6%E7%9B%AE%E9%8C%84(__file__%20%E7%9A%84%E4%BD%BF%E7%94%A8%E6%96%B9%E6%B3%95).md

rospack = rospkg.RosPack()
env_path = rospack.get_path('machine_learning')

sys.path.append(env_path+'/env')
from environment_dog1 import Env



from collections import deque
from std_msgs.msg import Float32MultiArray

from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.autograd import Variable


EPISODES = 10000


class DQN(nn.Module):
    def __init__(self,state_size,action_size):
        super(DQN, self).__init__()
        
        self.ln1 = nn.Linear(in_features = state_size, out_features = 64)
        self.ba = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.ln2 = nn.Linear(in_features = 64, out_features = 48)
        self.dr = nn.Dropout(0.5)
        self.ln3 = nn.Linear(in_features = 48, out_features = 12)
        self.dr2 = nn.Dropout(0.3)
        self.ln4 = nn.Linear(in_features = 12, out_features = action_size)
            
    def forward(self, s):
        s = self.ln1(s)
        #s = self.ba(s)
        s = self.relu1(s)
        s = self.ln2(s)
        s = self.dr(s)
        s = self.ln3(s)
        s = self.dr2(s)
        s = self.ln4(s)
        s = s.view(s.size(0),-1)
        return s



class ReinforceAgent():
    def __init__(self, state_size, action_size):
        #self.pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
        self.result = Float32MultiArray()

        self.load_model = False
        self.load_episode = 0
        self.state_size = state_size
        self.action_size = action_size
        self.episode_step = 6000
        self.target_update = 1000
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.05

        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # if torch.cuda.is_available():
        #     rospy.loginfo("Use gpu")

        self.batch_size = 128
        self.train_start = 256
        self.memory = deque(maxlen=10000)

        self.model = DQN(self.state_size,self.action_size)
        self.target_model = DQN(self.state_size,self.action_size).eval()

        self.devices = "cpu"
        # if torch.cuda.is_available():
        #     self.model.cuda()
        #     self.target_model.cuda()
        #     rospy.loginfo("model in gpu")

        self.steps_done = 0
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 200
        self.GAMMA = 0.999

        self.optimizer = optim.RMSprop(self.model.parameters())
    


    def selectAction(self, state):
        p = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)
        if random.random() <= p:
            k = random.randrange(7)
            #print(k)
            return k
        else:
            #print(state)
            #state = np.array(state)
            #print(state)
            state= state.tolist()
            #print(state)
            state = torch.FloatTensor([state])
            #print(state.shape)
            #print(state)
            return self.model(state).max(1)[1].view(1, 1)

    def getQvalue(self, reward, next_target, done):
        if done:
            return reward
        else:
            return reward + self.discount_factor * np.amax(next_target)

    def updateTargetModel(self):
        self.target_model.set_weights(self.model.get_weights())
            

    def appendMemory(self, state, action, reward, next_state,done):
        #print(len(self.memory))
        self.memory.append((state, action, reward, next_state,done))

    # def appendMemory(self, state, action, reward, next_state, done):
    #     self.memory.append((state, action, reward, next_state, done))

    def optimize(self,target=False):
        ############################################################
        trans = np.array(self.memory)
        #print(len(self.memory))

        pick = random.sample(trans, k=self.batch_size)

        #print(pick.shape)

        train_X = np.empty((0, self.state_size), dtype=np.float64)
        train_Y = np.empty((0, self.action_size), dtype=np.float64)

        for i in range(self.batch_size):
            
            #print("in batch: ",i)
            trans_state = pick[i][0]
            trans_action = pick[i][1]
            trans_reward = pick[i][2]
            trans_next_state = pick[i][3]
            trans_dones = pick[i][4]

            state = torch.FloatTensor(trans_state.reshape(1, len(trans_state)))
            with torch.no_grad():
                q_value = self.model(state)
            q_value = q_value.cpu().numpy()
            self.q_value = q_value


            next_state = torch.FloatTensor(trans_next_state.reshape(1, len(trans_next_state)))
            trans_next_state_values = 0
            with torch.no_grad():
                if target:
                    trans_next_state_values = self.target_model(next_state)
                else:
                    next_state_values = self.model(next_state)
            trans_next_state_values = trans_next_state_values
            trans_next_q_value = self.getQvalue(trans_reward, trans_next_state_values, trans_dones)

            #print("trans_state: ",np.array([trans_state]).shape)
            train_X = np.append(train_X,np.array([trans_state]),axis = 0)

            q_value[0][trans_action] = trans_next_q_value

            train_Y = np.append(train_Y, np.array([q_value[0]]),axis = 0)

        #print(train_X.shape)
        #print(train_Y.shape)
        train_X = torch.FloatTensor(train_X)
        train_Y = torch.FloatTensor(train_Y)

        output = self.model(train_X)
        loss = nn.functional.smooth_l1_loss(train_Y, output)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        #print(len(trans_next_state))
        #print("*************************************")
        #print(trans_next_state)

        #print("trans_state:",len(trans_state))
        #print("trans_action:",len(trans_action))
        # print("trans_reward:",trans_reward)
        # print("trans_next_state:",trans_next_state)


        # while 1:
        #     mLen = len(self.memory)
        #     n = np.arange(mLen).tolist()
        #     pick = random.sample(n, k=self.batch_size)

        #     trans_state_ = trans_state[pick].tolist()
        #     trans_action_ = trans_action[pick].tolist()
        #     trans_reward_ = trans_reward[pick].tolist()
        #     trans_next_state_ = trans_next_state[pick].tolist()
        #     if (None not in trans_next_state_):
        #         break
        #     #print("again")
        # #print(None in trans_next_state_)
        # #print(trans_next_state.shape)
        # ############################################################

        # trans_state = torch.FloatTensor(trans_state_)
        # trans_action = torch.FloatTensor(trans_action_)
        # trans_reward = torch.FloatTensor(trans_reward_)
        # trans_next_state = torch.FloatTensor(trans_next_state_)
        

        # state_action_values = self.model(trans_state).max(1)[0].detach().unsqueeze(1).type(torch.FloatTensor)
        # if target:
        #     next_state_values = self.target_model(trans_next_state).max(1)[0].detach()
        # else:
        #     next_state_values = self.model(trans_next_state).max(1)[0].detach()
        
        # expected_state_action_values = (next_state_values * 0.999) + trans_reward
        # expected_state_action_values = expected_state_action_values.unsqueeze(1).type(torch.FloatTensor)

        # loss = nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values)

        # loss = Variable(loss, requires_grad = True)

        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()
        # #print("optimize")



if __name__ == '__main__':
    #rospy.init_node('machine_dqn')
    rospy.init_node('ros_guide')

    pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
    pub_get_action = rospy.Publisher('get_action', Float32MultiArray, queue_size=5)

    result = Float32MultiArray()
    get_action = Float32MultiArray()

    state_size = 48+2
    action_size = 7

    env = Env(action_size)

    agent = ReinforceAgent(state_size, action_size)

    scores, episodes = [], []

    global_step = 0
    start_time = time.time()

    for e in range(EPISODES):
        rospy.loginfo('EPISODES : %d',e)
        done = False
        state = env.reset()
        rospy.loginfo("Reset env")
        #print(type(state))
        score = 0
        for t in count():
            action = agent.selectAction(state)

            next_state, reward, done = env.step(action)
            #print("state:",len(state))
            #print("next:",next_state)

            # if done:
            #     next_state = None

            agent.appendMemory(state, action, reward, next_state,done)
            # agent.appendMemory(state, action, reward, next_state, done)

            if len(agent.memory) >= agent.train_start:
                if global_step <= agent.target_update:
                    #print("op")
                    agent.optimize()
                else:
                    agent.optimize(True)


            score += reward
            state = next_state
            get_action.data = [action, score, reward]
            pub_get_action.publish(get_action)
            
            if t >= 70: # second for time out interrupt
                rospy.loginfo("Time out!!")
                done = True

            if done:
                #result.data = [score, np.max(agent.q_value)]
                #pub_result.publish(result)
                #agent.updateTargetModel()
                scores.append(score)
                episodes.append(e)

                m, s = divmod(int(time.time() - start_time), 60)
                h, m = divmod(m, 60)

                rospy.loginfo('Ep: %d score: %.2f memory: %d epsilon: %.2f time: %d:%02d:%02d',
                              e, reward, len(agent.memory), agent.epsilon, h, m, s)
                # param_keys = ['epsilon']
                # param_values = [agent.epsilon]
                # param_dictionary = dict(zip(param_keys, param_values))
                break

            global_step += 1
            if global_step % agent.target_update == 0:
                agent.target_model.load_state_dict(agent.model.state_dict())
                rospy.loginfo("UPDATE TARGET NETWORK")

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
        
    

    
