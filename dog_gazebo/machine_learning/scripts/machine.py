#!/home/ros/torch_gpu_ros/bin/python

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
        
        self.ln1 = nn.Linear(in_features = state_size, out_features = 24)
        #self.ba = nn.BatchNorm1d(24)
        self.ln2 = nn.Linear(in_features = 24, out_features = 12)
        self.dr = nn.Dropout(0.5)
        self.ln3 = nn.Linear(in_features = 12, out_features = action_size)
            
    def forward(self, s):
        s = self.ln1(s)
        #s = self.ba(s)
        s = self.ln2(s)
        s = self.dr(s)
        s = self.ln3(s)
        s = s.view(s.size(0),-1)
        return s



class ReinforceAgent():
    def __init__(self, state_size, action_size):
        #self.pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
        #self.dirPath = os.path.dirname(os.path.realpath(__file__))
        #self.dirPath = self.dirPath.replace('turtlebot3_dqn/nodes', 'turtlebot3_dqn/save_model/stage_1_')
        #self.result = Float32MultiArray()

        self.load_model = False
        self.load_episode = 0
        self.state_size = state_size
        self.action_size = action_size
        self.episode_step = 6000
        self.target_update = 2000
        self.discount_factor = 0.99
        self.learning_rate = 0.00025
        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.05

        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # if torch.cuda.is_available():
        #     rospy.loginfo("Use gpu")

        self.batch_size = 16
        self.train_start = 128
        self.memory = deque(maxlen=10000)

        self.model = DQN(self.state_size,self.action_size)
        self.target_model = DQN(self.state_size,self.action_size).eval()

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
            

    def appendMemory(self, state, action, reward, next_state):
        #print(len(self.memory))
        self.memory.append((state, action, reward, next_state))

    # def appendMemory(self, state, action, reward, next_state, done):
    #     self.memory.append((state, action, reward, next_state, done))

    def optimize(self,target=False):
        ############################################################
        trans = np.array(self.memory)

        trans_state = trans[:,0]
        trans_action = trans[:,1]
        trans_reward = trans[:,2]
        trans_next_state = trans[:,3]
        #print(len(trans_next_state))
        #print("*************************************")
        #print(trans_next_state)

        #print("trans_state:",len(trans_state))
        #print("trans_action:",len(trans_action))
        # print("trans_reward:",trans_reward)
        # print("trans_next_state:",trans_next_state)
        while 1:
            mLen = len(self.memory)
            n = np.arange(mLen).tolist()
            pick = random.sample(n, k=self.batch_size)

            trans_state_ = trans_state[pick].tolist()
            trans_action_ = trans_action[pick].tolist()
            trans_reward_ = trans_reward[pick].tolist()
            trans_next_state_ = trans_next_state[pick].tolist()
            if (None not in trans_next_state_):
                break
            #print("again")
        #print(None in trans_next_state_)
        #print(trans_next_state.shape)
        ############################################################

        trans_state = torch.FloatTensor(trans_state_)
        trans_action = torch.FloatTensor(trans_action_)
        trans_reward = torch.FloatTensor(trans_reward_)
        trans_next_state = torch.FloatTensor(trans_next_state_)
        

        state_action_values = self.model(trans_state).max(1)[0].detach().unsqueeze(1).type(torch.FloatTensor)
        next_state_values = self.target_model(trans_next_state).max(1)[0].detach()
        expected_state_action_values = (next_state_values * 0.999) + trans_reward
        expected_state_action_values = expected_state_action_values.unsqueeze(1).type(torch.FloatTensor)

        loss = nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values)

        loss = Variable(loss, requires_grad = True)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        #print("optimize")



if __name__ == '__main__':
    rospy.init_node('machine_dqn')
    ####
    #rospy.init_node('turtlebot3_dqn_stage_1')
    #pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
    #pub_get_action = rospy.Publisher('get_action', Float32MultiArray, queue_size=5)
    result = Float32MultiArray()
    get_action = Float32MultiArray()

    state_size = 48+2
    action_size = 7

    env = Env(action_size)

    agent = ReinforceAgent(state_size, action_size)

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

            if done:
                next_state = None

            agent.appendMemory(state, action, reward, next_state)
            # agent.appendMemory(state, action, reward, next_state, done)

            if len(agent.memory) >= agent.train_start:
                if global_step <= agent.target_update:
                    #print("op")
                    agent.optimize()
                else:
                    agent.optimize(True)


            if t >= 50: # second for time out interrupt
                rospy.loginfo("Time out!!")
                done = True

            if done:
                # result.data = [score, np.max(agent.q_value)]
                # pub_result.publish(result)
                # agent.updateTargetModel()
                # scores.append(score)
                # episodes.append(e)
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
        
    

    
