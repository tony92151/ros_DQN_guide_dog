#!/home/rospc/torch_gpu_ros/bin/python

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


EPISODES = 3000




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
    time.sleep(10)
    print("reset")
    env.reset()

            


    
