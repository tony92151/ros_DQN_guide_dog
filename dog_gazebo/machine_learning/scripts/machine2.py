#!/usr/bin/env python


import rospy
import rospkg
import os
import json
import numpy as np
import random
import time
import sys
#https://github.com/dokelung/Python-QA/blob/master/questions/standard_lib/Python%20%E7%8D%B2%E5%8F%96%E6%96%87%E4%BB%B6%E8%B7%AF%E5%BE%91%E5%8F%8A%E6%96%87%E4%BB%B6%E7%9B%AE%E9%8C%84(__file__%20%E7%9A%84%E4%BD%BF%E7%94%A8%E6%96%B9%E6%B3%95).md
rospy.loginfo(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
print(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

rospack = rospkg.RosPack()
env_path = rospack.get_path('machine_learning')

sys.path.append(env_path+'/env')


from environment_stage_1 import Env

print("767676")

    
