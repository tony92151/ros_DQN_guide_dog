#!/usr/bin/env python
#################################################################################
# Copyright 2018 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################

# Authors: Gilbert #

import rospy
import time
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose
from gazebo_msgs.msg import ModelState, ModelStates

class Moving():
    def __init__(self):
        self.pub_model = rospy.Publisher('gazebo/set_model_state', ModelState, queue_size=1)
        self.moving()

    def moving(self,goal_x,goal_y):
        while not rospy.is_shutdown():
            obstacle = ModelState()
            model = rospy.wait_for_message('gazebo/model_states', ModelStates)
            for i in range(len(model.name)):
                if model.name[i] == 'target':
                    obstacle.model_name = 'target'
                    po = model.pose[i]
                    po.position.x = goal_x
                    po.position.y = goal_y
                    po.position.z = 0
                    #po.orientation.x
                    obstacle.pose = po
                    #obstacle.twist = Twist()
                    #obstacle.twist.angular.z = 0.5
                    self.pub_model.publish(obstacle)
                    time.sleep(0.1)

def main():
    rospy.init_node('moving_obstacle')
    moving = Moving()

if __name__ == '__main__':
    main()
