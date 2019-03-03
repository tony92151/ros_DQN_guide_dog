#!/home/rospc/torch_gpu_ros/bin/python

# modefied from https://github.com/ROBOTIS-GIT/turtlebot3_machine_learning/blob/master/turtlebot3_dqn/nodes/moving_obstacle

import rospy
import time
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose
from gazebo_msgs.msg import ModelState, ModelStates




class MoveTarget():
    def __init__(self):
        #rospy.init_node('moving_target')
        self.pub_model = rospy.Publisher('gazebo/set_model_state', ModelState, queue_size=3)

        self.goalTable = [[1,1],[1,-1],[-1,-1],[-1,1]]
        

    def movingTo(self,goal_x,goal_y):
        #pub_model = rospy.Publisher('gazebo/set_model_state', ModelState, queue_size=1)
        
        obstacle = ModelState()
        model = rospy.wait_for_message('gazebo/model_states', ModelStates)
        #print(model.name)
        #print(model.twist[2])
        # print(type(obstacle))
        #time.sleep(5)
        for i in range(len(model.name)):
            if model.name[i] == 'target':  # the model name is defined in .xacro file
                obstacle.model_name = 'target'
                
                obstacle.pose = model.pose[i]
                obstacle.pose.position.x = float(goal_x)
                obstacle.pose.position.y = float(goal_y)
                # obstacle.twist = Twist()
                # obstacle.twist.angular.z = 0.5
                self.pub_model.publish(obstacle)
                #time.sleep(5)

    def movingAt(self,p):
        #pub_model = rospy.Publisher('gazebo/set_model_state', ModelState, queue_size=1)
        
        obstacle = ModelState()
        model = rospy.wait_for_message('gazebo/model_states', ModelStates)
        #print(model.name)
        #print(model.twist[2])
        # print(type(obstacle))
        #time.sleep(5)
        for i in range(len(model.name)):
            if model.name[i] == 'target':  # the model name is defined in .xacro file
                obstacle.model_name = 'target'
                
                obstacle.pose = model.pose[i]
                p = p%4
                #print(self.goalTable[p][0])
                obstacle.pose.position.x = float(self.goalTable[p][0])
                obstacle.pose.position.y = float(self.goalTable[p][1])
                # obstacle.twist = Twist()
                # obstacle.twist.angular.z = 0.5
                self.pub_model.publish(obstacle)
                rospy.loginfo("Goal at local %d" ,p)
                #time.sleep(5)


        # obstacle.model_name = 'target'
        
        # #obstacle.pose = model.pose[i]
        # p = p%4
        # #print(self.goalTable[p][0])
        # obstacle.pose.position.x = float(self.goalTable[p][0])
        # obstacle.pose.position.y = float(self.goalTable[p][1])
        # # obstacle.twist = Twist()
        # # obstacle.twist.angular.z = 0.5
        # self.pub_model.publish(obstacle)
        # #time.sleep(5)
        return float(self.goalTable[p][0]),float(self.goalTable[p][1])
                

            


# if __name__ == '__main__':
    
#     try:
#         move = MoveTarget()
#         rospy.loginfo("Init success !!")
#         #time.sleep(1)
#     except:
#         rospy.loginfo("Init error !!")
#     ti = 1
#     while 1:
#         move.movingAt(0)
#         rospy.loginfo("Moving!!")
#         time.sleep(ti)
#         move.movingAt(1)
#         time.sleep(ti)
#         move.movingAt(2)
#         time.sleep(ti)
#         move.movingAt(3)
#         time.sleep(ti)
