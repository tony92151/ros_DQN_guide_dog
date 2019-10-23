import argparse 
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.animation as animation 
import random
import time

import rospy
import sys
from std_msgs.msg import Float32MultiArray, Float32


class Plot:
    def __init__(self):
        self.sub = rospy.Subscriber("result", Float32MultiArray, self.get_array)
        plt.ion()
        self.fig, self.ax = plt.subplots(2)
        self.data = []
        self.ToReward = []
        self.AvgQ = []


    def plotUp(self):
        re = np.array(self.data)[:,0]
        #print(re)
        avg = np.array(self.data)[:,1]
        print(avg)
        #self.ax[0]
        self.ax[0].plot(avg)

        plt.show()

        self.fig.canvas.draw()
        #self.fig.canvas.flush_events()
        
        

    def get_array(self,array):
        temp = []
        array.data = list(array.data)
        for i in array.data:
            temp.append(round(i,3))
        self.data.append(temp)
        self.plotUp()

if __name__ == '__main__': 
    a = Plot()
    rospy.init_node('Plot')
    #a.plotUp()
    rospy.spin()
    

	# a = gameOfLive(30)
	# a.initGame()

	# # ani = animation.FuncAnimation(a.fig, 
	# # 							a.step, 
	# # 							frames = 1, 
	# # 							interval=50)
	# # plt.show()
	# #
	# for i in range(1000):
	# 	a.step(0)
	# 	a.fig.canvas.draw()
	# 	a.fig.canvas.flush_events()
	# 	#time.sleep(0.05)
