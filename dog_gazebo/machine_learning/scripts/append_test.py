

import random
from collections import deque
import time
import numpy as np

memory = deque(maxlen=10000)


def add(state, action, reward, next_state):
    global memory
    memory.append((state, action, reward, next_state))

if __name__ == '__main__':
    for i in range(3):
        add(i,i,i,i)
        print("memory : ")
        print(memory)
        time.sleep(1)
    for i in range(3):
        print("memory :", format(i))
        arr = np.array(memory)
        print(arr[i,:])
        time.sleep(1)
