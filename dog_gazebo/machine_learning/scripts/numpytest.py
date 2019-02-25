import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

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





if __name__ == '__main__':
    createMem(10)
    m = np.array(memory)

    me = m[:,0]
    print(me)

    n = np.arange(10)
    #l = random.sample(n, long-5)
    seq = [1, 2, 3, 4, 5]
    pick = random.sample(seq, k=5)
    print("=============================================================")
    print(me[pick])

    print()

    

    
