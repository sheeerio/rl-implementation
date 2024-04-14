import numpy as np
import matplotlib.pyplot as plt
import time
import random

class OG():
    def __init__(self, SIZE):
        self.size = SIZE
        self.x = np.random.randint(0, self.size)
        self.y = np.random.randint(0, self.size)
    
    def __str__(self):
        return f"{self.x}, {self.y}"
    
class OGEnvironment():
    def __init__(self):
        self.size = 5
        self.n_actions = 4
        self.player = OG(self.size)
        self.x = self.player.x
        self.y = self.player.y
        self.color = {"player":(0,0,255)}
        self.reward = 0
    
    def reset(self):
        self.x = self.player.x
        self.y = self.player.y
        return (self.x, self.y)
    
    def step(self, action):
        if action==-1:
            print("?")
            action = self.player.policy()
        if action==0:
            self.move(x=1,y=0)
        elif action==1:
            self.move(x=0,y=1)
        elif action==2:
            self.move(-1, 0)
        elif action==3:
            self.move(0, -1)
        return self.x, self.y, self.reward
    
    def move(self, x, y):
        self.reward = 0
        #goal
        if self.x==4 and self.y==4:
            self.reward = 10
            self.x = 0
            self.y = 0
        #terminal
        elif self.x==3 and self.y==4:
            self.reward = -15
            self.x = 0
            self.y = 0
        else:
            if self.x+x < 0 or self.x+x > self.size-1 or \
                self.y+y < 0 or self.y+y > self.size-1:
                self.reward = 0
            
            self.x = np.clip(self.x+x, 0, self.size-1)
            self.y = np.clip(self.y+y, 0, self.size-1)
    
    def render(self, renderTime = 100):
        env = np.ones((self.size, self.size, 3), dtype=np.uint8)*255
        env[self.x][self.y] = self.color["player"]
        plt.xticks(np.arange(-0.5,4.5,1),np.arange(5))
        plt.yticks(np.arange(-0.5,4.5,1),np.arange(5))
        plt.grid('True')
        plt.imshow(np.array(env))
        plt.pause(renderTime/100)

    def sample_action(self):
        return np.random.randint(0, self.n_actions)

    def plot_grid_values(self, values):
        fig, axs = plt.subplots()
        axs.axis('off')
        table = axs.table(cellText=values,bbox=[0, 0, 1, 1],cellLoc="center")
        plt.show()
    
    def plot_policy(self, policy):
        P = []
        for y in range(5):
            p = []
            for x in range(5):
                if policy[y,x] == 0:
                    p.append(">")
                elif policy[y,x] == 1:
                    p.append("v")
                elif policy[y,x] == 2:
                    p.append("<")
                else:
                    p.append("^")
            P.append(p)
        fig, axs = plt.subplots(1,1)
        axs.axis('off')
        table = axs.table(cellText=P,bbox=[0, 0, 1, 1],cellLoc="center")
        plt.show()



