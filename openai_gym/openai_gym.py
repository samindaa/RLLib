#import sys
#sys.path.append("/Users/saminda/Projects/gym")
import gym
import time
import socket

import numpy as np

def createMsg(observations, reward, episode_state):
    msg = ""
    for x in np.nditer(observations):
        msg += str(x)
        msg += " "
    msg += str(reward)
    msg += " "
    msg += str(episode_state)
    return msg

def Main():
    host = '127.0.0.1'
    port = 2345
    
    # Gym
    gymEnv = "MountainCar-v0"
    env = gym.make(gymEnv)

    # toRLLib
    mySocket = socket.socket()
    mySocket.connect((host,port))
    
    # Init evn
    observations = env.reset()
    reward = 0
    done = False
    info = None
    episode_state = 0 # 0 => new epoch starts, 1 episode starts, 2 episode continue, 3 episode ends
    
    msg = "__ENV__ " + gymEnv
    #send init command
    #msg = createMsg(observations, reward, episode_state)
    print("msg_init: " + msg)
    mySocket.send(msg.encode())
    msg = mySocket.recv(1024).decode()
    print("msg: " + msg)
    
    if (msg != "OK"):
        print("Agent is not ready")
        return
    
    print("epoch starts")
    
    episode_state = 1
    t = 0  
    while True:
        # Viz
        #env.render()
        # CREATE MSG
        msg = createMsg(observations, reward, episode_state)
        
        #print("msg: " + msg)
        mySocket.send(msg.encode())
        action_tp1 = mySocket.recv(1024).decode()
        #print("action_tp1: " + action_tp1)
        
        if (episode_state == 3):
            observations = env.reset()
            episode_state = 1
            #print("new episode starts")
            continue
            
        #state_var, reward, done, info = env.step(env.action_space.sample())
        observations, reward, done, info = env.step(int(action_tp1))
        
        t = t + 1;
        if (done == False):
            episode_state = 2
        else:
            # create terminal msg
            episode_state = 3
            print("Episode finished after {} timesteps".format(t))
            t = 0;
            
        #time.sleep(4)          
            
    mySocket.close()
    
if __name__ == '__main__':
    Main()