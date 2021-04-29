#!/usr/bin/env python
import gym
import os
import numpy as np
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.ppo import PPO
import threading
import time
import json
import random
import configparser
import dictionaryUtils
import loggers
import signal
import sys
# docker run --name tb -it -d -v "$(pwd)"/inputs:/inputs akimon/tailbench_2 /bin/bash
# docker run --name tb -it -d -v "$(pwd)"/inputs:/inputs akimon/tailbench_2 /bin/bash

config = configparser.ConfigParser()
config.read('config.ini')

containerReward = {
    'reward' : 0,
    'count' : 0,
    'lock' : threading.Lock()
}

containerState = {
    # First element is svc, 2nd is qtime
    'state' : [0,0],
    'count' : 0,
    'lock' : threading.Lock()
}

pcmState = {
    'state' : {},
    'count' : 0,
    'lock' : threading.Lock()
}

def getSystemState(state,usedCPUs):
    usedCPUs = { cpu:state['cpuStates'][cpu] for cpu in usedCPUs }
    unusedCPUs = { cpu:state['cpuStates'][cpu] for cpu in state['cpuStates'].keys() if cpu not in usedCPUs.keys() }

    usedCpuState = {}
    for cpuState in usedCPUs.values():
        usedCpuState = dictionaryUtils.addDictionaries(usedCpuState,cpuState)
    usedCpuState = dictionaryUtils.divideDictionaries(usedCpuState,len(usedCPUs))
    usedCpuState['socketState'] = state['socketState']
    usedCpuState = dictionaryUtils.divideDictionaries(usedCpuState,pcmState['count'])

    if(len(usedCPUs) == 24):
        unusedCpuState = { 'IPC': 0, 'Misses': 0, 'MissRatio': 0 }
    else:
        unusedCpuState = {}
        for cpuState in unusedCPUs.values():
            unusedCpuState = dictionaryUtils.addDictionaries(unusedCpuState,cpuState)
        unusedCpuState = dictionaryUtils.divideDictionaries(unusedCpuState,len(usedCPUs))
        unusedCpuState = dictionaryUtils.divideDictionaries(unusedCpuState,pcmState['count'])
        unusedCpuState.pop('cstate',None)

    return list(usedCpuState.values()) + list(unusedCpuState.values())

rewardLogger,waryLogger,sjrnLogger,stateLogger,coreLogger,rpsLogger = loggers.setupDataLoggers()

def getBestCPU(cpuStates,currentCPUs):
    bestCPU = '-1'
    bestCPUState = 9999
    for cpu,cpuState in cpuStates.items():
        if cpuState['cstate'] < bestCPUState and cpu not in currentCPUs:
            bestCPUState = cpuState['cstate']
            bestCPU = cpu
    return bestCPU


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        self.startingTime = round(time.time())
        threading.Thread(target=loggers.containerLogger, args=(containerState, containerReward, rpsLogger, self.startingTime,  ), daemon=True).start()
        threading.Thread(target=loggers.pcmLogger, args=(pcmState, ), daemon=True).start()

        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float64)
        self.appCacheWays = 2
        # { 12,13,14,15,16,17,18,19,20,21,22,23, 36,37,38,39,40,41,42,43,44,45,46,47 };
        self.cores  = [ str(core) for core in range(12,24) ]
        self.cores += [ str(core) for core in range(36,48) ]
        self.updateCPUs()
        print('Starting')
    
    def updateCPUs(self, core=None):
        cores = ','.join(self.cores)
        os.system('docker update --cpus=%s --cpuset-cpus="%s" tb > /dev/null' % (len(self.cores), cores))
        os.system('pqos -a "llc:1=%s;" > /dev/null' % cores)
        if(core != None):
            os.system('pqos -a "llc:0=%s;" > /dev/null' % core)

    def updateWays(self):
        os.system('sudo pqos -e "llc:1=%s;" > /dev/null' % self.formatForCAT(self.appCacheWays))

    def formatForCAT(self, ways):
        res = 1 << ways - 1
        res = res + res - 1
        return hex(res)

    def takeAction(self, action):
        global pcmState
        if(action == 0):
            if(self.appCacheWays < 20):
                self.appCacheWays += 1
                waryLogger.warn("Increasing ways to - %s %s" % (self.appCacheWays, round(time.time()) - self.startingTime ))
                self.updateWays()
        elif(action == 1):
            if(self.appCacheWays > 2):
                self.appCacheWays -= 1
                waryLogger.warn("Decreasing ways to - %s %s" % (self.appCacheWays, round(time.time()) - self.startingTime ))
                self.updateWays()
        elif(action == 2):
            if(len(self.cores) < 24):
                self.cores.append( getBestCPU(pcmState['state']['cpuStates'],self.cores) )
                coreLogger.warn("Increasing cores to - %s %s" % (len(self.cores), round(time.time()) - self.startingTime))
                self.updateCPUs()
        elif(action == 3):
            if(len(self.cores) > 2):
                core = self.cores.pop(random.randrange(len(self.cores)))
                coreLogger.warn("Decreasing cores to - %s %s" % (len(self.cores), round(time.time()) - self.startingTime))
                self.updateCPUs(core)
        else:
            waryLogger.warn("Maintaining - %s %s" % (self.appCacheWays, round(time.time()) - self.startingTime ))
            coreLogger.warn("Maintaining - %s %s" % (len(self.cores), round(time.time()) - self.startingTime))
        pcmState['state'] = {}
        pcmState['count'] = 0

    def getState(self):
        global pcmState, containerState
        while(pcmState['count'] == 0 or containerState['count'] == 0):
            time.sleep(0.01)
        pcmState['lock'].acquire()
        state = getSystemState(pcmState['state'],self.cores) 
        pcmState['lock'].release()
        containerState['lock'].acquire()
        state += [ v/(containerState['count'] * 1e9 ) for v in containerState['state']]
        containerState['state'] = [0,0]
        containerState['count'] = 0
        containerState['lock'].release()
        state += [self.appCacheWays, len(self.cores)]
        
        stateLogger.info("State is : %s %s" % (state,round(time.time()) - self.startingTime))
        return state

    def getReward(self):
        global containerReward
        while(containerReward['count']==0):
            time.sleep(0.01)
        containerReward['lock'].acquire()
        sjrn = containerReward['reward'] / containerReward['count']
        sjrn = round(sjrn/1e6)
        containerReward['reward'] = 0
        containerReward['count'] = 0
        containerReward['lock'].release()

        sjrnLogger.info("Response time is : " + str(sjrn) + " " + str( round(time.time()) - self.startingTime))
        reward = (-2) * sjrn / 2000 - self.appCacheWays/20 - len(self.cores)/24
        rewardLogger.info("Reward is : " + str(reward) + " " + str( round(time.time()) - self.startingTime))
        return reward


    def step(self, action):
        self.takeAction(action)
        time.sleep(5)
        state = self.getState()
        reward = self.getReward()
        return state, reward, 0, {}

    def reset(self):
        state = self.getState()
        return state  # reward, done, info can't be included

    def close(self):
        return

def signal_handler(sig, frame):
    print('Exiting')
    sys.exit(0)


env = CustomEnv()
model = PPO(MlpPolicy, env, verbose=1)

while True:
    model.learn(total_timesteps=10000)

