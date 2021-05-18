#!/usr/bin/env python
import gym
import os
import numpy as np
# from stable_baselines3.ppo import MlpPolicy
# from stable_baselines3.ppo import PPO
# from stable_baselines3.dqn import DQN
from stable_baselines import DQN
from stable_baselines.deepq.policies import MlpPolicy
import threading
import time
import configparser
import loggers
import requests
import numpy as np
# docker run --name tb -it -d -v "$(pwd)"/inputs:/inputs akimon/tailbench_2 /bin/bash
# docker run --name tb -it -d -v "$(pwd)"/inputs:/inputs akimon/tailbench_2 /bin/bash

config = configparser.ConfigParser()
config.read('config.ini')

containerReward = {
    # 'reward' : 0,
    'reward' : [],
    # 'count' : 0,
    'lock' : threading.Lock()
}

# containerState = {
#     # First element is svc, 2nd is qtime
#     'state' : [0,0],
#     'count' : 0,
#     'lock' : threading.Lock()
# }

# pcmState = {
#     'state' : {},
#     'count' : 0,
#     'lock' : threading.Lock()
# }



rewardLogger,waryLogger,sjrnLogger,stateLogger,coreLogger,rpsLogger = loggers.setupDataLoggers()






class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        self.startingTime = round(time.time())
        threading.Thread(target=loggers.containerLogger, args=( containerReward, rpsLogger, self.startingTime,  ), daemon=True).start()

        self.action_space = gym.spaces.Discrete(5)
        # self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float64)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64)
        os.system('pqos -R > /dev/null')
        os.system('pqos -e "llc:0=0x30000;" > /dev/null')
        self.appCacheWays = 20
        self.updateWays()
        self.cores  = [ core for core in range(12,24) ]
        self.cores += [ core for core in range(36,48) ]
        self.updateCPUs()
        self.coreStates = {}
        self.previousTime = 0

    def addLowestCStateResidencyCore(self):
        bestCore = -1
        lowestCState = 9999
        for core,coreState in self.coreStates.items():
            if coreState['CStateResidency'] < lowestCState and core not in self.cores:
                lowestCState = coreState['CStateResidency']
                bestCore = core
        self.cores.append(bestCore)
        return bestCore
    
    def popLowestCStateResidencyCore(self):
        bestCore = -1
        lowestCState = 9999
        for core,coreState in self.coreStates.items():
            if coreState['CStateResidency'] < lowestCState and core in self.cores:
                lowestCState = coreState['CStateResidency']
                bestCore = core
        self.cores.remove(bestCore)
        return bestCore
    
    
    def getPCMState(self):
        usedCoreState = [0] * 4
        unusedCoreState = [0] * 4
        nrOfUsedCores = len(self.cores)
        for core,coreState in self.coreStates.items():
            if core in self.cores:
                usedCoreState = [x + y for x, y in zip(usedCoreState, coreState.values())]
            else:
                unusedCoreState = [x + y for x, y in zip(unusedCoreState, coreState.values())]

        usedCoreState = [ metric/nrOfUsedCores for metric in usedCoreState ]
        # if( nrOfUsedCores < 24 ):
        #     unusedCoreState = [ metric/(24 - nrOfUsedCores) for metric in unusedCoreState ]
        return usedCoreState #+ unusedCoreState
    
    def updateCPUStates(self):
        # Hardware state is [ IPCU, MissesU, MissRatioU, IPCUn, MissesUn, MissRatioUn, CStateU, CStateUn ]
        # Hardware state is [ IPCU, MissesU, MissRatioU, CStateU, CStateUn ]
        headers = {'Accept': 'application/json'}
        resp = requests.get('http://localhost:9738/persecond/', headers=headers)
        resp = resp.json()
    
        state = {}
        for physicalCore in resp['Sockets'][1]['Cores']:
            for core in physicalCore['Threads']:
                coreState = {}
                coreState['IPC'] = core['Core Counters']['Instructions Retired Any'] / core['Core Counters']['Clock Unhalted Thread']
                misses = core['Core Counters']['L3 Cache Misses']
                instructions = core['Core Counters']['Instructions Retired Any']
                coreState['MPKI'] = misses/(instructions/1000)
                coreState['MissRatio'] = core['Core Counters']['L3 Cache Misses'] / ( core['Core Counters']['L3 Cache Misses'] + core['Core Counters']['L3 Cache Hits'] )
                coreState['CStateResidency'] = core['Energy Counters']['CStateResidency[0]']
                #Normalize IPC and MPKI with numbers from PARTIES
                coreState['IPC'] = coreState['IPC'] / 0.79
                coreState['MPKI'] = coreState['MPKI'] / 6.28

                state[core['OS ID']] = coreState
        self.coreStates = state
    
    def updateCPUs(self, core=None):
        cores = str(self.cores)[1:-1].replace(' ','')
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
        if(action == 0):
            if(self.appCacheWays < 20):
                self.appCacheWays += 1
                waryLogger.warn("Increasing ways to - %s %s" % (self.appCacheWays, round(time.time()) - self.startingTime ))
                self.updateWays()
            else:
                waryLogger.warn("Maintaining - %s %s" % (self.appCacheWays, round(time.time()) - self.startingTime ))
        elif(action == 1):
            if(self.appCacheWays > 6):
                self.appCacheWays -= 1
                waryLogger.warn("Decreasing ways to - %s %s" % (self.appCacheWays, round(time.time()) - self.startingTime ))
                self.updateWays()
            else:
                waryLogger.warn("Maintaining - %s %s" % (self.appCacheWays, round(time.time()) - self.startingTime ))
        elif(action == 2):
            if(len(self.cores) < 24):
                self.addLowestCStateResidencyCore()
                coreLogger.warn("Increasing cores to - %s %s" % (len(self.cores), round(time.time()) - self.startingTime))
                self.updateCPUs()
            else:
                coreLogger.warn("Maintaining - %s %s" % (len(self.cores), round(time.time()) - self.startingTime))
        elif(action == 3):
            if(len(self.cores) > 14):
                core = self.popLowestCStateResidencyCore()
                coreLogger.warn("Decreasing cores to - %s %s" % (len(self.cores), round(time.time()) - self.startingTime))
                self.updateCPUs(core)
            else:
                coreLogger.warn("Maintaining - %s %s" % (len(self.cores), round(time.time()) - self.startingTime))
        else:
            waryLogger.warn("Maintaining - %s %s" % (self.appCacheWays, round(time.time()) - self.startingTime ))
            coreLogger.warn("Maintaining - %s %s" % (len(self.cores), round(time.time()) - self.startingTime))

    def getState(self):
        # global containerState
        # # while(containerState['count'] == 0):
        # #     time.sleep(0.01)
        state = self.getPCMState()
        # containerState['lock'].acquire()
        # state += [ v/(containerState['count'] * 1e9 ) for v in containerState['state']]
        # containerState['state'] = [0,0]
        # containerState['count'] = 0
        # containerState['lock'].release()
        # state += [self.appCacheWays, len(self.cores)]
        
        stateLogger.info("State is : %s %s" % (state,round(time.time()) - self.startingTime))
        return state

    def getReward(self):
        global containerReward
        while( len(containerReward['reward']) == 0):
            time.sleep(0.01)
        containerReward['lock'].acquire()
        # sjrn = containerReward['reward'] / containerReward['count']
        sjrn95 = np.percentile(containerReward['reward'] , 95)
        sjrn = round(sjrn95/1e6)
        # containerReward['reward'] = []
        # containerReward['count'] = 0
        containerReward['lock'].release()

        sjrnLogger.info("95th percentile is : " + str(sjrn) + " " + str( round(time.time()) - self.startingTime))
        # reward = (-1) * sjrn / 2000 - self.appCacheWays/20 - len(self.cores)/24
        if sjrn > 2500:
            reward = max(-(sjrn/2500)**3, -50)
        else:
            reward = sjrn/2500 + (20/self.appCacheWays + 24/len(self.cores))
        rewardLogger.info("Reward is : " + str(reward) + " " + str( round(time.time()) - self.startingTime))
        return reward

    def clearReward(self):
        global containerReward
        containerReward['lock'].acquire()
        containerReward['reward'] = []
        # containerReward['reward'] = 0
        # containerReward['count'] = 0
        containerReward['lock'].release()

    def step(self, action):
        self.takeAction(action)
        self.clearReward()
        time.sleep(1)
        self.updateCPUStates()
        state = self.getState()
        reward = self.getReward()
        return state, reward, 0, {}

    def reset(self):
        self.updateCPUStates()
        state = self.getState()
        return state  # reward, done, info can't be included

    def close(self):
        return


from pathlib import Path
from datetime import datetime
dt = datetime.now().strftime("%m_%d_%H")
Path("./models/%s" % dt).mkdir(parents=True, exist_ok=True)

env = CustomEnv()

# import torch as th
# policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[dict(pi=[256, 256], vf=[64, 64])])
# policy_kwargs = dict( net_arch=[dict(pi=[512, 256], vf=[32, 16])])


# policy_kwargs = dict( net_arch=[dict(pi=[512, 256,128])])
# model = PPO(MlpPolicy, env, policy_kwargs=policy_kwargs, verbose=1, learning_rate=0.0025, n_steps=150)


# policy_kwargs = dict(   optimizer_class = th.optim.adam,
#                         activation_fn = th.nn.ReLU,
#                         net_arch=[dict(pi=[512, 256,128])])

import tensorflow as tf
# policy_kwargs = dict(  act_fun=tf.nn.relu , net_arch=[512, 256,128] )
policy_kwargs = dict(act_fun=tf.nn.relu, layers=[512,256,128])

model = DQN("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1,
            train_freq = 1  ,
            prioritized_replay=True,
            prioritized_replay_alpha=0.6, prioritized_replay_beta0=0.4, prioritized_replay_beta_iters=20000,
            double_q=True, 
            learning_rate=0.0025, target_network_update_freq=150, learning_starts = 750,
            batch_size=64, buffer_size=1000000,
            gamma=0.99, exploration_fraction = 0.1 , exploration_initial_eps = 1, exploration_final_eps=0.01,
            tensorboard_log="./logs/%s/" % dt, n_cpu_tf_sess=22
            )

# sudo taskset -c 0-11,24-35 python3 controller.py 

# model = PPO(MlpPolicy, env, verbose=1, learning_rate=0.01)

# while True:
model.learn(total_timesteps=250000)
model.save( "./models/%s/model.zip" % dt )
    




