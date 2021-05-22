#!/usr/bin/env python3
from datetime import datetime
from pathlib import Path
import subprocess
import gym
import os
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
from stable_baselines import DQN
import threading
import time
import configparser
import loggers
import tensorflow as tf
import numpy as np
import perfmon
import struct
import random
import psutil
from collections import deque
from neural import procedure_continuous_tasks as deepq

import logging
import os

# logging.disable(logging.WARNING)
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

config = configparser.ConfigParser()
config.read('config.ini')

containerReward = {
    'reward': [],
    'lock': threading.Lock()
}

window_size = 5

deques = {
    'UNHALTED_CORE_CYCLES': deque([], maxlen=window_size),
    'INSTRUCTION_RETIRED': deque([], maxlen=window_size),
    'PERF_COUNT_HW_CPU_CYCLES': deque([], maxlen=window_size),
    'UNHALTED_REFERENCE_CYCLES': deque([], maxlen=window_size),
    'UOPS_RETIRED': deque([], maxlen=window_size),
    'BRANCH_INSTRUCTIONS_RETIRED': deque([], maxlen=window_size),
    'MISPREDICTED_BRANCH_RETIRED': deque([], maxlen=window_size),
    'PERF_COUNT_HW_BRANCH_MISSES': deque([], maxlen=window_size),
    'LLC_MISSES': deque([], maxlen=window_size),
    'PERF_COUNT_HW_CACHE_L1D': deque([], maxlen=window_size),
    'PERF_COUNT_HW_CACHE_L1I': deque([], maxlen=window_size),
}


rewardLogger, wayLogger, sjrnLogger, stateLogger, coreLogger, rpsLogger = loggers.setupDataLoggers()


EVENTS = ['UNHALTED_CORE_CYCLES', 'INSTRUCTION_RETIRED', 'PERF_COUNT_HW_CPU_CYCLES', 'UNHALTED_REFERENCE_CYCLES', \
        'UOPS_RETIRED', 'BRANCH_INSTRUCTIONS_RETIRED', 'MISPREDICTED_BRANCH_RETIRED', \
        'PERF_COUNT_HW_BRANCH_MISSES', 'LLC_MISSES', 'PERF_COUNT_HW_CACHE_L1D', \
        'PERF_COUNT_HW_CACHE_L1I']
# EVENT_MAX = [1009566688, 2200098315, 1413332030, 4404609, 390883292,
#              18043023, 1413719982, 18032364, 20587451, 41154, 7496985285]

EVENT_MAX = [1251666326, 2697635738, 1502160478, 1385062673, 3899393008, 265396012, 42954597, 42960949, 1598918, 14667253, 30645]

EVENT_MAX = [e*2 for e in EVENT_MAX]


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        global deques, window_size
        self.deques = deques
        self.window_size = window_size

        self.startingTime = round(time.time())
        threading.Thread(target=loggers.containerLogger, args=(containerReward, rpsLogger, self.startingTime,), daemon=True).start()

        time.sleep(3)
        self.pid = 0
        for proc in psutil.process_iter():
            if 'img-dnn_integra' in proc.name():
                self.pid = proc.pid
                print(self.pid)
        if self.pid == 0 :
            print("Couldn't find app pid, exiting...")
            exit(-1)

        self.tid = list()
        for tid in psutil.Process(self.pid).threads():
            self.tid.append(tid.id)
        
        self.coreMapping = dict()
        self.threadMapping = dict()

        os.system('pqos -R > /dev/null')
        os.system('pqos -e "llc:0=0x30000;" > /dev/null')

        self.appCacheWays = 20
        self.updateWays()

        self.cores = [core for core in range(12, 24)]
        self.cores += [core for core in range(36, 48)]

        self.allCores = [core for core in range(12, 24)]
        self.allCores += [core for core in range(36, 48)]

        self.updateCPUs()

        self.startPerfmon()

    def increaseCores(self, desiredCores):
        for 



    def initialMapping(self):
        for pid, core in zip(self.tid, self.allCores):
            self.threads[pid] = psutil.Process(pid=pid)
            self.threads[pid].cpu_affinity([core])

    def startPerfmon(self):
        self.sessions = [None] * len(self.threads)
        for i,id in enumerate(self.threads):
            self.sessions[i] = perfmon.PerThreadSession(int(id), EVENTS)
            self.sessions[i].start()

    def getPMC(self):
        pmc = [0] * len(EVENTS)
        for i in range(0,len(EVENTS)):
            for session in self.sessions:
                count = struct.unpack("L", session.read(i))[0]
                pmc[i] += float(count)
            pmc[i] /= len(self.threads)
        return pmc

    def updateCPUs(self, core=None):
        cores = str(self.cores)[1:-1].replace(' ', '')
        os.system('taskset -apc %s %s > /dev/null' % (cores,self.pid))
        os.system('pqos -a "llc:1=%s;" > /dev/null' % cores)
        if(core != None):
            os.system('pqos -a "llc:0=%s;" > /dev/null' % core)

    def formatForCAT(self, ways):
        res = 1 << ways - 1
        res = res + res - 1
        return hex(res)

    def updateWays(self):
        os.system('sudo pqos -e "llc:1=%s;" > /dev/null' %
                  self.formatForCAT(self.appCacheWays))

    def takeAction(self, action):
        if(action == 0):
            if(self.appCacheWays < 20):
                self.appCacheWays += 1
                wayLogger.warn("Increasing ways to - %s %s" % (self.appCacheWays, round(time.time()) - self.startingTime))
                self.updateWays()
            else:
                wayLogger.warn("Maintaining - %s %s" % (self.appCacheWays, round(time.time()) - self.startingTime))
        elif(action == 1):
            if(self.appCacheWays > 3):
                self.appCacheWays -= 1
                wayLogger.warn("Decreasing ways to - %s %s" % (self.appCacheWays, round(time.time()) - self.startingTime))
                self.updateWays()
            else:
                wayLogger.warn("Maintaining - %s %s" % (self.appCacheWays, round(time.time()) - self.startingTime))
        elif(action == 2):
            if(len(self.cores) < 24):
                unusedCores = [core for core in range(12, 24) if core not in self.cores]
                unusedCores += [core for core in range(36, 48) if core not in self.cores]
                self.cores.append(random.choice(unusedCores))
                coreLogger.warn("Increasing cores to - %s %s" % (len(self.cores), round(time.time()) - self.startingTime))
                self.updateCPUs()
            else:
                coreLogger.warn("Maintaining - %s %s" % (len(self.cores), round(time.time()) - self.startingTime))
        elif(action == 3):
            if(len(self.cores) > 4):
                core = random.choice(self.cores)
                self.cores.remove(core)
                coreLogger.warn("Decreasing cores to - %s %s" % (len(self.cores), round(time.time()) - self.startingTime))
                self.updateCPUs(core)
            else:
                coreLogger.warn("Maintaining - %s %s" % (len(self.cores), round(time.time()) - self.startingTime))
        else:
            wayLogger.warn("Maintaining - %s %s" % (self.appCacheWays, round(time.time()) - self.startingTime))
            coreLogger.warn("Maintaining - %s %s" % (len(self.cores), round(time.time()) - self.startingTime))

    def running_mean(self, x, N):
        cumsum = np.cumsum(np.insert(x, 0, 0))
        return (cumsum[N:] - cumsum[:-N]) / float(N)

    def norm_data(self, cur_counter=None):
        state_space = []
        run_mean = []
        for i in range(0, len(EVENTS)):
            out = cur_counter[i]/(EVENT_MAX[i])
            state_space.append(out)
            deques[EVENTS[i]].append(out)
        if len(self.deques['UNHALTED_CORE_CYCLES']) < self.window_size:
            return np.array(state_space)
        else:
            for _, val in self.deques.items():
                run_mean.append(self.running_mean(val, self.window_size)[0])
            return np.array(run_mean)

    def getState(self, before, after):
        state = [0] * len(EVENTS)
        for i in range(0,len(EVENTS)):
            state[i] = after[i] - before[i]

        normalized = self.norm_data(state)
        stateLogger.info("State is : %s %s" % (list(normalized), round(time.time()) - self.startingTime))

        return normalized

    def getReward(self):
        global containerReward
        while(len(containerReward['reward']) == 0):
            time.sleep(0.01)
            rewardLogger.info("Waiting on reward " +
                              str(round(time.time()) - self.startingTime))
        containerReward['lock'].acquire()
        sjrn99 = np.percentile(containerReward['reward'], 99)
        qos = round(sjrn99/1e3)
        containerReward['lock'].release()

        sjrnLogger.info("99th percentile is : " + str(qos) + " " + str(round(time.time()) - self.startingTime))
        qosTarget = 3000
        if qos > qosTarget:
            reward = max(-(qos/qosTarget)**3, -50)
        else:
            # reward = qos/qosTarget + (20/self.appCacheWays + 24/len(self.cores))*2
            reward = qosTarget/qos + (20/self.appCacheWays + 24/len(self.cores))*2
        rewardLogger.info("Reward is : " + str(reward) + " " + str(round(time.time()) - self.startingTime))
        return reward

    def clearReward(self):
        global containerReward
        containerReward['lock'].acquire()
        containerReward['reward'] = []
        containerReward['lock'].release()
    
    def reconcile(self,cores,ways):
        if cores > self.cores:


    def step(self, action):
        pmc_before = self.getPMC()
        self.takeAction(action)
        self.clearReward()
        time.sleep(2)
        pmc_after = self.getPMC()
        state = self.getState(pmc_before, pmc_after)
        reward = self.getReward()
        return state, reward, 0, {}

    def reset(self):
        state = [0] * len(EVENTS)
        return state

    def close(self):
        return


dt = datetime.now().strftime("%m_%d_%H")
Path("./models/%s" % dt).mkdir(parents=True, exist_ok=True)

env = CustomEnv()

my_agent = deepq.learn_continuous_tasks(state_space_length=len(EVENTS),\
        way_space=np.arrange(0,20),\
        core_space=np.arrange(0,24,1),\
        NUM_APPS= 1)

current_state  = []
previous_state = []
current_step = -1

while True:
    if current_step >= 0:
        my_agent.compute_temporal_diff()
        my_agent.update_target_network_weights()

    action, action_idxes, _, _, _ = my_agent.determine_action(current_state, None)

    action_core_count = int(action[0])
    action_way_count = int(action[1])
    print('Taking action %s     %s' % (action_core_count,action_way_count))

    previous_state = current_state

    current_state, reward, _, _ = env.step([action_core_count,action_way_count])

    my_agent.add_to_replay_buff(prev_state = previous_state, action_idxes = action_idxes,\
            reward = reward, new_state = current_state, workload_num=1)


    current_step += 1
