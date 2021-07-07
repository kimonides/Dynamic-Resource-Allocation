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
import tensorflow as tf
import numpy as np
import perfmon
import struct
import random
import psutil
from collections import deque
import shlex

import logging
import loggers

import os

# import loggers

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

dt = datetime.now().strftime("%m_%d_%H")
Path("./models/%s_img_dnn" % dt).mkdir(parents=True, exist_ok=True)
# self.rewardLogger, wayLogger, self.sjrnLogger, self.stateLogger, coreLogger, rpsLogger, coreMapLogger = loggers.setupDataLoggers('img_dnn')


EVENTS = ['UNHALTED_CORE_CYCLES', 'INSTRUCTION_RETIRED', 'PERF_COUNT_HW_CPU_CYCLES', 'UNHALTED_REFERENCE_CYCLES', \
        'UOPS_RETIRED', 'BRANCH_INSTRUCTIONS_RETIRED', 'MISPREDICTED_BRANCH_RETIRED', \
        'PERF_COUNT_HW_BRANCH_MISSES', 'LLC_MISSES', 'PERF_COUNT_HW_CACHE_L1D', \
        'PERF_COUNT_HW_CACHE_L1I']

EVENT_MAX = [1251666326, 2697635738, 1502160478, 1385062673, 3899393008, 265396012, 42954597, 42960949, 1598918, 14667253, 30645]

EVENT_MAX = [e*2 for e in EVENT_MAX]


class CustomEnv(gym.Env):
    def __init__(self, cores, ways):
        super(CustomEnv, self).__init__()

        self.rewardLogger, self.wayLogger, self.sjrnLogger, self.stateLogger, self.coreLogger, self.rpsLogger, self.coreMapLogger = loggers.setupDataLoggers('img_dnn')

        global deques, window_size
        self.deques = deques
        self.window_size = window_size

        self.startingTime = round(time.time())
        self.process = None

        self.cores = cores
        # self.cores = [core for core in range(12, 24)]
        # self.cores += [core for core in range(36, 48)]

        self.allCores = [core for core in range(12, 24)]
        self.allCores += [core for core in range(36, 48)]

        # self.appCacheWays = 20
        self.appCacheWays = ways

        threading.Thread(target=self.containerLogger, args=(containerReward,), daemon=True).start()

        time.sleep(3)
        if self.process == None:
            print("Couldn't find app pid, exiting...")
            exit(-1)
        
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(low=0, high=1.5, shape=(13,), dtype=np.float64)

        os.system('pqos -R > /dev/null')
        os.system('pqos -e "llc:0=0x30000;" > /dev/null')

        # self.updateWays()
        self.updateCores()
        # self.initialMapping()

        cores = str(self.cores)[1:-1].replace(' ', '')
        os.system('pqos -a "llc:1=%s;" > /dev/null' % cores)

    def updateCores(self):
        cores_string = str(self.cores)[1:-1].replace(' ', '')
        os.system('pqos -a "llc:1=%s;" > /dev/null' % cores_string)

        unused_cores = [core for core in self.allCores if core not in  self.cores]
        unused_cores_string = str(unused_cores)[1:-1].replace(' ', '')
        os.system('pqos -a "llc:0=%s;" > /dev/null' % unused_cores_string)


        core_index = 0
        for t in self.tid:
            p = psutil.Process(pid=t)
            p.cpu_affinity([self.cores[core_index]])
            core_index = (core_index + 1) % len(self.cores) 
        
        self.coreLogger.warn("Update cores to - %s %s" % (len(self.cores) + 1, round(time.time()) - self.startingTime))

    def updateWays(self):
        self.wayLogger.warn("Update ways to - %s %s" % (self.appCacheWays, round(time.time()) - self.startingTime))
        os.system('sudo pqos -e "llc:1=%s;" > /dev/null' % self.formatForCAT(self.appCacheWays))

    def startPerfmon(self):
        self.sessions = [None] * len(self.tid)
        for i,id in enumerate(self.tid):
            self.sessions[i] = perfmon.PerThreadSession(int(id), EVENTS)
            self.sessions[i].start()

    def getPMC(self):
        pmc = [0] * len(EVENTS)
        for i in range(0,len(EVENTS)):
            for session in self.sessions:
                count = struct.unpack("L", session.read(i))[0]
                pmc[i] += float(count)
            pmc[i] /= len(self.tid)
        return pmc
    
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
        self.stateLogger.info("State is : %s %s" % (list(normalized), round(time.time()) - self.startingTime))
        normalized = np.append(normalized, [ len(self.cores)/24 , self.appCacheWays/20 ])

        return list(normalized)

    def formatForCAT(self, ways):
        res = 1 << ways - 1
        res = res + res - 1
        return hex(res)


    def getReward(self):
        global containerReward
        while(len(containerReward['reward']) == 0):
            time.sleep(0.01)
            self.rewardLogger.info("Waiting on reward " + str(round(time.time()) - self.startingTime))
        containerReward['lock'].acquire()
        sjrn99 = np.percentile(containerReward['reward'], 99)
        qos = round(sjrn99/1e3)
        containerReward['lock'].release()

        self.sjrnLogger.info("99th percentile is : " + str(qos) + " " + str(round(time.time()) - self.startingTime))
        qosTarget = 3000
        if qos > qosTarget:
            reward = max(-(qos/qosTarget)**3, -50)
        else:
            reward = qosTarget/qos + (20/self.appCacheWays) + 2*(24/len(self.cores))
        self.rewardLogger.info("Reward is : " + str(reward) + " " + str(round(time.time()) - self.startingTime))
        return reward

    def clearReward(self):
        global containerReward
        containerReward['lock'].acquire()
        containerReward['reward'] = []
        containerReward['lock'].release()

    def step(self, action):
        return [],0,0,{}
        pmc_before = self.getPMC()
        self.takeAction(action)
        self.clearReward()
        time.sleep(2)
        pmc_after = self.getPMC()
        state = self.getState(pmc_before, pmc_after)
        reward = self.getReward()
        return state, reward, 0, {}

    def startProcess(self):
        my_env = os.environ.copy()
        my_env["DATA_ROOT"] = "/home/akimon/inputs/tailbench.inputs"
        my_env["TBENCH_WARMUPREQS"] = '1'
        my_env["TBENCH_MAXREQS"] = '0'
        my_env["TBENCH_QPS"] = '500'
        my_env["TBENCH_MINSLEEPNS"] = '10000'
        my_env["TBENCH_MNIST_DIR"] = "/home/akimon/inputs/tailbench.inputs/img-dnn/mnist"
        my_env["TBENCH_QPS"] = '500'
        my_env["NTHREADS"] = '24'
        cores = str(self.cores)[1:-1].replace(' ', '')
        command = 'taskset -ac %s /home/akimon/tailbench_latest_latest/tailbenchQPS/img-dnn/run.sh' % (cores)
        command = shlex.split(command)
        process = subprocess.Popen(command, cwd="/home/akimon/tailbench_latest_latest/tailbenchQPS/img-dnn/", shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        time.sleep(2)
        self.process = psutil.Process(process.pid).children()[0]
        print(self.process)
        self.tid = list()
        for tid in self.process.threads():
            self.tid.append(tid.id)
        self.processStartTime = round(time.time())
        return process
    
    def containerLogger(self, containerReward):
        process = self.startProcess()
        self.startPerfmon()
        while True:
            output = process.stdout.readline()
            if output:
                try:
                    output = output.decode().strip()
                    if( output.startswith('RPS')  ):
                        rps = output.split(':')[1]
                        self.rpsLogger.warn("RPS - %s %s" % (rps, round(time.time()) - self.startingTime))
                        continue
                    if output.isdigit() and containerReward['lock'].acquire(blocking=False):
                        sjrn = float(output)
                        if sjrn < 20000000:
                            containerReward['reward'].append(sjrn)
                        containerReward['lock'].release()
                except ValueError:
                    containerReward['lock'].release()




    def reset(self):
        state = [0] * (len(EVENTS) + 2)
        return state

    def close(self):
        return

if __name__ == "__main__":
    dt = datetime.now().strftime("%m_%d_%H")
    Path("./models/%s" % dt).mkdir(parents=True, exist_ok=True)

    env = CustomEnv()

    # policy_kwargs = dict(act_fun=tf.nn.relu, layers=[512, 256, 128])
    policy_kwargs = dict(act_fun=tf.nn.relu, layers=[256, 128, 64])

    model = DQN("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1,
            train_freq=1,
            prioritized_replay=True,
            prioritized_replay_alpha=0.6, prioritized_replay_beta0=0.4, prioritized_replay_beta_iters=20000,
            double_q=True,
            learning_rate=0.0025, target_network_update_freq=150, learning_starts=750,
            batch_size=64, buffer_size=1000000,
            gamma=0.99, exploration_fraction=0.1, exploration_initial_eps=1, exploration_final_eps=0.01,
            tensorboard_log="./logs/%s/" % dt, n_cpu_tf_sess=22
            )
    model.learn(total_timesteps=20000)
    model.predict()
    model.save("./models/%s/model.zip" % dt)















