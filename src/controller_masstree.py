#!/usr/bin/env python3
from datetime import datetime
from pathlib import Path
import subprocess
import shlex
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
import psutil
from collections import deque

import os


# config = configparser.ConfigParser()
# config.read('config.ini')

containerReward = {
    'reward': [],
    'lock': threading.Lock()
}

processReady = threading.Lock()

window_size = 5

qosTarget = 3000

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
    'CACHE': deque([], maxlen=window_size),
    'CORES': deque([], maxlen=window_size),
}



rewardLogger, wayLogger, sjrnLogger, stateLogger, coreLogger, rpsLogger, coreMapLogger = loggers.setupDataLoggers()


EVENTS = ['UNHALTED_CORE_CYCLES', 'INSTRUCTION_RETIRED', 'PERF_COUNT_HW_CPU_CYCLES', 'UNHALTED_REFERENCE_CYCLES', \
        'UOPS_RETIRED', 'BRANCH_INSTRUCTIONS_RETIRED', 'MISPREDICTED_BRANCH_RETIRED', \
        'PERF_COUNT_HW_BRANCH_MISSES', 'LLC_MISSES', 'PERF_COUNT_HW_CACHE_L1D', \
        'PERF_COUNT_HW_CACHE_L1I']
EVENT_MAX = [1251666326, 2697635738, 1502160478, \
            1385062673, 3899393008, 265396012, \
            42954597, 42960949, 1598918, 
            14667253, 30645, 1, 1]

EVENT_MAX = [e*2 for e in EVENT_MAX]


class CustomEnv(gym.Env):
    def __init__(self,):
        super(CustomEnv, self).__init__()
        global deques, window_size
        self.deques = deques
        self.window_size = window_size

        self.startingTime = round(time.time())
        self.process = None

        self.cores = [core for core in range(12, 24)]
        self.cores += [core for core in range(36, 48)]

        self.allCores = [core for core in range(12, 24)]
        self.allCores += [core for core in range(36, 48)]

        self.appCacheWays = 20

        threading.Thread(target=self.containerLogger, args=(containerReward,), daemon=True).start()

        time.sleep(3)
        if self.process == None:
            print("Couldn't find app pid, exiting...")
            exit(-1)
        print(self.process)
        
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(low=0, high=1.5, shape=(13,), dtype=np.float64)

        os.system('pqos -R > /dev/null')
        os.system('pqos -e "llc:0=0x30000;" > /dev/null')

        self.updateWays()

        # self.updateCPUs()
        self.initialMapping()

        cores = str(self.cores)[1:-1].replace(' ', '')
        os.system('pqos -a "llc:1=%s;" > /dev/null' % cores)

        # self.startPerfmon()

        self.previousTime = time.time()
    
    def initialMapping(self):
        for pid, core in zip(self.tid, self.allCores):
            p = psutil.Process(pid=pid)
            p.cpu_affinity([core])
    
    def mapCores(self,action):
        # action is +-1
        if action == 1:
            unusedCores = [core for core in self.allCores if core not in self.cores]
            for core in unusedCores:
                if core % 2 == 0:
                    self.cores.append(core)
                    break
            else:
                newCore = unusedCores[0]
                self.cores.append(newCore)
            cores = str(self.cores)[1:-1].replace(' ', '')
            os.system('pqos -a "llc:1=%s;" > /dev/null' % cores)
        elif action == -1:
            for core in self.cores[::-1]:
                if core % 2 == 1:
                    self.cores.remove(core)
                    os.system('pqos -a "llc:0=%s;" > /dev/null' % core)
                    break
            else:
                core = self.cores.pop()
                os.system('pqos -a "llc:0=%s;" > /dev/null' % core)
        coreMapLogger.warn(str(self.cores))

        thread_index = 0
        for core in self.cores:
            pid = self.tid[thread_index]
            thread_index += 1
            p = psutil.Process(pid=pid)
            p.cpu_affinity([core])
        
        cores_reversed = self.cores[::-1]
        for i in range(thread_index,len(self.tid)):
            core = self.cores[ ( i - thread_index ) % len(cores_reversed) ]
            pid = self.tid[i]
            p = psutil.Process(pid=pid)
            p.cpu_affinity([core])

    def startPerfmon(self):
        self.sessions = [None] * len(self.tid)
        print(self.tid)
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

    def updateCPUs(self, core=None):
        cores = str(self.cores)[1:-1].replace(' ', '')
        os.system('taskset -apc %s %s > /dev/null' % (cores,self.process.pid))
        os.system('pqos -a "llc:1=%s;" > /dev/null' % cores)
        if(core != None):
            os.system('pqos -a "llc:0=%s;" > /dev/null' % core)

    def formatForCAT(self, ways):
        res = 1 << ways - 1
        res = res + res - 1
        return hex(res)

    def updateWays(self):
        os.system('sudo pqos -e "llc:1=%s;" > /dev/null' % self.formatForCAT(self.appCacheWays))

    def takeAction(self, action):
        if(action == 0):
            if(self.appCacheWays < 20):
                self.appCacheWays += 1
                wayLogger.warn("Increasing ways to - %s %s" % (self.appCacheWays, round(time.time()) - self.startingTime))
                self.updateWays()
            else:
                wayLogger.warn("Ignore - %s %s" % (self.appCacheWays, round(time.time()) - self.startingTime))
                return -1
        elif(action == 1):
            if(self.appCacheWays > 3):
                self.appCacheWays -= 1
                wayLogger.warn("Decreasing ways to - %s %s" % (self.appCacheWays, round(time.time()) - self.startingTime))
                self.updateWays()
            else:
                wayLogger.warn("Ignore - %s %s" % (self.appCacheWays, round(time.time()) - self.startingTime))
                return -1
        elif(action == 2):
            if(len(self.cores) < 24):
                coreLogger.warn("Increasing cores to - %s %s" % (len(self.cores) + 1, round(time.time()) - self.startingTime))
                self.mapCores(1)
            else:
                coreLogger.warn("Ignore - %s %s" % (len(self.cores), round(time.time()) - self.startingTime))
                return -1
        elif(action == 3):
            if(len(self.cores) > 3):
                coreLogger.warn("Decreasing cores to - %s %s" % (len(self.cores) - 1, round(time.time()) - self.startingTime))
                self.mapCores(-1)
            else:
                coreLogger.warn("Ignore - %s %s" % (len(self.cores), round(time.time()) - self.startingTime))
                return -1
        else:
            wayLogger.warn("Maintaining - %s %s" % (self.appCacheWays, round(time.time()) - self.startingTime))
            coreLogger.warn("Maintaining - %s %s" % (len(self.cores), round(time.time()) - self.startingTime))
        return 0

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
        state_space.append(cur_counter[-2])
        state_space.append(cur_counter[-1])
        deques['CACHE'].append( cur_counter[-2] )
        deques['CORES'].append( cur_counter[-1] )

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
        state.append( self.appCacheWays/20 )
        state.append( len(self.cores)/24 )
        normalized = self.norm_data(state)
        stateLogger.info("State is : %s %s" % (list(normalized), round(time.time()) - self.startingTime))
        # normalized = np.append(normalized, [ len(self.cores)/24 , self.appCacheWays/20 ])

        return list(normalized)

    def getReward(self, ignoreAction = 0):
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
        qosTarget = 2000
        if qos > qosTarget:
            reward = max(-(qos/qosTarget)**3, -50)
        else:
            reward = qosTarget/qos + (20/self.appCacheWays) + (24/len(self.cores))
        if ignoreAction != 0:
            reward = -10
        rewardLogger.info("Reward is : " + str(reward) + " " + str(round(time.time()) - self.startingTime))
        return reward

    def clearReward(self):
        global containerReward
        containerReward['lock'].acquire()
        containerReward['reward'] = []
        containerReward['lock'].release()

    def step(self, action):
        while(not processReady.acquire(blocking=False)):
            time.sleep(1)
            print('Waiting on process to be ready')
        pmc_before = self.getPMC()
        ignored_action = self.takeAction(action)
        self.clearReward()
        time.sleep(2)
        pmc_after = self.getPMC()
        state = self.getState(pmc_before, pmc_after)
        reward = self.getReward(ignored_action)
        processReady.release()
        return state, reward, 0, {}

    def reset(self):
        state = [0] * (len(EVENTS) + 2)
        return state

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
        command = 'taskset -ac %s /home/akimon/tailbench_latest_latest/tailbenchQPS/masstree/run.sh' % (cores)
        command = shlex.split(command)
        process = subprocess.Popen(command, cwd="/home/akimon/tailbench_latest_latest/tailbenchQPS/masstree/", shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        time.sleep(2)
        self.process = psutil.Process(process.pid).children()[0]
        print(self.process)
        self.tid = list()
        for tid in self.process.threads():
            self.tid.append(tid.id)
        self.initialMapping()
        self.processStartTime = round(time.time())
        return process
    
    def containerLogger(self, containerReward):
        process = self.startProcess()
        self.startPerfmon()
        while True:
            output = process.stdout.readline()
            if ( round(time.time()) - self.processStartTime ) > 4000:
                print('Killing App')
                processReady.acquire()
                process.kill()
                time.sleep(0.5)
                print('Clearing app states')
                for key in self.deques:
                    self.deques[key].clear()
                print('Clearing app reward')
                containerReward['lock'].acquire()
                containerReward['reward'] = []
                containerReward['lock'].release()
                print('Starting app')
                process = self.startProcess()
                self.startPerfmon()
                processReady.release()
                continue
            if output:
                try:
                    output = output.decode().strip()
                    if( output.startswith('RPS')  ):
                        rps = output.split(':')[1]
                        rpsLogger.warn("RPS - %s %s" % (rps, round(time.time()) - self.startingTime))
                        continue
                    if output.isdigit() and containerReward['lock'].acquire(blocking=False):
                        sjrn = float(output)
                        # print(sjrn)
                        if sjrn < 20000000:
                            containerReward['reward'].append(sjrn)
                        containerReward['lock'].release()
                except ValueError:
                    containerReward['lock'].release()

    def close(self):
        return


dt = datetime.now().strftime("%m_%d_%H")
Path("./models/%s" % dt).mkdir(parents=True, exist_ok=True)

env = CustomEnv()

policy_kwargs = dict(act_fun=tf.nn.relu, layers=[512, 256, 128])

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

if __name__ == "__main__":
    model.learn(total_timesteps=20000)
    model.save("./models/%s/model.zip" % dt)















