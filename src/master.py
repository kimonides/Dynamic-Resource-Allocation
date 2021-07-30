import controller_img_dnn
import random
import controller_masstree

from stable_baselines import DQN
import time
import os
import psutil
import shlex
import subprocess

def startSphinxBatch(cores):
    my_env = os.environ.copy()
    my_env["DATA_ROOT"] = "/home/akimon/inputs/tailbench.inputs"
    my_env["AUDIO_SAMPLES"] = "audio_samples"
    my_env["TBENCH_WARMUPREQS"] = '1'
    my_env["TBENCH_MAXREQS"] = '0'
    my_env["TBENCH_QPS"] = '2'
    my_env["TBENCH_MINSLEEPNS"] = '1000'
    my_env["TBENCH_QPS"] = '500'
    my_env["NTHREADS"] = '24'
    cores = str(cores)[1:-1].replace(' ', '')
    command = 'taskset -ac %s /home/akimon/tailbench_latest_latest/tailbenchQPS/sphinx/run.sh' % (cores)
    command = shlex.split(command)
    process = subprocess.Popen(command, cwd="/home/akimon/tailbench_latest_latest/tailbenchQPS/sphinx/", shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    time.sleep(2)
    process = psutil.Process(process.pid).children()[0]
    return process

def startMasstreeBatch(cores):
        my_env = os.environ.copy()
        my_env["DATA_ROOT"] = "/home/akimon/inputs/tailbench.inputs"
        my_env["TBENCH_WARMUPREQS"] = '1'
        my_env["TBENCH_MAXREQS"] = '0'
        my_env["TBENCH_QPS"] = '1'
        my_env["TBENCH_MINSLEEPNS"] = '10000'
        my_env["TBENCH_QPS"] = '500'
        my_env["NTHREADS"] = '24'
        cores = str(cores)[1:-1].replace(' ', '')
        command = 'taskset -ac %s /home/akimon/tailbench_latest_latest/tailbenchQPS/masstree/run.sh' % (cores)
        command = shlex.split(command)
        process = subprocess.Popen(command, cwd="/home/akimon/tailbench_latest_latest/tailbenchQPS/masstree/", shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        time.sleep(2)
        process = psutil.Process(process.pid).children()[0]
        return process

def updateBatchCores(ps, cores):
    ps.cpu_affinity( cores )


if __name__ == "__main__":
    allCores = [core for core in range(12, 24)]
    allCores += [core for core in range(36, 48)]

    def updateWays(main_env, other_env, action, reverse=False):
        if 2 <= main_env.appCacheWays + action <= 20:
            # main_env.appCacheWays += action
            main_env.updateWays(reverse)

    def updateCores(main_env, other_env, action, reverse=False):
        unused_cores = [x for x in allCores if x not in main_env.cores and x not in other_env.cores]
        if action == 1:
            if len(main_env.cores) + len(other_env.cores) == len(allCores):
                return
            core = unused_cores[-1] if reverse else unused_cores[0]
            # core = random.choice(unused_cores)
            unused_cores.remove(core)
            main_env.cores.append(core)
            main_env.updateCores(unused_cores)
        elif action == -1:
            main_env.cores.pop()
            main_env.updateCores(unused_cores)

    def actionMapper(action, main_env, other_env, reverse=False):
        if action == 0:
            updateWays(main_env, other_env, 1, reverse)
        elif action == 1:
            updateWays(main_env, other_env, -1, reverse)
        elif action == 2:
            updateCores(main_env, other_env, 1, reverse)
        elif action == 3:
            updateCores(main_env, other_env, -1, reverse)

    cores1 = [core for core in range(12,18)]
    cores2 = [core for core in range(42, 48)]

    cacheWays1 = 8
    cacheWays2 = 8


    batch_cores = [x for x in allCores if x not in cores1 and x not in cores2]
    
    env1 = controller_img_dnn.CustomEnv( cores1 , cacheWays1, batch_cores)
    env2 = controller_masstree.CustomEnv( cores2 , cacheWays2, batch_cores)
    
    batch_processes = []
    # batch_processes.append(startMasstreeBatch(batch_cores))
    # batch_processes.append(startSphinxBatch(batch_cores))

    model1 = DQN.load(load_path='/home/akimon/Dynamic-Resource-Allocation/src/models/07_29_07_img_dnn_best/model.zip/rl_model_12500_steps.zip')
    model2 = DQN.load(load_path='./models/07_03_06_masstree/model.zip')

    time.sleep(2)

    while True:

        env1.getReward()
        env2.getReward()

        pmc_before1 = env1.getPMC()
        pmc_before2 = env2.getPMC()
        time.sleep(2)
        pmc_after1 = env1.getPMC()
        pmc_after2 = env2.getPMC()


        state1 = env1.getState(pmc_before1, pmc_after1)
        state2 = env2.getState(pmc_before2, pmc_after2)

        act1 , _ = model1.predict(state1, deterministic=True)
        act2 , _ = model2.predict(state2, deterministic=True)
    
        actionMapper(act1, env1, env2)
        actionMapper(act2, env2, env1, reverse=True)

        batch_cores = [x for x in allCores if x not in env1.cores and x not in env2.cores]

        for p in batch_processes:
            updateBatchCores(p, batch_cores)