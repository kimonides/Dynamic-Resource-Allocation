import gym
import tensorflow as tf
import controller_img_dnn
import random
import controller_masstree

from stable_baselines import DQN
import time
import os


if __name__ == "__main__":

    allCores = [core for core in range(12, 24)]
    allCores += [core for core in range(36, 48)]

    def updateWays(main_env, other_env, action):
        if 2 <= main_env.appCacheWays + action <= 20:
            main_env.appCacheWays += action
            main_env.updateWays()
            # ways = main_env.appCacheWays
            # res = 1 << ways - 1
            # res = res + res - 1
            # os.system('sudo pqos -e "llc:1=%s;" > /dev/null' % hex(res))

    def updateCores(main_env, other_env, action):
        if action == 1:
            core = -1
            if len(main_env.cores) + len(other_env.cores) == len(allCores):
                return
            while True:
                core = random.choice(allCores)
                if core not in main_env.cores and core not in other_env.cores:
                    break
            main_env.cores.append(core)
            main_env.updateCores()
        elif action == -1:
            main_env.cores.pop()
            main_env.updateCores()

    def actionMapper(action, main_env, other_env):
        if action == 0:
            updateWays(main_env, other_env, 1)
        elif action == 1:
            updateWays(main_env, other_env, -1)
        elif action == 2:
            updateCores(main_env, other_env, 1)
        elif action == 3:
            updateCores(main_env, other_env, -1)

    cores1 = [12,13,14,15,16,17]
    cacheWays1 = 8
    env1 = controller_img_dnn.CustomEnv( cores1 , cacheWays1)

    cores2 = [18,19,20,21,23,24]
    cacheWays2 = 8
    env2 = controller_masstree.CustomEnv( cores2 , cacheWays2)

    model1 = DQN.load(load_path='./models/07_06_09_img_dnn/model.zip')
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

        act1 , _ = model1.predict(state1)
        act2 , _ = model2.predict(state2, deterministic=True)
        
        # print("Action for 1 is %d and for 2 is %d" % (act1,act2))
        # print("Action for 1 is %d and state is %s %s" % ( act1, env1.cores, env1.appCacheWays ))
    
        actionMapper(act1, env1, env2)
        actionMapper(act2, env2, env1)