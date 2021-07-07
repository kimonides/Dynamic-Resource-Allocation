from datetime import datetime
from pathlib import Path
import logging
import shlex
import subprocess
import json
import time

def setupLoger(name, file):
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    fh = logging.FileHandler(file, mode='w')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)
    log.addHandler(fh)

    return log

def setupDataLoggers(appName):
    def setupLoger(name, file):
        log = logging.getLogger(name)
        log.setLevel(logging.DEBUG)
        fh = logging.FileHandler(file, mode='w')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(message)s')
        fh.setFormatter(formatter)
        log.addHandler(fh)

        return log
    loggers = []
    dt = datetime.now().strftime("%m_%d_%H")
    Path("./logs/%s_%s" % (dt,appName)).mkdir(parents=True, exist_ok=True)
    loggers.append(setupLoger('reward_%s' % appName, './logs/{0}_{1}/reward.log'.format(dt,appName)))
    loggers.append(setupLoger('ways_%s' % appName, './logs/{0}_{1}/ways.log'.format(dt,appName)))
    loggers.append(setupLoger('sjrn_%s' % appName, './logs/{0}_{1}/sjrn.log'.format(dt,appName)))
    loggers.append(setupLoger('state_%s' % appName, './logs/{0}_{1}/state.log'.format(dt,appName)))
    loggers.append(setupLoger('cores_%s' % appName, './logs/{0}_{1}/cores.log'.format(dt,appName)))
    loggers.append(setupLoger('rps_%s' % appName, './logs/{0}_{1}/rps.log'.format(dt,appName)))
    loggers.append(setupLoger('core_mapping_%s' % appName, './logs/{0}_{1}/core_mapping.log'.format(dt,appName)))

    return loggers

def containerLogger(containerReward, rpsLogger, startTime):
    import subprocess, os
    my_env = os.environ.copy()
    my_env["DATA_ROOT"] = "/home/akimon/inputs/tailbench.inputs"
    my_env["TBENCH_WARMUPREQS"] = '0'
    my_env["TBENCH_MAXREQS"] = '0'
    my_env["TBENCH_QPS"] = '500'
    my_env["TBENCH_MINSLEEPNS"] = '10000'
    my_env["TBENCH_MNIST_DIR"] = "/home/akimon/inputs/tailbench.inputs/img-dnn/mnist"
    my_env["TBENCH_QPS"] = '500'
    command = 'taskset -c 12-23,36-47 /home/akimon/tailbench_latest_latest/tailbenchQPS/img-dnn/run.sh'
    # command = 'taskset -c 12-23,36-47 /home/akimon/tailbench_latest_latest/tailbenchQPS/xapian/run.sh'
    # command = 'taskset -c 12-23,36-47 /home/akimon/tailbench_latest_latest/tailbenchQPS/masstree/run.sh'
    command = shlex.split(command)
    # process = subprocess.Popen(command, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process = subprocess.Popen(command, cwd="/home/akimon/tailbench_latest_latest/tailbenchQPS/img-dnn/", shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # process = subprocess.Popen(command, cwd="/home/akimon/tailbench_latest_latest/tailbenchQPS/sphinx/", shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # process = subprocess.Popen(command, cwd="/home/akimon/tailbench_latest_latest/tailbenchQPS/xapian/", shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # process = subprocess.Popen(command, cwd="/home/akimon/tailbench_latest_latest/tailbenchQPS/masstree/", shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # bullshitLogger = setupLoger('core_mapping', './logs/{0}/bullshit.log'.format(datetime.now().strftime("%m_%d_%H")))
    while True:
        output = process.stdout.readline()
        
        if process.poll() is not None:
            break
        if output:
            try:
                output = output.decode().strip()
                # print(output)
                if( output.startswith('RPS')  ):
                    rps = output.split(':')[1]
                    rpsLogger.warn("RPS - %s %s" % (rps, round(time.time()) - startTime))
                    continue
                if output.isdigit() and containerReward['lock'].acquire(blocking=False):
                    sjrn = float(output)
                    containerReward['reward'].append(sjrn)
                    containerReward['lock'].release()
            except ValueError:
                containerReward['lock'].release()
    rc = process.poll()



