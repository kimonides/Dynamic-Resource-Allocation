from datetime import datetime
from pathlib import Path
import logging
import shlex
import subprocess
import json
import dictionaryUtils
import time

def setupDataLoggers():
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
    Path("./logs/%s" % dt).mkdir(parents=True, exist_ok=True)
    loggers.append(setupLoger('reward', './logs/{0}/reward.log'.format(dt)))
    loggers.append(setupLoger('ways', './logs/{0}/ways.log'.format(dt)))
    loggers.append(setupLoger('sjrn', './logs/{0}/sjrn.log'.format(dt)))
    loggers.append(setupLoger('state', './logs/{0}/state.log'.format(dt)))
    loggers.append(setupLoger('cores', './logs/{0}/cores.log'.format(dt)))
    loggers.append(setupLoger('rps', './logs/{0}/rps.log'.format(dt)))

    return loggers

def containerLogger(containerReward, rpsLogger, startTime):
# def containerLogger(containerReward, rpsLogger, startTime):
    # sjrn , qtime, servieTime
    command = shlex.split("docker exec -t -w /tailbenchQPS/sphinx tb ./run.sh")
    process = subprocess.Popen(command, shell=False, stdout=subprocess.PIPE)

    while True:
        output = process.stdout.readline()
        if process.poll() is not None:
            break
        if output:
            try:
                if( output.strip().decode().startswith('RPS')  ):
                    rps = output.strip().decode('utf-8').split(':')[1]
                    rpsLogger.warn("RPS - %s %s" % (rps, round(time.time()) - startTime))
                    continue
                # if containerReward['lock'].acquire(blocking=False) and containerState['lock'].acquire(blocking=False):
                if containerReward['lock'].acquire(blocking=False):
                    sjrn, qtime, svc = [float(v) for v in output.strip().decode('utf-8').split(',')]
                    containerReward['reward'].append(sjrn)
                    # containerReward['reward'] += sjrn
                    # containerReward['count'] += 1
                    containerReward['lock'].release()
                    # containerState['state'][0] += svc
                    # containerState['state'][1] += qtime
                    # containerState['count'] += 1
                    # containerState['lock'].release()
            except ValueError:
                containerReward['lock'].release()
                # containerState['lock'].release()
    rc = process.poll()

def pcmLogger(pcmState):
    command = shlex.split("./start_pcm.sh")
    process = subprocess.Popen(command, shell=False, stdout=subprocess.PIPE)

    while True:
        output = process.stdout.readline()
        if process.poll() is not None:
            break
        if output:
            if pcmState['lock'].acquire(blocking=False):
                currentState = json.loads(output.decode())
                pcmState['state'] = dictionaryUtils.addDictionaries(pcmState['state'],currentState)
                pcmState['count'] += 1
                pcmState['lock'].release()
    rc = process.poll()