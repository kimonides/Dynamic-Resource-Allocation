#!/usr/bin/env python3
import perfmon
import subprocess
import struct
import shlex
import time

cores  = [ core for core in range(12,24) ]
cores += [ core for core in range(36,48) ]
# cores = [12]

EVENTS = ['UNHALTED_CORE_CYCLES', 'INSTRUCTION_RETIRED', 'PERF_COUNT_HW_CPU_CYCLES', 'UNHALTED_REFERENCE_CYCLES', \
        'UOPS_RETIRED', 'BRANCH_INSTRUCTIONS_RETIRED', 'MISPREDICTED_BRANCH_RETIRED', \
        'PERF_COUNT_HW_BRANCH_MISSES', 'LLC_MISSES', 'PERF_COUNT_HW_CACHE_L1D', \
        'PERF_COUNT_HW_CACHE_L1I']

max_counter = {
    #TODO: Run microbenchmark stress_cpu.c
    'UNHALTED_CORE_CYCLES' :     None,
    'INSTRUCTION_RETIRED' :      None,
    'PERF_COUNT_HW_CPU_CYCLES':  None,
    'UNHALTED_REFERENCE_CYCLES': None,
    'UOPS_RETIRED':              None,

    #TODO: Run microbenchmark branch_misses.cpp
    'BRANCH_INSTRUCTIONS_RETIRED': None,
    'MISPREDICTED_BRANCH_RETIRED': None,
    'PERF_COUNT_HW_BRANCH_MISSES': None,

    #TODO: Run microbenchmark stream
    'LLC_MISSES':              None,
    'PERF_COUNT_HW_CACHE_L1D': None,
    'PERF_COUNT_HW_CACHE_L1I': None,
    }


state = [0] * 11
import time

bench_indexes = {
    'stress_cpu' : [0,1,2,3,4],
    'branch_misses': [5,6,7],
    'stream_c.exe': [8,9,10]
}

bench_times = {
    'stress_cpu' : 0,
    'branch_misses': 0,
    'stream_c.exe': 0
}

# for benchmark in ['stress_cpu','branch_misses','stream_c.exe']:
for benchmark in ['stress_cpu','branch_misses','stream_c.exe']:
    print("Running %s" % benchmark)
    if benchmark == 'stream_c.exe':
        command = shlex.split('sudo taskset -c %s ./STREAM/%s' % (str(cores)[1:-1].replace(' ', ''),benchmark))
    else:
        command = shlex.split('sudo taskset -c %s ./%s' % (str(cores)[1:-1].replace(' ', ''),benchmark))
    proc = subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    import psutil
    time.sleep(0.1)
    pid = 0
    for p in psutil.process_iter():
        if benchmark in p.name():
            pid = p.pid
            print(pid)
    if pid == 0 :
        print("Couldn't find app pid, exiting...")
        exit(-1)
    session = perfmon.PerThreadSession(pid,EVENTS)
    session.start()
    prevState = [0] * 11
    start = time.perf_counter()
    for i in bench_indexes[benchmark]:
        count = struct.unpack("L", session.read(i))[0]
        prevState[i] = int(count)

    while proc.poll() is None:
        time.sleep(0.01)
        for i in bench_indexes[benchmark]:
            count = struct.unpack("L", session.read(i))[0]
            state[i] += int(count) - prevState[i]
            prevState[i] = count
    end = time.perf_counter()
    bench_times[benchmark] = end-start

    
for benchmark in ['stress_cpu','branch_misses','stream_c.exe']:
    for i in bench_indexes[benchmark]:
        state[i] /= bench_times[benchmark]

print(state)






















# state = [0] * 11
# c = 0
# import time

# start = time.perf_counter()
# # for benchmark in ['stress_cpu','branch_misses','stream_c.exe']:
# for benchmark in ['stress_cpu','branch_misses','stream_c.exe']:
#     print("Running %s" % benchmark)
#     if benchmark == 'stream_c.exe':
#         command = shlex.split('sudo taskset -c %s ./STREAM/%s' % (str(cores)[1:-1].replace(' ', ''),benchmark))
#     else:
#         command = shlex.split('sudo taskset -c %s ./%s' % (str(cores)[1:-1].replace(' ', ''),benchmark))
#     proc = subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
#     import psutil
#     time.sleep(3)
#     pid = 0
#     for p in psutil.process_iter():
#         if benchmark in p.name():
#             pid = p.pid
#             print(pid)
#     if pid == 0 :
#         print("Couldn't find app pid, exiting...")
#         exit(-1)
#     session = perfmon.PerThreadSession(pid,EVENTS)
#     session.start()
#     prevState = [0] * 11
#     for j in range(0,len(EVENTS)):
#         count = struct.unpack("L", session.read(j))[0]
#         prevState[j] = float(count)

#     while proc.poll() is None:
#         time.sleep(1)
#         for i in range(0,len(EVENTS)):
#             count = struct.unpack("L", session.read(i))[0]
#             # states[i] = max(int(count) - prevState[i], states[i] )
#             state[i] += int(count) - prevState[i]
#             prevState[i] = count
#             c += 1


    
# for i in range(EVENTS):
#     state[i] /= state[i]/c

# print(state[i])

    
