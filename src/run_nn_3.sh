#!/bin/bash
function exit_job()
{
    pqos -R > /dev/null
    # # Turn Intel Turbo boost back on
    # echo "0" > /sys/devices/system/cpu/intel_pstate/no_turbo
    printf "\nExitting\n"
}
trap exit_job SIGINT

# sudo modprobe msr
#Turn Intel Turbo boost off
# echo "1" > /sys/devices/system/cpu/intel_pstate/no_turbo 

pqos -R > /dev/null
pqos -e "llc:0=0x30000;" > /dev/null
pqos -a "llc:1=12;" > /dev/null
pqos -e "llc:1=0x3f" > /dev/null


taskset -c 0 /home/akimon/drl_pcm/pcm.x -r --kimo 0.25
