# sudo python3 master.py &



# for benchmark in sphinx leslie3d astar

#!/bin/bash
array[0]="sphinx"
array[1]="leslie3d"
array[2]="astar"
array[3]="gcc"
array[4]="libquantum"
array[5]="soplex"
# array=["sphinx","leslie3d","astar","gcc","libquantum","soplex"]

size=${#array[@]}
# index=$(($RANDOM % $size))
# echo ${array[$index]}

function start_benchmark {
    index=$(($RANDOM % $size))
    echo ${array[$index]} 

    docker run --rm --cpus=1 --cpuset-cpus=$core --name "$benchmark"_"$ways"_ways -e BENCHMARK=$benchmark pl4tinum/spec2006
}

for i in {1..15}
do
   start_benchmark &
done

start_benchmark
