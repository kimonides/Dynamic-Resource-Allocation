#!/bin/bash

# sudo python3 controller.py
sudo taskset -c 0-11,24-35 python3 controller.py 

