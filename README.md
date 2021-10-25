# Dynamic-Resource-Allocation

A controller that does dynamic cpu and cache way allocation for latency critical applications.

## Description

The controller leverages deep reinforcement learning to achieve the best possible resource utilization by making sure that the latency critical application gets all the resources it needs to achieve a good response time while giving any unneeded resources to batch workloads.

## Getting Started

### Dependencies
* Python3
* Stable Baselines 3
* [Custom PCM for this project](https://github.com/kimonides/drl_pcm)
* [Custom Tailbench for testing purposes](https://github.com/kimonides/drl_tailbench)
* Numpy

### Executing program

```
sudo python3 controller.py
```
## License

This project is licensed under the MIT License - see the LICENSE file for details
