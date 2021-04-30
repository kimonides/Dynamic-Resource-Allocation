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

### Installing

* No installation needed, just edit /src/config.ini to add the path to the PCM pcm.x executable and the name of the container that runs the application

### Executing program

```
sudo python3 controller.py
```
## License

This project is licensed under the MIT License - see the LICENSE file for details

## Acknowledgments

Inspiration, code snippets, etc.
* [awesome-readme](https://github.com/matiassingers/awesome-readme)
* [PurpleBooth](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
* [dbader](https://github.com/dbader/readme-template)
* [zenorocha](https://gist.github.com/zenorocha/4526327)
* [fvcproductions](https://gist.github.com/fvcproductions/1bfc2d4aecb01a834b46)
