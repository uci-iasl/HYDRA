# Welcome to HYDRA framework repository

HyDRA - Resilient computing for Heterogeneous Autonomous Devices - is a Python3 middleware architecture enabling the dynamic migration of computationally intense tasks from embedded devices to edge servers.

Our system was designed, but is not limited to, enhance capabilities of Unmanned Aerial Vehicles (commonly drones) through computational offloading. A drone-client will probe to find nearby edge servers that offer their resources.

## Index
[Why another framework](#why-another-framework): simply we could not find what we need!

[How does it work](#how-does-it-work): explains the main ideas behind the framework.

[Where should I start](#where-should-i-start): run your first script!

[Contacts](#contact-me): let me know what you think about HyDRA!

### Why another framework
We developed because we couldn't find an easy way to code data analysis pipelines, that would give us the capability to easily change the analysis functions based on the context. Things change, and to make the best decisions you need awareness: this is why monitoring has a huge role in our framework. Find more in [devices](./devices/README.md) 

![Random movement gif](./resources/drone.gif)

### How does it work?
The system is composed by a series of pipelines, where each **pipeline** is a cascade of operations transforming data into control. Each of these operations is encapsulated into a **module**, composed by an input queue, where the input comes in, a function executed on each piece of incoming information, and a series of output queues, where the data is communicated to the next modules. Note that multiple output queues allow data replication, used to enhance reliability and improve performance overall.

The system's internal communication is built using the producer-consumer paradigm, allowing maximum flexibility and safe interaction between concurrent threads.

In this way the system is highly customizable on the specific application, and can exploit the high flexibility offered by HYDRA, to change execution plan on the fly based on a higher level controller.

This level controller is called Device, and hosts a running loop that updates the information flow over time. When created it will start running the necessary modules, and over time manage the output queues of different modules based on the chosen scheduling logic.

Even though the applications of this framework are many, we start offering a motivating example. We started from a very natural challenge to evalaute the reactivity of our system: tracking and following.
The code you find here defines a set of modules that go from data acquisition, to data analysis and control of real drones. Our interface communicates with any camera compatible with OpenCV, and any vehicle (drones, but also rovers) mouting a PixHawk.

### Where should I start?
Find in [devices](./devices/README.md) scripts to:
* define some commonly used modules
* find the data types natively supported, and good examples on how to extend them
* skeleton of a Device class
* real and abstract drones
* loggers to read your Operative System information
* networking code to establish, maintain, use a TCP connection

Check [track and follow](examples/track_and_follow/README.md) example, to see how to use some of these functionalities.

### How to cite
We published a series of papers using this framework as a base! Find the paper, with some related articles, at:  
[Dynamic Distributed Computing for Infrastructure-Assisted Autonomous UAVs](https://iasl.ics.uci.edu/people/dcallega/#callegaro-baidya-2020-icc)
> @inproceedings{callegaro-baidya-dynamic-distributed-2020-icc,  
>  author = {Davide Callegaro and Sabur Baidya and Marco Levorato},  
>  title = {Dynamic Distributed Computing for Infrastructure-Assisted Autonomous {UAVs}},  
>  booktitle = {Proceedings of the IEEE International Conference on Communications (ICC)},  
>  year = {2020},  
> }  

### Contact me
Send an email to [Davide](mailto:dcallega@uci.edu?subject=[GitHubHydra])!
