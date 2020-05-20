# Welcome to HYDRA framework repository

HyDRA - Resilient computing for Heterogeneous Autonomous Devices - is a Python3
middleware architecture enabling the dynamic migration within interconnected systems of modules composing pipelines for autonomy.
The system is hierarchical and composed of two main entities: clients and servers.

Our system was designed, but is not limited to, enhance capabilities of Unmanned Aerial Vehicles (commonly drones) through computational offloading.
We recently developed the opportunity for server to be flying as well, as we will show in future publications.

The system you will find herei s based on the producer-consumer paradigm, where in a single device, different threads will read from an input buffer, and write on to an output buffer.

In this way the system is highly customizable on the specific application, and can exploit the high flexibility offered by HYDRA, to change executuion plan on the fly based on a higher level controller.
