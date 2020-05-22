# Target tracking
In this experiment we use one drone and 3 stationary edges. The client-drone, using a camera mounted onboard, will track
and follow an object moving around it. 

The task is to follow an object offloading the frames captured by the camera in real-time. The client-drone is going to 
use an algorithm to decide to which edges it should offload the tasks (images). The choosing algorithm is predicting how 
quickly each edge can respond, and uses such prediction to activate the minimum number of pipelines in order to guarantee
time below a given threshold.