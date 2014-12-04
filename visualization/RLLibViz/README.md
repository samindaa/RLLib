RLLibViz
========

RLLibViz is a [Qt5](http://qt-project.org/) based visualization tool to observe the behaviors of algorithms in RLLib library. RLLib provides a simple and modularize framework to realize hard-to-engineer behaviors in reinforcement learning (RL). We have realized and visualized the following on-policy and off-policy problems:

* On-policy: SwingPendulum with continuous actions. This problem uses AverageRewardActorCritic RL agent. 
* Off-policy: MountainCar and ContinuousGridworld with discrete actions. This problem uses OffPAC RL agent.

Configuration
-------------
	qmake RLLibViz.pro
	make 
	./RLLibViz

Operating Systems
-----------------

Ubuntu >= 11.04
Windows >= 7
Mac OS
