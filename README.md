RLLib (Lightweight Standard and On/Off Policy Reinforcement Learning Library (C++))
=====

The RLLib is an implementation of incremental standard and gradient temporal-difference learning (GTDL) algorithms  for robotics applications using C++ programing language. The implementation of this highly optimized and lightweight library is inspired by the API of RLPark, which is a library of temporal-difference learning algorithms implemented in Java. 

RLLib features:

    Off-policy prediction algorithms: GTD(lambda), and GQ(lambda),
    Off-policy control algorithms:  Greedy-GQ(lambda), Softmax-GQ(lambda), and Off-PAC,
    On-policy algorithms: TD(lambda), SARSA(lambda), Expected-SARSA(lambda), and Actor-Critic (continuous and discrete actions), 
    Policies: Random, Random50%Bias, Greedy, Epsilon-greedy, Boltzmann, Normal, and Softmax,
    Efficient dot product implementation for tile coding base feature representations (with culling traces),
    Benchmarks: Mountain Car, Mountain Car 3D, Swinging Pendulum, and Continuous grid world (Off-PAC paper) environments,
    Optimized for very fast duty cycles (e.g., with culling traces, tested on the Robocup 3D simulator and on the Nao V4  (cognition thread)), 
    Main algorithm usage is very much similar to RLPark, therefore, swift learning curve, and
    A plethora of examples demonstrating on-policy control experiments and off-policy control experiments.

Documentation: http://web.cs.miami.edu/home/saminda/rllib.html

Saminda Abeyruwan (saminda@cs.miami.edu)

