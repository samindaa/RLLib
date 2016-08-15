RLLib 
=====
(C++ Template Library to Predict, Control,  Learn Behaviors, and Represent Learnable Knowledge using On/Off Policy Reinforcement Learning)
----------------------------------------------------------------------------------------------------------------------

RLLib is a lightweight C++ template library that implements `incremental`, `standard`, and `gradient temporal-difference` learning algorithms in Reinforcement Learning. It is an optimized library for robotic applications and embedded devices that operates under fast duty cycles (e.g., < 30 ms). RLLib has been tested and evaluated on RoboCup 3D soccer simulation agents,  physical NAO V4 humanoid robots, and Tiva C series launchpad microcontrollers  to predict, control, learn behaviors, and represent learnable knowledge.  The implementation of the RLLib library is inspired by the RLPark API, which is a library of temporal-difference learning algorithms written in Java.

Features
--------

* **Off-policy prediction algorithms**: 
 * `GTD(λ)`
 * `GTD(λ)True`
 * `GQ(λ)`
* **Off-policy control algorithms**:  
 * `Q(λ)`
 * `Greedy-GQ(λ)`
 * `Softmax-GQ(λ)`
 * `Off-PAC (can be used in on-policy setting)`
* **On-policy algorithms**: 
 * `TD(λ)`
 * `TD(λ)AlphaBound`
 * `TD(λ)True` 
 * `Sarsa(λ)`
 * `Sarsa(λ)AlphaBound`
 * `Sarsa(λ)True`
 * `Sarsa(λ)Expected`
 * `Actor-Critic (continuous actions, discrete actions, discounted reward settting, averaged reward settings, and so on)` 
* **Supervised learning algorithms**: 
 * `Adaline`
 * `IDBD`
 * `K1`
 * `SemiLinearIDBD`
 * `Autostep`
* **Policies**: 
 `Random`
 `RandomX%Bias`
 `Greedy`
 `Epsilon-greedy`
 `Boltzmann`
 `Normal`
 `Softmax`
* **Dot product**: 
 An efficient implementation of the dot product for tile coding based feature representations (with culling traces).
* **Benchmarking environments**: 
 `Mountain Car`
 `Mountain Car 3D`
 `Swinging Pendulum`
 `Continuous Grid World`
 `Bicycle`
 `Cart Pole`
 `Acrobot`
 `Non-Markov Pole Balancing`
 `Helicopter`
* **Optimization**: 
 Optimized for very fast duty cycles (e.g., with culling traces, RLLib has been tested on `the Robocup 3D simulator agent`, and on `the NAO V4  (cognition thread)`). 
* **Usage**: 
 The algorithm usage is very much similar to RLPark, therefore, swift learning curve.
* **Examples**: 
 There are a plethora of examples demonstrating on-policy and off-policy control experiments.
* **Visualization**:
 We provide a Qt4 based application to visualize benchmark problems.  

Build
----

[![Build Status](https://travis-ci.org/samindaa/RLLib.svg?branch=master)](https://travis-ci.org/samindaa/RLLib)
 
New: OpenAI Gym Binding
----

[Open AI Gym](https://gym.openai.com) is a toolkit for developing and comparing reinforcement learning algorithms. We have developed a bridge between Gym and RLLib to use all the functionalities provided by Gym, while writing the agents (on/off-policy) in RLLib. The directory, [openai_gym](openai_gym/README.md), contains our bridge as well as RLLib agents that learn and control the [classic control environments](https://gym.openai.com/envs#classic_control).
 

Extension
-----

* Extension for Tiva C Series EK-TM4C123GXL LaunchPad, and Tiva C Series TM4C129 Connected LaunchPad microcontrollers.

* Tiva C series launchpad microcontrollers: [https://github.com/samindaa/csc688](https://github.com/samindaa/csc688)

Demo
----

[![Off-PAC ContinuousGridworld](http://i1.ytimg.com/vi/SpBbdvhx4tM/3.jpg?time=1382317024739)](http://www.youtube.com/watch?v=9THBj9nX5gU)
[![AverageRewardActorCritic SwingPendulum (Continuous Actions)](http://i1.ytimg.com/vi/nwxAG2WXl3Y/3.jpg?time=1382317239212)](http://www.youtube.com/watch?v=ktNYS-ApAko)

Usage
-----

RLLib is a C++ template library. The header files are located in the `include` directly. You can simply include/add this directory from your projects, e.g., `-I./include`, to access the algorithms.

To access the control algorithms:
    
    #include "ControlAlgorithm.h"

To access the predication algorithms:
   
    #include "PredictorAlgorithm"
 
To access the supervised learning algorithms:
   
    #include "SupervisedAlgorithm.h"

RLLib uses the namespace: 

    using namespace RLLib


Testing
-------

RLLib provides a flexible testing framework. Follow these steps to quickly write a test case.

* To access the testing framework: `#include "HeaderTest.h"`

```javascript
#include "HeaderTest.h"

RLLIB_TEST(YourTest)

class YourTest Test: public YourTestBase
{
  public:
    YourTestTest() {}

    virtual ~Test() {}
    void run();

  private:
    void testYourMethod();
};

void YourTestBase::testYourMethod() {/** Your test code */}

void YourTestBase::run() { testYourMethod(); }
```
  
* Add `YourTest` to the `test/test.cfg` file.
* You can use `@YourTest` to execute only `YourTest`. For example, if you need to execute only MountainCar test cases, use @MountainCarTest.

Test Configuration
-------------------

We are using CMAKE >= 2.8.7 to build and run the test suite.

   * mkdir build
   * cd build; cmake .. 
   * make -j

Visualization
-------------

RLLib provides a [QT5](http://qt-project.org/qt5) based Reinforcement Learning problems and algorithms visualization tool named `RLLibViz`. Currently RLLibViz visualizes following problems and algorithms:

* On-policy:
    * SwingPendulum problem with continuous actions. We use AverageRewardActorCritic algorithm.

* Off-policy: 
    * ContinuousGridworld and MountainCar problems with discrete actions. We use Off-PAC algorithm.

* In order to run the visualization tool, you need to have QT4.8 installed in your system. 

* In order to install RLLibViz:     
    * Change directory to `visualization/RLLibViz`
    * qmake RLLibViz.pro
    * make -j
    * ./RLLibViz
	
Documentation
------------- 
   
* [http://web.cs.miami.edu/home/saminda/rllib.html](http://web.cs.miami.edu/home/saminda/rllib.html)
* [mloss.org](https://mloss.org/software/view/502/)  

Operating Systems
-----------------

* Ubuntu >= 11.04
* Windows (Visual Studio 2013)
* Mac OS X

TODO
----

* Variable action per state.
* Non-linear algorithms.
* Deep learning algorithms. 

Publications
------------

* [Dynamic Role Assignment using General ValueFunctions](http://www.humanoidsoccer.org/ws12/papers/HSR12_Abeyruwan.pdf)
* [Humanoid Robots and Spoken Dialog Systems for Brief Health Interventions](http://www.aaai.org/ocs/index.php/FSS/FSS14/paper/download/9122/9125)

Contact
-------

   Saminda Abeyruwan, PhD (saminda@cs.miami.edu, samindaa@gmail.com)


