RLLib (A Lightweight, Standard, and On/Off Policy Reinforcement Learning C++ Template Library)
==============================================================================================

RLLib is a lightweight C++ template library that implements `incremental`, `standard`, and `gradient temporal-difference` learning algorithms in Reinforcement Learning. It is a highly optimized library that is designed and written specifically for robotic applications. The implementation of the RLLib library is inspired by the RLPark API, which is a library of temporal-difference learning algorithms written in Java. 

Features
--------

* Off-policy prediction algorithms: `GTD(lambda)`, and `GQ(lambda)`,
* Off-policy control algorithms:  `Greedy-GQ(lambda)`, `Softmax-GQ(lambda)`, and `Off-PAC` (can be used in on-policy setting),
* On-policy algorithms: `TD(lambda)`, `SARSA(lambda)`, `Expected-SARSA(lambda)`, and `Actor-Critic` (continuous and discrete actions), 
* Supervised learning algorithms: `Adaline`, `IDBD`, `SemiLinearIDBD`, and `Autostep`, 
* Policies: `Random`, `Random50%Bias`, `Greedy`, `Epsilon-greedy`, `Boltzmann`, `Normal`, and `Softmax`,
* Dot product: an efficient dot product implementation for tile coding base feature representations (with culling traces),
* Benchmarks: `Mountain Car`, `Mountain Car 3D`, `Swinging Pendulum`, `Helicopter`, and `Continuous Grid World` environments,
* Optimization: optimized for very fast duty cycles (e.g., with culling traces, RLLib has been tested on the Robocup 3D simulator, and on the NAO V4  (cognition thread)), 
* Usage: main algorithm usage is very much similar to RLPark, therefore, swift learning curve, and
* Examples: a plethora of examples demonstrating on-policy and off-policy control experiments.


Usage
-----

RLLib is a C++ template library. The header files are located in the `src` directly. You can simply include this directory from your projects, e.g., `-I./src`, to access the algorithms.

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

    \#include "HeaderTest.h"
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


* Add `YourTest` to the `test/test.cfg` file.

Test Configuration
-------------------

The test cases are executed using:
   
    ./configure
    make
    ./RLLibTest

Documentation
------------- 
   
* [http://web.cs.miami.edu/home/saminda/rllib.html](http://web.cs.miami.edu/home/saminda/rllib.html)
* [mloss.org](https://mloss.org/software/view/502/)  


Contact
-------

   Saminda Abeyruwan (saminda@cs.miami.edu)


