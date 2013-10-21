RLLib 
=====
(C++ Template Library to Learn Behaviors and Represent Learnable Knowledge using On/Off Policy Reinforcement Learning)
----------------------------------------------------------------------------------------------------------------------

RLLib is a lightweight C++ template library that implements `incremental`, `standard`, and `gradient temporal-difference` learning algorithms in Reinforcement Learning. It is a highly optimized library that is designed and written specifically for robotic applications. The implementation of the RLLib library is inspired by the RLPark API, which is a library of temporal-difference learning algorithms written in Java. 

Features
--------

* **Off-policy prediction algorithms**: 
 `GTD(lambda)`
 `GQ(lambda)`
* **Off-policy control algorithms**:  
 `Greedy-GQ(lambda)`
 `Softmax-GQ(lambda)`
 `Off-PAC` (can be used in on-policy setting)
* **On-policy algorithms**: 
 `TD(lambda)`
 `SARSA(lambda)`
 `Expected-SARSA(lambda)`
 `Actor-Critic (continuous and discrete actions, discounted, averaged reward settings, etc.)` 
* **Supervised learning algorithms**: 
 `Adaline`
 `IDBD`
 `SemiLinearIDBD`
 `Autostep`
* **Policies**: 
 `Random`
 `Random50%Bias`
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
 `Helicopter`
`Continuous Grid World`
* **Optimization**: 
 Optimized for very fast duty cycles (e.g., with culling traces, RLLib has been tested on `the Robocup 3D simulator agent`, and on `the NAO V4  (cognition thread)`). 
* **Usage**: 
 The algorithm usage is very much similar to RLPark, therefore, swift learning curve.
* **Examples**: 
 There are a plethora of examples demonstrating on-policy and off-policy control experiments.
* **Visualization**:
 We provide a Qt4 based application to visualize benchmark problems.  


Demo
----

[![Off-PAC ContinuousGridworld](http://i1.ytimg.com/vi/SpBbdvhx4tM/3.jpg?time=1382317024739)](http://www.youtube.com/watch?v=9THBj9nX5gU)
[![AverageRewardActorCritic SwingPendulum (Continuous Actions)](http://i1.ytimg.com/vi/nwxAG2WXl3Y/3.jpg?time=1382317239212)](http://www.youtube.com/watch?v=ktNYS-ApAko)

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


Publication
-----------

[Dynamic Role Assignment using General ValueFunctions](http://www.humanoidsoccer.org/ws12/papers/HSR12_Abeyruwan.pdf)

Contact
-------

   Saminda Abeyruwan (saminda@cs.miami.edu)


