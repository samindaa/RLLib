# This file contains the minimum information needed to compile
# the simulations.
# Saminda Abeyruwan 
# saminda@cs.miami.edu
# To access RLLib documentation, please visit http://saminda.org

CC = g++
CFLAGS = -I. -I./src -I./test -I./simulation -Wall -Werror -O3
INCLUDES = Action.h Control.h ControlAlgorithm.h Math.h Policy.h Predictor.h PredictorAlgorithm.h \
		   Projector.h Representation.h Supervised.h SupervisedAlgorithm.h Tiles.h TilesImpl.h Trace.h \
		   Vector.h

all: Test

Test:
	$(CC) $(CFLAGS) \
	test/VectorTest.cpp \
	test/TraceTest.cpp \
	test/LearningAlgorithmTest.cpp \
	test/HelicopterTest.cpp \
	test/PendulumOnPolicyLearning.cpp \
	test/ProjectorTest.cpp \
	test/MountainCarTest.cpp \
	test/ContinuousGridworldTest.cpp \
	test/SwingPendulumTest.cpp \
	test/ExtendedProblemsTest.cpp \
	test/Test.cpp -o Test
	
clean:
	rm -f Test