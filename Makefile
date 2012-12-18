# This file contains the minimum information needed to compile
# the simulations.
# Saminda Abeyruwan 
# saminda@cs.miami.edu
# To access RLLib documentation, please visit http://saminda.org

CC = g++
CFLAGS = -I. -I./src -Wall -Werror -g

all: Main VectorTest

Main:
	$(CC) $(CFLAGS) simulation/Main.cpp -o Main

VectorTest:
	$(CC) $(CFLAGS) test/VectorTest.cpp -o VectorTest

clean:
	rm -f Main
	rm -f VectorTest
