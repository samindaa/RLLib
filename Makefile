# This file contains the minimum information needed to compile
# the simulations.
# Saminda Abeyruwan 
# saminda@cs.miami.edu
# To access RLLib documentation, please visit http://saminda.org

CFLAGS = -I.

all: Main

Main:
	g++ $(CFLAGS) simulation/Main.cpp -o Main

clean:
	rm Main
