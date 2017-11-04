CXXFLAGS=-g -std=c++11 -Wall -Wextra -lGL -lglfw -lm -ldl

all:
	g++ ${CXXFLAGS} gl3w.c main.cpp -o ao

clean:
	rm ao
