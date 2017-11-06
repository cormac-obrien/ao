CC=clang
CXX=clang

CFLAGS=-g -std=c11 -Wall -Wextra
CXXFLAGS=-g -std=c++11 -Wall -Wextra
LDFLAGS=-lGL -lglfw -lm -ldl -lstdc++ -lSOIL

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

ao: main.o gl3w.o
	$(CXX) $(LDFLAGS) $^ -o $@

clean:
	rm *.o ao
