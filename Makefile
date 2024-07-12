SETTING=-O3 -std=c++17 -larmadillo -lmlpack -lboost_serialization -fopenmp -fpic -march=native -mavx -mavx2 -msse3

SOURCE=$(wildcard ./src/*.cpp)

scimi:$(SRCS)
	rm -rf suco
	g++ $(SOURCE) -o suco $(SETTING)

clean:
	rm -rf suco
