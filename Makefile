SETTING=-O3 -std=c++17 -larmadillo -lmlpack -lboost_serialization -fopenmp -fpic -march=native -mavx512f

SOURCE=$(wildcard ./src/*.cpp)

suco:$(SRCS)
	rm -rf suco
	g++ $(SOURCE) -o suco $(SETTING)

clean:
	rm -rf suco
