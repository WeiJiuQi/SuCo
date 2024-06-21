# Subspace Collision: An Efficient and Accurate Framework for High-dimensional Approximate Nearest Neighbor Search

This is the source code of the method proposed in paper: **Subspace Collision: An Efficient and Accurate Framework for High-dimensional Approximate Nearest Neighbor Search (submitted to SIGMOD 2025)**.

## Dependency

+ g++ version: 11.4.0 
+ CMake  >= 2.8.5
+ Armadillo \>= 6.500.0
  + sudo apt-get install liblapack-dev libblas-dev libboost-dev
  + sudo apt-get install libarmadillo-dev
+ Boost
  + sudo apt-get install libboost-math-dev libboost-program-options-dev libboost-random-dev libboost-test-dev libxml2-dev
+ Mlpack
  + sudo apt-get install libmlpack-dev

## Compilation

```
make
```

## Usage

### Command and Parameters

```
./scimi --dataset-path $PATH_TO_DATASET$ --query-path $PATH_TO_QUERY$ --groundtruth-path $PATH_TO_GROUNDTRUTH$ --dataset-size $n$ --query-size $QUERY_SIZE$ --k-size $k$ --data-dimensionality $DIMENSIONALITY$ --subspace-dimensionality $SUBSPACE_DIMENSIONALITY$ --subspace-num $SUBSPACE_NUMBER$ --candidate-ratio $beta$ --collision-ratio $alpha$ --kmeans-num-centroid $K$ --kmeans-num-iters $ITERATION$ --index-path $PATH_TO_INDEX$ --load-index
```

+ --dataset-path: the path to dataset file
+ --query-path: the path to query file 
+ --groundtruth-path: the path to groundtruth file
+ --dataset-size: n, a positive integer, the cardinality of dataset
+ --query-size: a positive integer, the number of queries in the query file
+ --k-size: a positive integer, the number of returned points for k-ANN queries
+ --data-dimensionality: a positive integer, the dimensionality of dataset
+ --subspace-dimensionality: a positive integer, the dimensionality of each subspace
+ --subspace-num: a positive integer, the number of subspaces
+ --candidate-ratio: beta, a positive number in (0,1], beta*n candidates will be selected for reranking
+ --collision-ratio alpha, a positive number in (0,1], alpha*n points will be considered as colliding with query
+ --kmeans-num-centroid: K, a positive integer, the number of clusters for K-means clustering
+ --kmeans-num-iters: a positive integer, the number of iterations for K-means clustering
+ --index-path: the path to index file, when --load-index is set, load the index from this file, when --load-index is not set, generate index and save to this file
+ --load-index: whether generate or load index from the --index-path

## Dataset, Query, and Benchmark

#### Dataset

We use eight real-world datasets for ANN search: [Deep1M](https://www.cse.cuhk.edu.hk/systems/hash/gqr/datasets.html), [Gist1M](https://www.cse.cuhk.edu.hk/systems/hash/gqr/datasets.html), [Sift10M](http://corpus-texmex.irisa.fr/), [Microsoft SPACEV10M](https://big-ann-benchmarks.com/neurips21.html), [Yandex Deep10M](https://big-ann-benchmarks.com/neurips21.html), TinyImages80M, [Sift100M](http://corpus-texmex.irisa.fr/),  and [Yandex Deep100M](https://big-ann-benchmarks.com/neurips21.html). The key statistics of the datasets are summarized as follows.

|       Dataset       | Cardinality | Dimensions |  LID  |
| :-----------------: | :---------: | :--------: | :---: |
|       Deep1M        |  1,000,000  |    256     | 37.26 |
|       Gist1M        |  1,000,000  |    960     | 70.15 |
|       Sift10M       | 10,000,000  |    128     | 22.05 |
| Microsoft SPACEV10M | 10,000,000  |    100     | 41.72 |
|   Yandex Deep10M    | 10,000,000  |     96     | 29.10 |
|    TinyImages80M    | 79,302,017  |    384     | 61.75 |
|      Sift100M       | 100,000,000 |    128     | 23.79 |
|   Yandex Deep100M   | 100,000,000 |     96     | 29.61 |

#### Query

We randomly select 100 data points as queries and remove them from the original datasets.

#### Format of Dataset and Query

Our input dataset file and query file is in binary format without any indice, i.e., the binary file is organized as the following format:

```
{The binary vectors of data points, arranged in turn}
{The binary vectors of query points, arranged in turn}
```

#### Groundtruth

The format of groundtruth file is:

```
{The index (type: long int) of exact k-NN points in the dataset, arranged in turn for each query point}
```

## Parameters used in Our Paper

### Parameters shared by different datasets

|      Parameter      |            Value            |
| :-----------------: | :-------------------------: |
|       k-size        |             50              |
|   candidate-ratio   | [0.001, 0.005] (best range) |
|   collision-ratio   |  [0.03, 0.05] (best range)  |
| kmeans-num-centroid |             50              |
|  kmeans-num-iters   |              2              |

### Unique parameters for different datasets

|       Dataset       | dataset-size | data-dimensionality | subspace-dimensionality | subspace-num |
| :-----------------: | :----------: | :-----------------: | :---------------------: | :----------: |
|       Deep1M        |  1,000,000   |         256         |           32            |      8       |
|       Gist1M        |  1,000,000   |         960         |           120           |      8       |
|       Sift10M       |  10,000,000  |         128         |           16            |      8       |
| Microsoft SPACEV10M |  10,000,000  |         100         |           10            |      10      |
|   Yandex Deep10M    |  10,000,000  |         96          |           12            |      8       |
|    TinyImages80M    |  79,302,017  |         384         |           48            |      8       |
|      Sift100M       | 100,000,000  |         128         |           16            |      8       |
|   Yandex Deep100M   | 100,000,000  |         96          |           12            |      8       |

