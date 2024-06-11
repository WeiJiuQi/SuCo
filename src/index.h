#pragma once
#include <iostream>
#include <vector>
#include <mlpack/methods/kmeans/kmeans.hpp>
#include <armadillo>
#include <unordered_map>
#include <omp.h>
#include <sys/time.h>

#include "utils.h"

using namespace std;

void load_indexes(char * index_path, vector<unordered_map<pair<int, int>, vector<int>, hash_pair>> &indexes, float * centroids_list, int * assignments_list, long int dataset_size, int kmeans_dim, int subspace_num, int kmeans_num_centroid);
void gen_indexes(vector<arma::mat> data_list, vector<unordered_map<pair<int, int>, vector<int>, hash_pair>> &indexes, long int dataset_size, float * centroids_list, int * assignments_list, int kmeans_dim, int subspace_num, int kmeans_num_centroid, int kmeans_num_iters, long int &index_time);
void save_indexes(char * index_path, float * centroids_list, int * assignments_list, long int dataset_size, int kmeans_dim, int subspace_num, int kmeans_num_centroid);