#pragma once
#include <iostream>
#include <vector>
#include <mlpack/methods/kmeans/kmeans.hpp>
#include <armadillo>
#include <unordered_map>
#include <omp.h>
#include <sys/time.h>
#include <queue>

#include "dist_calculation.h"
#include "utils.h"

using namespace std;

struct Compare {
    bool operator()(const std::pair<float, int>& a, const std::pair<float, int>& b) {
        return a.first > b.first;
    }
};

void ann_query(float ** &dataset, int ** &queryknn_results, long int dataset_size, int data_dimensionality, int query_size, int k_size, float ** &querypoints, vector<unordered_map<pair<int, int>, vector<int>, hash_pair>> &indexes, float * &centroids_list, int subspace_num, int subspace_dimensionality, int kmeans_num_centroid, int kmeans_dim, int collision_num, int candidate_num, int number_of_threads, long int &query_time);

void dynamic_activate(vector<unordered_map<pair<int, int>, vector<int>, hash_pair>> &indexes, vector<pair<int, int>> &retrieved_cell, vector<float> &first_half_dists, vector<int> &first_half_idx, vector<float> &second_half_dists, vector<int> &second_half_idx, int collision_num, int kmeans_num_centroid, int subspace_idx);

void scalable_dynamic_activate(vector<unordered_map<pair<int, int>, vector<int>, hash_pair>> &indexes, vector<pair<int, int>> &retrieved_cell, vector<float> &first_half_dists, vector<int> &first_half_idx, vector<float> &second_half_dists, vector<int> &second_half_idx, int collision_num, int kmeans_num_centroid, int subspace_idx);
