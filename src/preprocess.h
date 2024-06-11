#pragma once
#include <iostream>
#include <armadillo>
#include <getopt.h>

using namespace std;

void load_data(float ** &dataset, char * dataset_path, long int dataset_size, int data_dimensionality);
void load_query(float ** &querypoints, char * query_path, int query_size, int data_dimensionality);
void load_groundtruth(long int ** &gt, char * groundtruth_path, int query_size, int k_size);

void transfer_data(float ** &dataset, vector<arma::mat> &data_list, long int dataset_size, int subspace_num, int subspace_dimensionality);