#pragma once
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <iostream>
#include <numeric>
#include <cstring>
#include <armadillo>

struct SRHTContext {
    int d_orig;
    int d_padded;
    int m;
    bool use_fwht;

    std::vector<float> signs;
    std::vector<int> sample_idx;

    arma::fmat ortho_mat;

    unsigned int seed;
};

void init_srht(SRHTContext &ctx, int d_orig, int m, unsigned int seed);
void fwht_inplace(float *vec, int n);
void apply_srht_batch(const SRHTContext &ctx, float **&data, int num_vectors);
