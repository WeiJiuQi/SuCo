#pragma once
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <iostream>
#include <numeric>

struct SRHTContext {
    int d_orig;
    int d_padded;
    int m;
    std::vector<float> signs;
    std::vector<int> sample_idx;
    std::vector<float> wrap_scale;
    unsigned int seed;
};

void init_srht(SRHTContext &ctx, int d_orig, int m, unsigned int seed);
void fwht_inplace(float *vec, int n);
void apply_srht_batch(const SRHTContext &ctx, float **&data, int num_vectors);
