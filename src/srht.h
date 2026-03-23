#pragma once
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <iostream>
#include <numeric>
#include <cstring>

struct SRHTContext {
    int d_orig;
    int d_padded;
    int trunc_dim;
    bool is_pow2;

    // round_signs[r][j] = +1/-1 for round r, coordinate j
    // pow2: 1 round, each of size d_orig
    // non-pow2: 4 rounds, each of size d_padded
    std::vector<std::vector<float>> round_signs;

    unsigned int seed;
};

void init_srht(SRHTContext &ctx, int d_orig, unsigned int seed);
void fwht_inplace(float *vec, int n);
void apply_srht_batch(const SRHTContext &ctx, float **&data, int num_vectors);
