#include "srht.h"

using namespace std;

static int next_power_of_2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

void fwht_inplace(float *vec, int n) {
    for (int len = 1; len < n; len <<= 1) {
        for (int i = 0; i < n; i += len << 1) {
            for (int j = 0; j < len; j++) {
                float u = vec[i + j];
                float v = vec[i + j + len];
                vec[i + j]       = u + v;
                vec[i + j + len] = u - v;
            }
        }
    }
}

void init_srht(SRHTContext &ctx, int d_orig, int m, unsigned int seed) {
    assert(m > 0 && m <= d_orig);

    ctx.d_orig = d_orig;
    ctx.d_padded = next_power_of_2(d_orig);
    ctx.m = m;
    ctx.seed = seed;

    mt19937 rng(seed);

    ctx.signs.resize(ctx.d_padded);
    uniform_int_distribution<int> coin(0, 1);
    for (int i = 0; i < ctx.d_padded; i++) {
        ctx.signs[i] = coin(rng) ? 1.0f : -1.0f;
    }

    vector<int> all_indices(ctx.d_padded);
    iota(all_indices.begin(), all_indices.end(), 0);
    shuffle(all_indices.begin(), all_indices.end(), rng);

    ctx.sample_idx.assign(all_indices.begin(), all_indices.begin() + m);
    sort(ctx.sample_idx.begin(), ctx.sample_idx.end());

    ctx.wrap_scale.resize(d_orig);
    for (int k = 0; k < d_orig; k++) {
        int count = ctx.d_padded / d_orig + (k < ctx.d_padded % d_orig ? 1 : 0);
        ctx.wrap_scale[k] = 1.0f / sqrtf((float)count);
    }

    cout << ">>> SRHT initialized: d_orig=" << d_orig
         << ", d_padded=" << ctx.d_padded
         << ", m=" << m
         << ", seed=" << seed << endl;
}

void apply_srht_batch(const SRHTContext &ctx, float **&data, int num_vectors) {
    int d_padded = ctx.d_padded;
    int d_orig = ctx.d_orig;
    int m = ctx.m;
    float scale = 1.0f / sqrtf((float)m);

    float **new_data = new float*[num_vectors];

    for (int i = 0; i < num_vectors; i++) {
        float *buf = new float[d_padded];
        for (int j = 0; j < d_padded; j++) {
            int k = j % d_orig;
            buf[j] = data[i][k] * ctx.signs[j] * ctx.wrap_scale[k];
        }

        fwht_inplace(buf, d_padded);

        new_data[i] = new float[m];
        for (int j = 0; j < m; j++) {
            new_data[i][j] = buf[ctx.sample_idx[j]] * scale;
        }

        delete[] buf;
        delete[] data[i];
    }

    delete[] data;
    data = new_data;

    cout << ">>> SRHT applied to " << num_vectors << " vectors: "
         << d_orig << "d -> " << m << "d" << endl;
}
