#include "srht.h"

using namespace std;

static bool is_power_of_2(int n) {
    return n > 0 && (n & (n - 1)) == 0;
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
    ctx.m = m;
    ctx.seed = seed;
    ctx.use_fwht = is_power_of_2(d_orig);

    if (ctx.use_fwht) {
        ctx.d_padded = d_orig;

        mt19937 rng(seed);

        ctx.signs.resize(d_orig);
        uniform_int_distribution<int> coin(0, 1);
        for (int i = 0; i < d_orig; i++) {
            ctx.signs[i] = coin(rng) ? 1.0f : -1.0f;
        }

        vector<int> all_indices(d_orig);
        iota(all_indices.begin(), all_indices.end(), 0);
        shuffle(all_indices.begin(), all_indices.end(), rng);

        ctx.sample_idx.assign(all_indices.begin(), all_indices.begin() + m);
        sort(ctx.sample_idx.begin(), ctx.sample_idx.end());

        cout << ">>> SRHT initialized (FWHT path): d=" << d_orig
             << ", m=" << m << ", seed=" << seed << endl;
    } else {
        ctx.d_padded = d_orig;

        arma::arma_rng::set_seed(seed);
        arma::fmat G = arma::randn<arma::fmat>(d_orig, d_orig);
        arma::fmat Q, R;
        arma::qr(Q, R, G);
        ctx.ortho_mat = Q.head_cols(m).t();

        cout << ">>> SRHT initialized (ortho path): d=" << d_orig
             << ", m=" << m << ", seed=" << seed << endl;
    }
}

void apply_srht_batch(const SRHTContext &ctx, float **&data, int num_vectors) {
    int d_orig = ctx.d_orig;
    int m = ctx.m;

    float **new_data = new float*[num_vectors];

    if (ctx.use_fwht) {
        int d_padded = ctx.d_padded;
        float scale = 1.0f / sqrtf((float)m);

        for (int i = 0; i < num_vectors; i++) {
            float *buf = new float[d_padded];
            for (int j = 0; j < d_padded; j++)
                buf[j] = data[i][j] * ctx.signs[j];

            fwht_inplace(buf, d_padded);

            new_data[i] = new float[m];
            for (int j = 0; j < m; j++)
                new_data[i][j] = buf[ctx.sample_idx[j]] * scale;

            delete[] buf;
            delete[] data[i];
        }
    } else {
        float scale = sqrtf((float)d_orig / (float)m);

        arma::fmat X(d_orig, num_vectors);
        for (int i = 0; i < num_vectors; i++)
            memcpy(X.colptr(i), data[i], d_orig * sizeof(float));

        arma::fmat Y = scale * (ctx.ortho_mat * X);

        for (int i = 0; i < num_vectors; i++) {
            new_data[i] = new float[m];
            memcpy(new_data[i], Y.colptr(i), m * sizeof(float));
            delete[] data[i];
        }
    }

    delete[] data;
    data = new_data;

    cout << ">>> SRHT applied to " << num_vectors << " vectors: "
         << d_orig << "d -> " << m << "d"
         << " [" << (ctx.use_fwht ? "FWHT" : "ortho/GEMM") << "]" << endl;
}
