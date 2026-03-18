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

static void gen_random_ortho_matrix(vector<float> &out, int d, int m, unsigned int seed) {
    mt19937 rng(seed);
    normal_distribution<float> gauss(0.0f, 1.0f);

    vector<float> cols(d * d);
    for (int i = 0; i < d * d; i++)
        cols[i] = gauss(rng);

    for (int i = 0; i < d; i++) {
        float *vi = cols.data() + i * d;

        for (int j = 0; j < i; j++) {
            const float *vj = cols.data() + j * d;
            float proj = 0.0f;
            for (int k = 0; k < d; k++)
                proj += vi[k] * vj[k];
            for (int k = 0; k < d; k++)
                vi[k] -= proj * vj[k];
        }

        float norm = 0.0f;
        for (int k = 0; k < d; k++)
            norm += vi[k] * vi[k];
        norm = sqrtf(norm);
        if (norm > 1e-12f) {
            for (int k = 0; k < d; k++)
                vi[k] /= norm;
        }
    }

    out.resize(m * d);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < d; j++)
            out[i * d + j] = cols[i * d + j];
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

        gen_random_ortho_matrix(ctx.ortho_matrix, d_orig, m, seed);

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
        const float *Q = ctx.ortho_matrix.data();

        for (int i = 0; i < num_vectors; i++) {
            new_data[i] = new float[m];
            for (int j = 0; j < m; j++) {
                float dot = 0.0f;
                const float *row = Q + j * d_orig;
                for (int k = 0; k < d_orig; k++)
                    dot += row[k] * data[i][k];
                new_data[i][j] = dot * scale;
            }
            delete[] data[i];
        }
    }

    delete[] data;
    data = new_data;

    cout << ">>> SRHT applied to " << num_vectors << " vectors: "
         << d_orig << "d -> " << m << "d"
         << " [" << (ctx.use_fwht ? "FWHT" : "ortho") << "]" << endl;
}
