#include "srht.h"

using namespace std;

static bool is_power_of_2(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

static int floor_log2(int n) {
    int r = 0;
    while (n > 1) { r++; n >>= 1; }
    return r;
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

static void kacs_walk(float *data, int len) {
    int half = len / 2;
    for (int i = 0; i < half; i++) {
        float a = data[i];
        float b = data[i + half];
        data[i]        = a + b;
        data[i + half] = a - b;
    }
}

void init_srht(SRHTContext &ctx, int d_orig, unsigned int seed) {
    assert(d_orig > 0);

    ctx.d_orig = d_orig;
    ctx.seed = seed;
    ctx.is_pow2 = is_power_of_2(d_orig);

    mt19937 rng(seed);
    uniform_int_distribution<int> coin(0, 1);

    if (ctx.is_pow2) {
        ctx.d_padded = d_orig;
        ctx.trunc_dim = d_orig;

        ctx.round_signs.resize(1);
        ctx.round_signs[0].resize(d_orig);
        for (int i = 0; i < d_orig; i++)
            ctx.round_signs[0][i] = coin(rng) ? 1.0f : -1.0f;

        cout << ">>> SRHT initialized (FWHT path): d=" << d_orig
             << ", seed=" << seed << endl;
    } else {
        ctx.trunc_dim = 1 << floor_log2(d_orig);
        ctx.d_padded = (d_orig % 2 == 0) ? d_orig : d_orig + 1;

        int num_rounds = 4;
        ctx.round_signs.resize(num_rounds);
        for (int r = 0; r < num_rounds; r++) {
            ctx.round_signs[r].resize(ctx.d_padded);
            for (int i = 0; i < ctx.d_padded; i++)
                ctx.round_signs[r][i] = coin(rng) ? 1.0f : -1.0f;
        }

        cout << ">>> SRHT initialized (FWHT+Kac path): d=" << d_orig
             << ", d_padded=" << ctx.d_padded
             << ", trunc_dim=" << ctx.trunc_dim
             << ", seed=" << seed << endl;
    }
}

void apply_srht_batch(const SRHTContext &ctx, float **&data, int num_vectors) {
    int d_orig = ctx.d_orig;
    int d_padded = ctx.d_padded;

    float **new_data = new float*[num_vectors];

    if (ctx.is_pow2) {
        float scale = 1.0f / sqrtf((float)d_orig);

        for (int i = 0; i < num_vectors; i++) {
            float *buf = new float[d_orig];
            for (int j = 0; j < d_orig; j++)
                buf[j] = data[i][j] * ctx.round_signs[0][j];

            fwht_inplace(buf, d_orig);

            new_data[i] = new float[d_orig];
            for (int j = 0; j < d_orig; j++)
                new_data[i][j] = buf[j] * scale;

            delete[] buf;
            delete[] data[i];
        }
    } else {
        int trunc = ctx.trunc_dim;
        int start = d_padded - trunc;
        float fac = 1.0f / sqrtf((float)trunc);

        for (int i = 0; i < num_vectors; i++) {
            float *buf = new float[d_padded]();
            memcpy(buf, data[i], d_orig * sizeof(float));

            // Round 0: sign-flip(all) → FWHT([0..trunc)) → rescale → Kac
            for (int j = 0; j < d_padded; j++) buf[j] *= ctx.round_signs[0][j];
            fwht_inplace(buf, trunc);
            for (int j = 0; j < trunc; j++) buf[j] *= fac;
            kacs_walk(buf, d_padded);

            // Round 1: sign-flip(all) → FWHT([start..start+trunc)) → rescale → Kac
            for (int j = 0; j < d_padded; j++) buf[j] *= ctx.round_signs[1][j];
            fwht_inplace(buf + start, trunc);
            for (int j = start; j < start + trunc; j++) buf[j] *= fac;
            kacs_walk(buf, d_padded);

            // Round 2: sign-flip(all) → FWHT([0..trunc)) → rescale → Kac
            for (int j = 0; j < d_padded; j++) buf[j] *= ctx.round_signs[2][j];
            fwht_inplace(buf, trunc);
            for (int j = 0; j < trunc; j++) buf[j] *= fac;
            kacs_walk(buf, d_padded);

            // Round 3: sign-flip(all) → FWHT([start..start+trunc)) → rescale → Kac
            for (int j = 0; j < d_padded; j++) buf[j] *= ctx.round_signs[3][j];
            fwht_inplace(buf + start, trunc);
            for (int j = start; j < start + trunc; j++) buf[j] *= fac;
            kacs_walk(buf, d_padded);

            // Compensate 4x norm growth from Kac's Walk
            for (int j = 0; j < d_padded; j++) buf[j] *= 0.25f;

            new_data[i] = new float[d_orig];
            memcpy(new_data[i], buf, d_orig * sizeof(float));

            delete[] buf;
            delete[] data[i];
        }
    }

    delete[] data;
    data = new_data;

    cout << ">>> SRHT applied to " << num_vectors << " vectors: "
         << d_orig << "d"
         << " [" << (ctx.is_pow2 ? "FWHT" : "FWHT+Kac") << "]" << endl;
}
