#include "layered_bitmap_sc.h"
#include <omp.h>

void init_layered_bitmap(LayeredBitmapSC &ctx, long int num_points, int max_layers) {
    ctx.num_points = num_points;
    ctx.max_layers = max_layers;
    ctx.words_per_layer = (int)((num_points + 63) / 64);
    ctx.current_max_score = 0;

    ctx.layers.resize(max_layers + 1);
    for (int t = 0; t <= max_layers; t++) {
        ctx.layers[t].assign(ctx.words_per_layer, 0ULL);
    }
}

void reset_layered_bitmap(LayeredBitmapSC &ctx) {
    for (int t = 1; t <= ctx.current_max_score; t++) {
        std::memset(ctx.layers[t].data(), 0, ctx.words_per_layer * sizeof(uint64_t));
    }
    ctx.current_max_score = 0;
}

// One-shot conversion: read per-point collision counts, scatter each point
// into the layer bitmap matching its score, clear the count, and track the
// maximum score seen.  Parallelized by bitmap-word position (each word covers
// 64 consecutive points and is fully independent of all other words).
void build_layers_from_counts(LayeredBitmapSC &ctx, unsigned char *counts, int num_threads) {
    int W = ctx.words_per_layer;
    long int N = ctx.num_points;
    int observed_max = 0;

    #pragma omp parallel for num_threads(num_threads) schedule(static) reduction(max:observed_max)
    for (int w = 0; w < W; w++) {
        int base = w * 64;
        int end = base + 64;
        if (end > N) end = (int)N;

        for (int b = base; b < end; b++) {
            int score = counts[b];
            if (score > 0) {
                ctx.layers[score][w] |= 1ULL << (b - base);
                counts[b] = 0;
                if (score > observed_max) observed_max = score;
            }
        }
    }

    ctx.current_max_score = observed_max;
}

// Count the popcount of one layer bitmap.
static int layer_popcount(const std::vector<uint64_t> &layer, int W) {
    int cnt = 0;
    const uint64_t *data = layer.data();
    for (int w = 0; w < W; w++) {
        cnt += __builtin_popcountll(data[w]);
    }
    return cnt;
}

// Extract all point IDs from a single layer bitmap into out.
static void extract_layer(const std::vector<uint64_t> &layer, int W, std::vector<int> &out) {
    const uint64_t *data = layer.data();
    for (int w = 0; w < W; w++) {
        uint64_t word = data[w];
        int base = w * 64;
        while (word) {
            int bit = __builtin_ctzll(word);
            out.push_back(base + bit);
            word &= word - 1;
        }
    }
}

// Extract candidate point IDs from highest-score layers downward.
// Matches the original SuCo behavior: finds a boundary score threshold,
// then includes ALL points at or above that threshold. This means the
// returned count may exceed budget (the entire boundary layer is included).
int extract_candidates(const LayeredBitmapSC &ctx, std::vector<int> &out, int budget) {
    int W = ctx.words_per_layer;
    int remaining = budget;

    int boundary_layer = 0;
    for (int t = ctx.current_max_score; t >= 1; t--) {
        int layer_count = layer_popcount(ctx.layers[t], W);
        if (layer_count <= remaining) {
            remaining -= layer_count;
        } else {
            boundary_layer = t;
            break;
        }
    }

    for (int t = ctx.current_max_score; t >= boundary_layer && t >= 1; t--) {
        extract_layer(ctx.layers[t], W, out);
    }

    return (int)out.size();
}
