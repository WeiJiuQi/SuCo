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

    ctx.collision_flag.assign(num_points, 0);
}

void reset_layered_bitmap(LayeredBitmapSC &ctx) {
    for (int t = 1; t <= ctx.current_max_score; t++) {
        std::memset(ctx.layers[t].data(), 0, ctx.words_per_layer * sizeof(uint64_t));
    }
    ctx.current_max_score = 0;
}

// Fused flag-to-bitmap conversion + score layer update, parallelized by word.
// Each word position is fully independent: convert 64 flag bytes into one
// bitmap word, then run the layer promotion logic for that word, then clear
// the flag bytes. No atomics needed.
void update_score_layers_from_flags(LayeredBitmapSC &ctx, int num_threads) {
    int W = ctx.words_per_layer;
    long int N = ctx.num_points;
    int max_t = ctx.current_max_score;
    int next_layer = max_t + 1;
    int promoted = 0;

    #pragma omp parallel for num_threads(num_threads) schedule(static) reduction(|:promoted)
    for (int w = 0; w < W; w++) {
        // Phase A: convert 64 flag bytes → 1 bitmap word, clear flags
        uint64_t remaining = 0;
        int base = w * 64;
        int end = base + 64;
        if (end > N) end = (int)N;
        uint8_t *p = ctx.collision_flag.data() + base;
        for (int b = 0; b < end - base; b++) {
            remaining |= (uint64_t)p[b] << b;
            p[b] = 0;
        }

        if (remaining == 0) continue;

        // Phase B: score layer promotion for this word
        for (int t = max_t; t >= 1; t--) {
            uint64_t move = ctx.layers[t][w] & remaining;
            ctx.layers[t][w]     ^= move;
            ctx.layers[t + 1][w] |= move;
            remaining ^= move;
        }
        ctx.layers[1][w] |= remaining;

        if (next_layer <= ctx.max_layers && ctx.layers[next_layer][w] != 0) {
            promoted = 1;
        }
    }

    if (promoted && ctx.current_max_score < ctx.max_layers) {
        ctx.current_max_score++;
    }
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
