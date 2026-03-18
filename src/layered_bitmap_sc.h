#pragma once
#include <vector>
#include <cstdint>
#include <cstring>

struct LayeredBitmapSC {
    long int num_points;
    int max_layers;        // = subspace_num (max possible SC-score)
    int words_per_layer;   // = ceil(num_points / 64)
    int current_max_score;

    // layers[t] = bitmap of points whose SC-score is exactly t, t in [1..max_layers]
    // layers[0] is unused (implicit zero-score layer)
    std::vector<std::vector<uint64_t>> layers;
};

void init_layered_bitmap(LayeredBitmapSC &ctx, long int num_points, int max_layers);
void reset_layered_bitmap(LayeredBitmapSC &ctx);
void build_layers_from_counts(LayeredBitmapSC &ctx, unsigned char *counts, int num_threads);
int  extract_candidates(const LayeredBitmapSC &ctx, std::vector<int> &out, int budget);
