#include "layered_bitmap_sc.h"

void init_layered_bitmap(LayeredBitmapSC &ctx, long int num_points, int max_layers) {
    ctx.num_points = num_points;
    ctx.max_layers = max_layers;
    ctx.words_per_layer = (int)((num_points + 63) / 64);
    ctx.current_max_score = 0;

    ctx.layers.resize(max_layers + 1);
    for (int t = 0; t <= max_layers; t++) {
        ctx.layers[t].assign(ctx.words_per_layer, 0ULL);
    }

    ctx.collision_bitmap.assign(ctx.words_per_layer, 0ULL);
}

void reset_layered_bitmap(LayeredBitmapSC &ctx) {
    for (int t = 1; t <= ctx.current_max_score; t++) {
        std::memset(ctx.layers[t].data(), 0, ctx.words_per_layer * sizeof(uint64_t));
    }
    ctx.current_max_score = 0;
}

void clear_collision_bitmap(LayeredBitmapSC &ctx) {
    std::memset(ctx.collision_bitmap.data(), 0, ctx.words_per_layer * sizeof(uint64_t));
}

void update_score_layers(LayeredBitmapSC &ctx) {
    int W = ctx.words_per_layer;

    // Process layers from highest score down to 1.
    // For each word position, compute the set of points that are both in
    // layer t and in the collision bitmap ("move"), promote them to t+1,
    // and remove them from both the layer and the remaining collision set.
    // After all existing layers are processed, any remaining bits in the
    // collision bitmap are newly-colliding points that enter layer 1.
    for (int t = ctx.current_max_score; t >= 1; t--) {
        uint64_t *lt  = ctx.layers[t].data();
        uint64_t *lt1 = ctx.layers[t + 1].data();
        uint64_t *rem = ctx.collision_bitmap.data();

        for (int w = 0; w < W; w++) {
            uint64_t move = lt[w] & rem[w];
            lt[w]  ^= move;
            lt1[w] |= move;
            rem[w] ^= move;
        }
    }

    // Remaining bits are new points (were score-0) → enter layer 1
    {
        uint64_t *l1  = ctx.layers[1].data();
        uint64_t *rem = ctx.collision_bitmap.data();
        for (int w = 0; w < W; w++) {
            l1[w] |= rem[w];
        }
    }

    // Update current_max_score: check if layer current_max_score+1 became non-empty
    if (ctx.current_max_score < ctx.max_layers) {
        // The only layer that could have newly appeared is current_max_score+1
        int next = ctx.current_max_score + 1;
        const uint64_t *ln = ctx.layers[next].data();
        for (int w = 0; w < W; w++) {
            if (ln[w] != 0) {
                ctx.current_max_score = next;
                break;
            }
        }
    }
}

// Extract at most budget candidate point IDs from highest-score layers downward.
// Single-pass: traverse layers and bits once, stop when out.size() >= budget.
// Returns the number of candidates extracted (<= budget).
int extract_candidates(const LayeredBitmapSC &ctx, std::vector<int> &out, int budget) {
    int W = ctx.words_per_layer;

    for (int t = ctx.current_max_score; t >= 1 && (int)out.size() < budget; t--) {
        const uint64_t *layer = ctx.layers[t].data();
        for (int w = 0; w < W && (int)out.size() < budget; w++) {
            uint64_t word = layer[w];
            int base = w * 64;
            while (word && (int)out.size() < budget) {
                int bit = __builtin_ctzll(word);
                out.push_back(base + bit);
                word &= word - 1;
            }
        }
    }
    return (int)out.size();
}
