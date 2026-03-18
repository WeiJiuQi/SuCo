#include "query.h"

void ann_query(float ** &dataset, int ** &queryknn_results, long int dataset_size, int data_dimensionality, int query_size, int k_size, float ** &querypoints, vector<unordered_map<pair<int, int>, vector<int>, hash_pair>> &indexes, float * &centroids_list, int subspace_num, int subspace_dimensionality, int kmeans_num_centroid, int kmeans_dim, int collision_num, int candidate_num, int number_of_threads, long int &query_time) {
    struct timeval start_query, end_query;
    progress_display pd_query(query_size);

    // Layered Bitmap SC aggregator (replaces dense collision_count array)
    LayeredBitmapSC lbsc;
    init_layered_bitmap(lbsc, dataset_size, subspace_num);

    // Reused buffers per query to avoid repeated allocation in subspace loop
    vector<float> first_half_dists(kmeans_num_centroid);
    vector<int> first_half_idx(kmeans_num_centroid);
    vector<float> second_half_dists(kmeans_num_centroid);
    vector<int> second_half_idx(kmeans_num_centroid);

    for (int i = 0; i < query_size; i++) {
        gettimeofday(&start_query, NULL);

        for (int j = 0; j < subspace_num; j++) {

            // first half dist
            for (int z = 0; z < kmeans_num_centroid; z++) {
                first_half_dists[z] = faiss::fvec_L2sqr_avx512(&querypoints[i][j * subspace_dimensionality], &centroids_list[j * 2 * kmeans_num_centroid * kmeans_dim + z * kmeans_dim], kmeans_dim);
            }

            // first half sort
            iota(first_half_idx.begin(), first_half_idx.end(), 0);
            sort(first_half_idx.begin(), first_half_idx.end(), [&first_half_dists](int i1, int i2) {return first_half_dists[i1] < first_half_dists[i2];});

            // second half dist
            for (int z = 0; z < kmeans_num_centroid; z++) {
                second_half_dists[z] = faiss::fvec_L2sqr_avx512(&querypoints[i][j * subspace_dimensionality + kmeans_dim], &centroids_list[(j * 2 + 1) * kmeans_num_centroid * kmeans_dim + z * kmeans_dim], kmeans_dim);
            }

            // second half sort
            iota(second_half_idx.begin(), second_half_idx.end(), 0);
            sort(second_half_idx.begin(), second_half_idx.end(), [&second_half_dists](int i1, int i2) {return second_half_dists[i1] < second_half_dists[i2];});

            // dynamic activate algorithm
            vector<pair<int, int>> retrieved_cell;
            dynamic_activate(indexes, retrieved_cell, first_half_dists, first_half_idx, second_half_dists, second_half_idx, collision_num, kmeans_num_centroid, j);
            // scalable_dynamic_activate(indexes, retrieved_cell, first_half_dists, first_half_idx, second_half_dists, second_half_idx, collision_num, kmeans_num_centroid, j);

            // Build collision bitmap for this subspace (parallel over retrieved_cells),
            // then update score layers. Atomic OR is needed because different point IDs
            // from different cells may map to the same uint64_t word.
            clear_collision_bitmap(lbsc);
            #pragma omp parallel for num_threads(number_of_threads)
            for (size_t z = 0; z < retrieved_cell.size(); z++) {
                auto it = indexes[j].find(retrieved_cell[z]);
                if (it != indexes[j].end()) {
                    for (size_t t = 0; t < it->second.size(); t++) {
                        int pid = it->second[t];
                        __sync_fetch_and_or(&lbsc.collision_bitmap[pid >> 6], 1ULL << (pid & 63));
                    }
                }
            }
            update_score_layers(lbsc);
        }

        // Extract candidates from highest-score layers downward
        vector<int> candidate_idx;
        candidate_idx.reserve(candidate_num + 1024);
        extract_candidates(lbsc, candidate_idx, candidate_num);

        vector<float> candidate_dists(candidate_idx.size());

        #pragma omp parallel for num_threads(number_of_threads)
        for (size_t j = 0; j < candidate_idx.size(); j++) {
            candidate_dists[j] = faiss::fvec_L2sqr_avx512(querypoints[i], dataset[candidate_idx[j]], data_dimensionality);
        }

        vector<int> candidate_sort_idx(candidate_idx.size());
        iota(candidate_sort_idx.begin(), candidate_sort_idx.end(), 0);
        partial_sort(candidate_sort_idx.begin(), candidate_sort_idx.begin() + k_size, candidate_sort_idx.end(), [&candidate_dists](int i1, int i2){return candidate_dists[i1] < candidate_dists[i2];});

        gettimeofday(&end_query, NULL);
        query_time += (1000000 * (end_query.tv_sec - start_query.tv_sec) + end_query.tv_usec - start_query.tv_usec);

        for (int j = 0; j < k_size; j++) {
            queryknn_results[i][j] = candidate_idx[candidate_sort_idx[j]];
        }

        reset_layered_bitmap(lbsc);

        ++pd_query;
    }
}


void dynamic_activate(vector<unordered_map<pair<int, int>, vector<int>, hash_pair>> &indexes, vector<pair<int, int>> &retrieved_cell, vector<float> &first_half_dists, vector<int> &first_half_idx, vector<float> &second_half_dists, vector<int> &second_half_idx, int collision_num, int kmeans_num_centroid, int subspace_idx) {
    vector<pair<float, int>> activated_cell;

    int retrieved_num = 0;
    activated_cell.push_back(pair<float, int>(first_half_dists[first_half_idx[0]] + second_half_dists[second_half_idx[0]], 0));
    while (true) {
        int cell_position = min_element(activated_cell.begin(), activated_cell.end()) - activated_cell.begin();
        auto iterator = indexes[subspace_idx].find(pair<int, int>(first_half_idx[cell_position], second_half_idx[activated_cell[cell_position].second]));
        if (iterator != indexes[subspace_idx].end() && activated_cell[cell_position].first < FLT_MAX) {
            retrieved_cell.push_back(pair<int, int>(first_half_idx[cell_position], second_half_idx[activated_cell[cell_position].second]));

            retrieved_num += iterator->second.size();

            if (retrieved_num >= collision_num) {
                break;
            }
        }

        if (activated_cell[cell_position].second == 0 && cell_position < kmeans_num_centroid - 1) {
            activated_cell.push_back(pair<float, int>(first_half_dists[first_half_idx[cell_position + 1]] + second_half_dists[second_half_idx[0]], 0));
        }

        if (activated_cell[cell_position].second < kmeans_num_centroid - 1) {
            activated_cell[cell_position].second++;
            activated_cell[cell_position].first = first_half_dists[first_half_idx[cell_position]] + second_half_dists[second_half_idx[activated_cell[cell_position].second]];
        } else {
            activated_cell[cell_position].first = FLT_MAX;
        }
    }
}

void scalable_dynamic_activate(vector<unordered_map<pair<int, int>, vector<int>, hash_pair>> &indexes, vector<pair<int, int>> &retrieved_cell, vector<float> &first_half_dists, vector<int> &first_half_idx, vector<float> &second_half_dists, vector<int> &second_half_idx, int collision_num, int kmeans_num_centroid, int subspace_idx) {
    priority_queue<pair<float, int>, vector<pair<float, int>>, Compare> activated_cell;
    vector<int> activated_idx(kmeans_num_centroid, 0);

    int retrieved_num = 0;
    activated_cell.push(pair<float, int>(first_half_dists[first_half_idx[0]] + second_half_dists[second_half_idx[0]], 0));
    while (true) {
        pair<float, int> selected_cell = activated_cell.top();
        int cell_position = selected_cell.second;
        auto iterator = indexes[subspace_idx].find(pair<int, int>(first_half_idx[cell_position], second_half_idx[activated_idx[cell_position]]));
        if (iterator != indexes[subspace_idx].end()) {
            retrieved_cell.push_back(pair<int, int>(first_half_idx[cell_position], second_half_idx[activated_idx[cell_position]]));

            retrieved_num += iterator->second.size();

            if (retrieved_num >= collision_num) {
                break;
            }
        }

        if (activated_idx[cell_position] == 0 && cell_position < kmeans_num_centroid - 1) {
            activated_cell.push(pair<float, int>(first_half_dists[first_half_idx[cell_position + 1]] + second_half_dists[second_half_idx[0]], cell_position + 1));
        }

        activated_cell.pop();

        if (cell_position < kmeans_num_centroid - 1) {
            activated_idx[cell_position]++;
            selected_cell.first = first_half_dists[first_half_idx[cell_position]] + second_half_dists[second_half_idx[activated_idx[cell_position]]];
            activated_cell.push(selected_cell);
        }
    }
}

