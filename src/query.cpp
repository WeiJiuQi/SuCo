#include "query.h"

void ann_query(float ** &dataset, int ** &queryknn_results, long int dataset_size, int data_dimensionality, int query_size, int k_size, float ** &querypoints, vector<unordered_map<pair<int, int>, vector<int>, hash_pair>> &indexes, float * &centroids_list, int subspace_num, int subspace_dimensionality, int kmeans_num_centroid, int kmeans_dim, int collision_num, int candidate_num, int number_of_threads, long int &query_time) {
    struct timeval start_query, end_query;
    
    progress_display pd_query(query_size);

    vector<unsigned char> collision_count(dataset_size, 0);
    
    for (int i = 0; i < query_size; i++) {
        gettimeofday(&start_query, NULL);

        for (int j = 0; j < subspace_num; j++) {
            // first half dist
            vector<float> first_half_dists(kmeans_num_centroid);
            for (int z = 0; z < kmeans_num_centroid; z++) {
                first_half_dists[z] = euclidean_distance(&querypoints[i][j * subspace_dimensionality], &centroids_list[j * 2 * kmeans_num_centroid * kmeans_dim + z * kmeans_dim], kmeans_dim);
            }

            // first half sort
            vector<int> first_half_idx(kmeans_num_centroid);
            iota(first_half_idx.begin(), first_half_idx.end(), 0);
            sort(first_half_idx.begin(), first_half_idx.end(), [&first_half_dists](int i1, int i2) {return first_half_dists[i1] < first_half_dists[i2];});

            // second half dist
            vector<float> second_half_dists(kmeans_num_centroid);
            for (int z = 0; z < kmeans_num_centroid; z++) {
                second_half_dists[z] = euclidean_distance(&querypoints[i][j * subspace_dimensionality + kmeans_dim], &centroids_list[(j * 2 + 1) * kmeans_num_centroid * kmeans_dim + z * kmeans_dim], kmeans_dim);
            }

            // second half sort
            vector<int> second_half_idx(kmeans_num_centroid);
            iota(second_half_idx.begin(), second_half_idx.end(), 0);
            sort(second_half_idx.begin(), second_half_idx.end(), [&second_half_dists](int i1, int i2) {return second_half_dists[i1] < second_half_dists[i2];});


            // dynamic activate algorithm
            vector<pair<int, int>> retrieved_cell;
            dynamic_activate(indexes, retrieved_cell, first_half_dists, first_half_idx, second_half_dists, second_half_idx, collision_num, kmeans_num_centroid, j);

            // count collision, parallelization here is recommended for large datasets (greater than 10 million) rather than small datasets (less than 1 million)
            #pragma omp parallel for num_threads(number_of_threads)
            for (int z = 0; z < retrieved_cell.size(); z++) {
                auto iterator = indexes[j].find(retrieved_cell[z]);
                for (int t = 0; t < iterator->second.size(); t++) {
                    collision_count[iterator->second[t]]++;
                }
            }
        }

        int * collision_num_count = new int[subspace_num + 1]();
        int ** local_collision_num_count = new int * [number_of_threads];
        for (int j = 0; j < number_of_threads; j++) {
            local_collision_num_count[j] = new int [subspace_num + 1]();
        }

        #pragma omp parallel for num_threads(number_of_threads)
        for (int j = 0; j < dataset_size; j++) {
            int id = omp_get_thread_num();
            local_collision_num_count[id][collision_count[j]]++;
        }

        for (int j = 0; j < subspace_num + 1; j++) {
            for (int z = 0; z < number_of_threads; z++) {
                collision_num_count[j] += local_collision_num_count[z][j];
            }
        }

        for (int j = 0; j < number_of_threads; j++) {
            delete[] local_collision_num_count[j];
        }
        delete[] local_collision_num_count;

        // release the candidate number to include all points in last_collision_num, saving the time for checking points whose collision_num_count is last_collision_num
        int last_collision_num;
        int sum_candidate = 0;
        for (int j = subspace_num; j >= 0; j--) {
            if (collision_num_count[j] <= candidate_num - sum_candidate) {
                sum_candidate += collision_num_count[j];
            } else {
                last_collision_num = j;
                break;
            }
        }
        delete[] collision_num_count;

        vector<int> candidate_idx;
        vector<vector<int>> local_candidate_idx(number_of_threads);
        // vector<vector<int>> local_boundary_candidate_idx(number_of_threads);

        #pragma omp parallel for num_threads(number_of_threads)
        for (int j = 0; j < dataset_size; j++) {
            int id = omp_get_thread_num();
            if (collision_count[j] >= last_collision_num) {
                local_candidate_idx[id].push_back(j);
            }
        }

        // #pragma omp parallel for num_threads(number_of_threads)
        // for (int j = 0; j < dataset_size; j++) {
        //     int id = omp_get_thread_num();
        //     if (collision_count[j] > last_collision_num) {
        //         local_candidate_idx[id].push_back(j);
        //     } else if (collision_count[j] == last_collision_num) {
        //         local_boundary_candidate_idx[id].push_back(j);
        //     }
        // }

        for (int j = 0; j < number_of_threads; j++) {
            candidate_idx.insert(candidate_idx.end(), local_candidate_idx[j].begin(), local_candidate_idx[j].end());
        }

        // for (int j = 0; j < number_of_threads; j++) {
        //     if (candidate_num - candidate_idx.size() >= local_boundary_candidate_idx[j].size()) {
        //         candidate_idx.insert(candidate_idx.end(), local_boundary_candidate_idx[j].begin(), local_boundary_candidate_idx[j].end());
        //     } else {
        //         candidate_idx.insert(candidate_idx.end(), local_boundary_candidate_idx[j].begin(), local_boundary_candidate_idx[j].begin() + (candidate_num - candidate_idx.size() + 1));
        //         break;
        //     }
        // }

        vector<float> candidate_dists(candidate_idx.size());

        #pragma omp parallel for num_threads(number_of_threads)
        for (int j = 0; j < candidate_idx.size(); j++) {
            // candidate_dists[j] = euclidean_distance(querypoints[i], dataset[candidate_idx[j]], data_dimensionality);
            // candidate_dists[j] = euclidean_distance_SIMD(querypoints[i], dataset[candidate_idx[j]], data_dimensionality);
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

        fill(collision_count.begin(), collision_count.end(), 0);

        // cout << "Finish the " << i + 1 << "-th query." << endl;
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
