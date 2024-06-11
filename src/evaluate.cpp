#include "evaluate.h"

void recall_and_ratio(float ** &dataset, float ** &querypoints, int data_dimensionality, int ** &queryknn_results, long int ** &gt, int query_size) {
    int ks[6] = {1, 10, 20, 30, 40, 50};
    
    for (int k_index = 0; k_index < sizeof(ks) / sizeof(ks[0]); k_index++) {
        int retrived_data_num = 0;

        for (int i = 0; i < query_size; i++)
        {
            for (int j = 0; j < ks[k_index]; j++)
            {
                for (int z = 0; z < ks[k_index]; z++) {
                    if (queryknn_results[i][j] == gt[i][z]) {
                        retrived_data_num++;
                        break;
                    }
                }
            }
        }

        float ratio = 0.0f;
        for (int i = 0; i < query_size; i++)
        {
            for (int j = 0; j < ks[k_index]; j++)
            {
                float groundtruth_square_dist = euclidean_distance(querypoints[i], dataset[gt[i][j]], data_dimensionality);
                float otbained_square_dist = euclidean_distance(querypoints[i], dataset[queryknn_results[i][j]], data_dimensionality);
                if (groundtruth_square_dist == 0) {
                    ratio += 1.0f;
                } else {
                    ratio += sqrt(otbained_square_dist) / sqrt(groundtruth_square_dist);
                }
            }
        }

        float recall_value = float(retrived_data_num) / (query_size * ks[k_index]);
        float overall_ratio = ratio / (query_size * ks[k_index]);

        cout << "When k = " << ks[k_index] << ", (recall, ratio) = (" << recall_value << ", " << overall_ratio << ")" << endl;
    }
}