#pragma once
#include <iostream>
#include <math.h>
#include "dist_calculation.h"

void recall_and_ratio(float ** &dataset, float ** &querypoints, int data_dimensionality, int ** &queryknn_results, long int ** &gt, int query_size);

using namespace std;