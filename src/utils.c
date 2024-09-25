#include "../include/utils.h"

#include <stdlib.h>
#include <math.h>

double rand_double_range(int min, int max) {
    double scale = rand() / (double) RAND_MAX;
    return min + scale * (max - min);
}

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}
