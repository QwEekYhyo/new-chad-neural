#include "../include/utils.h"

#include <stdlib.h>

double rand_double_range(int min, int max) {
    double scale = rand() / (double) RAND_MAX;
    return min + scale * (max - min);
}
