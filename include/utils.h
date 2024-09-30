#ifndef NCN_UTILS_H
#define NCN_UTILS_H

#include <stddef.h>

inline size_t max(size_t a, size_t b) {
    return a >= b ? a : b;
}

double rand_double_range(int min, int max);

double identity(double x);
double identity_derivative(double x);

double sigmoid(double x);
double sigmoid_derivative(double x);

#endif // NCN_UTILS_H
