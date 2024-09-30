#ifndef NCN_UTILS_H
#define NCN_UTILS_H

#include <math.h>
#include <stddef.h>
#include <stdbool.h>

#define EPSILON 0.000000000000001

inline size_t max(size_t a, size_t b) {
    return a >= b ? a : b;
}

inline bool are_double_equals(double a, double b) {
    return fabs(a - b) < EPSILON;
}

double rand_double_range(int min, int max);

double identity(double x);
double identity_derivative(double x);

double sigmoid(double x);
double sigmoid_derivative(double x);

#endif // NCN_UTILS_H
