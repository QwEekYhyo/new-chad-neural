#include <utils.h>

#include <stdlib.h>
#include <math.h>

double rand_double_range(int min, int max) {
#if defined(_POSIX_VERSION)
    int n = random();
#else
    int n = rand();
#endif
    double scale = n / (double) RAND_MAX;
    return min + scale * (max - min);
}

double identity(double x) {
    return x;
}

double identity_derivative(double x) {
    return 1;
}

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// This is not actually the derivative of the sigmoid
// But it is going to be called on data that has already passed through the sigmoid
// Basically here we assume that x = sigmoid(y)
double sigmoid_derivative(double x) {
    return x * (1 - x);
}
