/* New Chad Neural - C library to train neural networks
 * Copyright (C) 2024 Lucas Logan
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include <utils.h>

#include <stdlib.h>

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

double mean_squared_error(double target, double output) {
    return (target - output) * (target - output);
}

double mean_squared_error_derivative(double target, double output) {
    return target - output;
}

double binary_cross_entropy(double target, double output) {
    return - (target * log(output) + (1 - target) * log(1 - output));
}

double binary_cross_entropy_derivative(double target, double output) {
    return (output - target) / (output * (1 - output));
}
