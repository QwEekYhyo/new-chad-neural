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

#include <functions.h>

#include <math.h>
#include <stdio.h>

static const double MINIMUM_EPSILON = 1e-12;

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// This is not actually the derivative of the sigmoid
// But it is going to be called on data that has already passed through the sigmoid
// Basically here we assume that x = sigmoid(y)
double sigmoid_derivative(double x) {
    return x * (1 - x);
}

void softmax(Matrix* output) {
    for (size_t b = 0; b < output->columns; b++) {
        double max_val = MAT(output, 0, b);
        for (size_t r = 1; r < output->rows; r++) {
            if (MAT(output, r, b) > max_val)
                max_val = MAT(output, r, b);
        }

        double sum = 0.0;
        for (size_t r = 0; r < output->rows; r++) {
            MAT(output, r, b) = exp(MAT(output, r, b) - max_val);
            sum += MAT(output, r, b);
        }

        for (size_t r = 0; r < output->rows; r++) {
            MAT(output, r, b) /= sum;
        }
    }
}

double squared_error(double target, double output) {
    return (target - output) * (target - output);
}

double squared_error_derivative(double target, double output) {
    return 2 * (output - target);
}

double binary_cross_entropy(double target, double output) {
    // The fmin fmax shit is to avoid log(0) and log(1)
    // it doesn't seem to add performance overhead
    output = fmin(fmax(output, MINIMUM_EPSILON), 1.0 - MINIMUM_EPSILON);
    return - (target * log(output) + (1 - target) * log(1 - output));
}

double binary_cross_entropy_derivative(double target, double output) {
    return output - target;
}

/* I have actually no clue if the formulas below are correct xd */
double categorical_cross_entropy(double target, double output) {
    if (target == 1.0) {
        // The fmin fmax shit is to avoid log(0)
        // it doesn't seem to add performance overhead
        output = fmin(fmax(output, MINIMUM_EPSILON), 1.0 - MINIMUM_EPSILON);
        return -log(output);
    }
    return 0;
}

double categorical_cross_entropy_derivative(double target, double output) {
    return output - target;
}
