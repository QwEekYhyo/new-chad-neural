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

#ifndef NCN_FUNCTIONS_H
#define NCN_FUNCTIONS_H

#include <stddef.h>
#include <stdbool.h>
#include <matrix.h>

/***** Activation functions *****/
double sigmoid(double x);
/*
 * This is not actually the derivative of the sigmoid
 * But it is going to be called on data that has already passed through the sigmoid
 * Basically here we assume that x = sigmoid(y)
 */
double sigmoid_derivative(double x);

void softmax(Matrix* output);

/***** Loss functions *****/
double squared_error(double target, double output);
double squared_error_derivative(double target, double output);

double binary_cross_entropy(double target, double output);
double binary_cross_entropy_derivative(double target, double output);

double categorical_cross_entropy(double target, double output);
double categorical_cross_entropy_derivative(double target, double output);

#endif // NCN_FUNCTIONS_H
