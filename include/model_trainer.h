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

#ifndef NCN_MODEL_TRAINER_H
#define NCN_MODEL_TRAINER_H

#include <neural_network.h>
#include <common_defs.h>

#include <stddef.h>
#include <stdint.h>

typedef struct {
    NeuralNetwork* nn;

    double learning_rate;
    size_t epochs;
    size_t batch_size;

    loss_function loss_function;
} ModelTrainer;

int set_loss_function(ModelTrainer* trainer, enum LossFunction loss);

// train_data & train_output HAVE to be sized just like the neural network input & output
// or else consequences
void _train(ModelTrainer* trainer, double* train_data, double* train_output, size_t dataset_size, uint_least8_t with_history, double** loss_history);

void train(ModelTrainer* trainer, double* train_data, double* train_output, size_t dataset_size);
double* train_with_history(ModelTrainer* trainer, double* train_data, double* train_output, size_t dataset_size);

/* This is recommended for large datasets with a lot of input and/or output nodes,
 * as no allocation occurs in these functions except for the loss history
 * => dataset is only allocated once and not "twice" like the functions above
 */
void _train_bare(ModelTrainer* trainer, double* train_data, double* train_output, size_t dataset_size, uint_least8_t with_history, double** loss_history);

void train_bare(ModelTrainer* trainer, double* train_data, double* train_output, size_t dataset_size);
double* train_with_history_bare(ModelTrainer* trainer, double* train_data, double* train_output, size_t dataset_size);

#endif // NCN_MODEL_TRAINER_H
