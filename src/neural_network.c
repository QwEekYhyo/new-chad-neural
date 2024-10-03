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

#include "matrix.h"
#include <neural_network.h>
#include <utils.h>

#include <stdlib.h>
#include <stdio.h>

NeuralNetwork* new_neural_network(size_t num_inputs, size_t num_hidden, size_t num_outputs) {
    NeuralNetwork* new_nn = malloc(sizeof(NeuralNetwork));

    new_nn->input_size = num_inputs;
    size_t default_batch_size = 1;
    new_nn->hidden_layer = new_uninitialized_matrix(num_hidden, default_batch_size);
    new_nn->output_layer = new_uninitialized_matrix(num_outputs, default_batch_size);

    new_nn->input_hidden_weights = new_random_matrix(num_hidden, num_inputs);
    new_nn->hidden_output_weights = new_random_matrix(num_outputs, num_hidden);

    new_nn->hidden_biases = new_random_vector(num_hidden);
    new_nn->output_biases = new_random_vector(num_outputs);

    set_hidden_activation_functions(new_nn, identity, identity_derivative); // default activation functions
    set_output_activation_functions(new_nn, identity, identity_derivative);
    new_nn->loss_function_derivative = mean_squared_error_derivative; // default loss_function_derivative

    return new_nn;
}

/* all the functions below might need some error checking maybe ? */

void free_neural_network(NeuralNetwork* nn) {
    free_matrix(nn->hidden_layer);
    free_matrix(nn->output_layer);

    free_matrix(nn->input_hidden_weights);
    free_matrix(nn->hidden_output_weights);

    free_vector(nn->hidden_biases);
    free_vector(nn->output_biases);

    free(nn);
}

void set_hidden_activation_functions(NeuralNetwork* nn, activation_function af, activation_function daf) {
    nn->hidden_layer_af = af;
    nn->hidden_layer_afd = daf;
}

void set_output_activation_functions(NeuralNetwork* nn, activation_function af, activation_function daf) {
    nn->output_layer_af = af;
    nn->output_layer_afd = daf;
}

void set_batch_size(NeuralNetwork* nn, size_t batch_size) {
    set_columns(nn->hidden_layer, batch_size);
    set_columns(nn->output_layer, batch_size);
}

void forward_pass(NeuralNetwork* nn, Matrix* inputs) {
    if (inputs->rows != nn->input_size) {
        printf(
                "Size of given inputs (%zu) doesn't match number of input nodes in neural network (%zu)\n",
                inputs->rows,
                nn->input_size
        );
        return;
    }

    if (inputs->columns != nn->output_layer->columns) {
        printf(
                "Batch size is not set correctly, got a batch of size %zu while NN is set to %zu\n",
                inputs->columns,
                nn->output_layer->columns
        );
        return;
    }

    // Input to hidden layer
    for (size_t b = 0; b < nn->hidden_layer->columns; b++) {
        for (size_t i = 0; i < nn->hidden_layer->rows; i++) {
            nn->hidden_layer->buffer[i][b] = nn->hidden_biases->buffer[i];
            for (size_t j = 0; j < nn->input_size; j++) {
                nn->hidden_layer->buffer[i][b] += inputs->buffer[j][b] * nn->input_hidden_weights->buffer[i][j];
            }
            nn->hidden_layer->buffer[i][b] = nn->hidden_layer_af(nn->hidden_layer->buffer[i][b]);
        }
    }

    // Hidden to output layer
    for (size_t b = 0; b < nn->output_layer->columns; b++) {
        for (size_t i = 0; i < nn->output_layer->rows; i++) {
            nn->output_layer->buffer[i][b] = nn->output_biases->buffer[i];
            for (size_t j = 0; j < nn->hidden_layer->rows; j++) {
                nn->output_layer->buffer[i][b] += nn->hidden_layer->buffer[j][b] * nn->hidden_output_weights->buffer[i][j];
            }
            nn->output_layer->buffer[i][b] = nn->output_layer_af(nn->output_layer->buffer[i][b]);
        }
    }
}

// forward_pass needs to be called before this function
void back_propagation(NeuralNetwork* nn, Matrix* inputs, Matrix* targets, double learning_rate) {
    // Calculate output error
    Matrix* output_errors = new_uninitialized_matrix(nn->output_layer->rows, nn->output_layer->columns);
    for (size_t b = 0; b < nn->output_layer->columns; b++) {
        for (size_t o = 0; o < nn->output_layer->rows; o++) {
            double current_output = nn->output_layer->buffer[o][b];
            output_errors->buffer[o][b] =
                nn->loss_function_derivative(targets->buffer[o][b], current_output)
                * nn->output_layer_afd(current_output);
        }
    }

    // Calculate hidden layer error
    Matrix* hidden_errors = new_uninitialized_matrix(nn->hidden_layer->rows, nn->hidden_layer->columns);
    for (size_t b = 0; b < nn->hidden_layer->columns; b++) {
        for (size_t h = 0; h < nn->hidden_layer->rows; h++) {
            hidden_errors->buffer[h][b] = 0;
            for (size_t j = 0; j < nn->output_layer->rows; j++) {
                hidden_errors->buffer[h][b] += output_errors->buffer[j][b] * nn->hidden_output_weights->buffer[j][h];
            }
            hidden_errors->buffer[h][b] *= nn->hidden_layer_afd(nn->hidden_layer->buffer[h][b]);
        }
    }

    // Update hidden to output weights and output biases
    for (size_t o = 0; o < nn->output_layer->rows; o++) {
        for (size_t h = 0; h < nn->hidden_layer->rows; h++) {
            double weight_update = 0.0;
            for (size_t b = 0; b < nn->output_layer->columns; b++) {
                weight_update += output_errors->buffer[o][b] * nn->hidden_layer->buffer[h][b];
            }
            nn->hidden_output_weights->buffer[o][h] += learning_rate * weight_update / nn->output_layer->columns; // average over batch
        }
        // Update output biases (biases are shared across batch examples, so sum the errors)
        double bias_update = 0.0;
        for (size_t b = 0; b < nn->output_layer->columns; b++) {
            bias_update += output_errors->buffer[o][b];
        }
        nn->output_biases->buffer[o] += learning_rate * bias_update / nn->output_layer->columns; // average over batch
    }


    // Update input to hidden weights and hidden biases
    for (size_t h = 0; h < nn->hidden_layer->rows; h++) {
        for (size_t i = 0; i < nn->input_size; i++) {
            double weight_update = 0.0;
            for (size_t b = 0; b < nn->hidden_layer->columns; b++) {
                weight_update += hidden_errors->buffer[h][b] * inputs->buffer[i][b];
            }
            nn->input_hidden_weights->buffer[h][i] += learning_rate * weight_update / nn->hidden_layer->columns; // average over batch
        }
        // Update hidden biases
        double bias_update = 0.0;
        for (size_t b = 0; b < nn->hidden_layer->columns; b++) {
            bias_update += hidden_errors->buffer[h][b];
        }
        nn->hidden_biases->buffer[h] += learning_rate * bias_update / nn->hidden_layer->columns; // average over batch
    }

    free_matrix(output_errors);
    free_matrix(hidden_errors);
}

int save_neural_network(NeuralNetwork* nn, const char* filename) {
    // No error checking for remove() because:
    // - if the file doesn't exist -> I don't care + ratio
    // - if permissions aren't sufficient -> fopen will fail
    // - if file contains invalid characters -> fopen will fail
    remove(filename);

    FILE* save_file = fopen(filename, "a");
    if (!save_file) {
        printf("Could not open file to save Neural Network\n");
        return -1;
    }

    save_matrix(nn->input_hidden_weights, save_file);
    save_matrix(nn->hidden_output_weights, save_file);

    save_vector(nn->hidden_biases, save_file);
    save_vector(nn->output_biases, save_file);

    fclose(save_file);

    return 0;
}

NeuralNetwork* new_neural_network_from_file(const char* filename) {
    NeuralNetwork* new_nn = malloc(sizeof(NeuralNetwork));

    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Could not open file for reading to load Neural Network\n");
        return NULL;
    }

    new_nn->input_hidden_weights = new_matrix_from_file(file);
    new_nn->hidden_output_weights = new_matrix_from_file(file);

    new_nn->hidden_biases = new_vector_from_file(file);
    new_nn->output_biases = new_vector_from_file(file);

    new_nn->input_size = new_nn->input_hidden_weights->columns;

    size_t default_batch_size = 1;
    new_nn->hidden_layer = new_uninitialized_matrix(new_nn->hidden_biases->size, default_batch_size);
    new_nn->output_layer = new_uninitialized_matrix(new_nn->output_biases->size, default_batch_size);

    set_hidden_activation_functions(new_nn, identity, identity_derivative); // default activation functions
    set_output_activation_functions(new_nn, identity, identity_derivative);
    new_nn->loss_function_derivative = mean_squared_error_derivative; // default loss_function_derivative

    fclose(file);

    return new_nn;
}
