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

    set_activation_functions(new_nn, identity, identity_derivative);

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

void set_activation_functions(NeuralNetwork* nn, activation_function af, activation_function daf) {
    nn->activation_function = af;
    nn->activation_function_derivative = daf;
}

void set_batch_size(NeuralNetwork* nn, size_t batch_size) {
    size_t num_hidden = nn->hidden_layer->rows;
    size_t num_outputs = nn->output_layer->rows;
    free_matrix(nn->hidden_layer);
    free_matrix(nn->output_layer);
    nn->hidden_layer = new_uninitialized_matrix(num_hidden, batch_size);
    nn->output_layer = new_uninitialized_matrix(num_outputs, batch_size);
}

void forward_pass(NeuralNetwork* nn, Matrix* inputs) {
    if (inputs->rows != nn->input_size) {
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
        printf(
                "Size of given inputs (%zu) doesn't match number of input nodes in neural network (%zu)\n",
                inputs->rows,
                nn->input_size
        );
#else
        printf(
                "Size of given inputs (%lu) doesn't match number of input nodes in neural network (%lu)\n",
                inputs->rows,
                nn->input_size
        );
#endif
        return;
    }

    if (inputs->columns != nn->output_layer->columns) {
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
        printf(
                "Batch size is not set correctly, got a batch of size %zu while NN is set to %zu\n",
                inputs->columns,
                nn->output_layer->columns
        );
#else
        printf(
                "Batch size is not set correctly, got a batch of size %lu while NN is set to %lu\n",
                inputs->columns,
                nn->output_layer->columns
        );
#endif
        return;
    }

    // Input to hidden layer
    for (size_t b = 0; b < nn->hidden_layer->columns; b++) {
        for (size_t i = 0; i < nn->hidden_layer->rows; i++) {
            nn->hidden_layer->buffer[i][b] = nn->hidden_biases->buffer[i];
            for (size_t j = 0; j < nn->input_size; j++) {
                nn->hidden_layer->buffer[i][b] += inputs->buffer[j][b] * nn->input_hidden_weights->buffer[i][j];
            }
            nn->hidden_layer->buffer[i][b] = nn->activation_function(nn->hidden_layer->buffer[i][b]);
        }
    }

    // Hidden to output layer
    for (size_t b = 0; b < nn->output_layer->columns; b++) {
        for (size_t i = 0; i < nn->output_layer->rows; i++) {
            nn->output_layer->buffer[i][b] = nn->output_biases->buffer[i];
            for (size_t j = 0; j < nn->hidden_layer->rows; j++) {
                nn->output_layer->buffer[i][b] += nn->hidden_layer->buffer[j][b] * nn->hidden_output_weights->buffer[i][j];
            }
            nn->output_layer->buffer[i][b] = nn->activation_function(nn->output_layer->buffer[i][b]);
        }
    }
}

// forward_pass needs to be called before this function
void back_propagation(NeuralNetwork* nn, Matrix* inputs, Matrix* targets) {
    // Calculate output error
    Matrix* output_errors = new_uninitialized_matrix(nn->output_layer->rows, nn->output_layer->columns);
    for (size_t b = 0; b < nn->output_layer->columns; b++) {
        for (size_t o = 0; o < nn->output_layer->rows; o++) {
            double current_output = nn->output_layer->buffer[o][b];
            output_errors->buffer[o][b] = (targets->buffer[o][b] - current_output) * nn->activation_function_derivative(current_output);
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
            hidden_errors->buffer[h][b] *= nn->activation_function_derivative(nn->hidden_layer->buffer[h][b]);
        }
    }

    // Update hidden to output weights and output biases
    for (size_t o = 0; o < nn->output_layer->rows; o++) {
        for (size_t h = 0; h < nn->hidden_layer->rows; h++) {
            double weight_update = 0.0;
            for (size_t b = 0; b < nn->output_layer->columns; b++) {
                weight_update += output_errors->buffer[o][b] * nn->hidden_layer->buffer[h][b];
            }
            nn->hidden_output_weights->buffer[o][h] += LEARNING_RATE * weight_update / nn->output_layer->columns; // average over batch
        }
        // Update output biases (biases are shared across batch examples, so sum the errors)
        double bias_update = 0.0;
        for (size_t b = 0; b < nn->output_layer->columns; b++) {
            bias_update += output_errors->buffer[o][b];
        }
        nn->output_biases->buffer[o] += LEARNING_RATE * bias_update / nn->output_layer->columns; // average over batch
    }


    // Update input to hidden weights and hidden biases
    for (size_t h = 0; h < nn->hidden_layer->rows; h++) {
        for (size_t i = 0; i < nn->input_size; i++) {
            double weight_update = 0.0;
            for (size_t b = 0; b < nn->hidden_layer->columns; b++) {
                weight_update += hidden_errors->buffer[h][b] * inputs->buffer[i][b];
            }
            nn->input_hidden_weights->buffer[h][i] += LEARNING_RATE * weight_update / nn->hidden_layer->columns; // average over batch
        }
        // Update hidden biases
        double bias_update = 0.0;
        for (size_t b = 0; b < nn->hidden_layer->columns; b++) {
            bias_update += hidden_errors->buffer[h][b];
        }
        nn->hidden_biases->buffer[h] += LEARNING_RATE * bias_update / nn->hidden_layer->columns; // average over batch
    }

    free_matrix(output_errors);
    free_matrix(hidden_errors);
}
