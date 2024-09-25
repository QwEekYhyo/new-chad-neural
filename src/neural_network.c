#include "../include/neural_network.h"
#include "../include/utils.h"

#include <stdlib.h>
#include <stdio.h>

NeuralNetwork* new_neural_network(size_t num_inputs, size_t num_hidden, size_t num_outputs) {
    NeuralNetwork* new_nn = malloc(sizeof(NeuralNetwork));

    new_nn->input_layer = new_uninitialized_vector(num_inputs);
    new_nn->hidden_layer = new_uninitialized_vector(num_hidden);
    new_nn->output_layer = new_uninitialized_vector(num_outputs);

    new_nn->input_hidden_weights = new_random_matrix(num_hidden, num_inputs);
    new_nn->hidden_output_weights = new_random_matrix(num_outputs, num_hidden);

    new_nn->hidden_biases = new_random_vector(num_hidden);
    new_nn->output_biases = new_random_vector(num_outputs);

    return new_nn;
}

void free_neural_network(NeuralNetwork* nn) {
    free_vector(nn->input_layer);
    free_vector(nn->hidden_layer);
    free_vector(nn->output_layer);

    free_matrix(nn->input_hidden_weights);
    free_matrix(nn->hidden_output_weights);

    free_vector(nn->hidden_biases);
    free_vector(nn->output_biases);

    free(nn);
}

void forward_pass(NeuralNetwork* nn, Vector* inputs) {
    if (inputs->size != nn->input_layer->size) {
        printf(
                "Number of given inputs (%lu) doesn't match number of input nodes in neural network (%lu)",
                inputs->size,
                nn->input_layer->size
        );
        return;
    }

    // Input to hidden layer
    for (size_t i = 0; i < nn->hidden_layer->size; i++) {
        nn->hidden_layer->buffer[i] = 0;
        for (size_t j = 0; j < nn->input_layer->size; j++) {
            nn->hidden_layer->buffer[i] += inputs->buffer[j] * nn->input_hidden_weights->buffer[i][j];
        }
        nn->hidden_layer->buffer[i] += nn->hidden_biases->buffer[i];
        nn->hidden_layer->buffer[i] = sigmoid(nn->hidden_layer->buffer[i]);
    }

    // Hidden to output layer
    for (size_t i = 0; i < nn->output_layer->size; i++) {
        nn->output_layer->buffer[i] = 0;
        for (size_t j = 0; j < nn->hidden_layer->size; j++) {
            nn->output_layer->buffer[i] += nn->hidden_layer->buffer[j] * nn->hidden_output_weights->buffer[i][j];
        }
        nn->output_layer->buffer[i] += nn->output_biases->buffer[i];
        nn->output_layer->buffer[i] = sigmoid(nn->output_layer->buffer[i]);
    }
}
