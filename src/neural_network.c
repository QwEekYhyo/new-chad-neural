#include "../include/neural_network.h"

#include <stdlib.h>

NeuralNetwork* new_neural_network(size_t num_inputs, size_t num_hidden, size_t num_outputs) {
    NeuralNetwork* new_nn = malloc(sizeof(NeuralNetwork));

    new_nn->input_layer = new_uninitialized_vector(num_inputs);
    new_nn->hidden_layer = new_uninitialized_vector(num_inputs);
    new_nn->output_layer = new_uninitialized_vector(num_inputs);

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
}
