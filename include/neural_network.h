#ifndef NCN_NEURAL_NETWORK_H
#define NCN_NEURAL_NETWORK_H

#include <common_defs.h>
#include <vector.h>
#include <matrix.h>

static const double LEARNING_RATE = 0.01;

typedef struct {
    size_t input_size;
    Vector* hidden_layer;
    Vector* output_layer;

    Matrix* input_hidden_weights;
    Matrix* hidden_output_weights;

    Vector* hidden_biases;
    Vector* output_biases;

    activation_function activation_function;
    activation_function activation_function_derivative;
} NeuralNetwork;

NeuralNetwork* new_neural_network(size_t num_inputs, size_t num_hidden, size_t num_outputs);
void free_neural_network(NeuralNetwork* nn);

void set_activation_functions(NeuralNetwork* nn, activation_function af, activation_function daf);

void forward_pass(NeuralNetwork* nn, Vector* inputs);
void back_propagation(NeuralNetwork* nn, Vector* inputs, Vector* expected_outputs);

#endif // NCN_NEURAL_NETWORK_H
