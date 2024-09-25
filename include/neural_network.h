#ifndef NCN_NEURAL_NETWORK_H
#define NCN_NEURAL_NETWORK_H

#include "common_defs.h"
#include "vector.h"
#include "matrix.h"

typedef struct {
    Vector* input_layer;
    Vector* hidden_layer;
    Vector* output_layer;

    Matrix* input_hidden_weights;
    Matrix* hidden_output_weights;

    Vector* hidden_biases;
    Vector* output_biases;
} NeuralNetwork;

NeuralNetwork* new_neural_network(size_t num_inputs, size_t num_hidden, size_t num_outputs);
void free_neural_network(NeuralNetwork* nn);

void forward_pass(NeuralNetwork* nn, Vector* inputs);

#endif // NCN_NEURAL_NETWORK_H
