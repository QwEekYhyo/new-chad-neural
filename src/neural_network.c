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
        printf(
                "Size of given inputs (%lu) doesn't match number of input nodes in neural network (%lu)",
                inputs->rows,
                nn->input_size
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
            nn->hidden_layer->buffer[i][b] = nn->activation_function(nn->hidden_layer->buffer[i][b]);
        }
    }

    // Hidden to output layer
    for (size_t b = 0; b < nn->hidden_layer->columns; b++) {
        for (size_t i = 0; i < nn->output_layer->rows; i++) {
            nn->output_layer->buffer[i][b] = nn->output_biases->buffer[i];
            for (size_t j = 0; j < nn->hidden_layer->rows; j++) {
                nn->output_layer->buffer[i][b] += nn->hidden_layer->buffer[j][b] * nn->hidden_output_weights->buffer[i][j];
            }
            nn->output_layer->buffer[i][b] = nn->activation_function(nn->output_layer->buffer[i][b]);
        }
    }
}

/*
// forward_pass needs to be called before this function
void back_propagation(NeuralNetwork* nn, Vector* inputs, Vector* targets) {
    // Calculate output error
    Vector* output_errors = new_uninitialized_vector(nn->output_layer->size);
    for (size_t i = 0; i < nn->output_layer->size; i++) {
        double current_output = nn->output_layer->buffer[i];
        output_errors->buffer[i] = (targets->buffer[i] - current_output) * nn->activation_function_derivative(current_output);
    }

    // Calculate hidden layer error
    Vector* hidden_errors = new_uninitialized_vector(nn->hidden_layer->size);
    for (size_t i = 0; i < nn->hidden_layer->size; i++) {
        hidden_errors->buffer[i] = 0;
        for (size_t j = 0; j < nn->output_layer->size; j++) {
            hidden_errors->buffer[i] += output_errors->buffer[j] * nn->hidden_output_weights->buffer[j][i];
        }
        hidden_errors->buffer[i] *= nn->activation_function_derivative(nn->hidden_layer->buffer[i]);
    }

    // Update hidden to output weights and output biases
    for (size_t i = 0; i < nn->output_layer->size; i++) {
        for (size_t j = 0; j < nn->hidden_layer->size; j++) {
            nn->hidden_output_weights->buffer[i][j] += LEARNING_RATE * output_errors->buffer[i] * nn->hidden_layer->buffer[j];
        }
        nn->output_biases->buffer[i] += LEARNING_RATE * output_errors->buffer[i];
    }

    // Update input to hidden weights and hidden biases
    for (size_t i = 0; i < nn->hidden_layer->size; i++) {
        for (size_t j = 0; j < nn->input_size; j++) {
            nn->input_hidden_weights->buffer[i][j] += LEARNING_RATE * hidden_errors->buffer[i] * inputs->buffer[j];
        }
        nn->hidden_biases->buffer[i] += LEARNING_RATE * hidden_errors->buffer[i];
    }

    free_vector(output_errors);
    free_vector(hidden_errors);
}
*/
