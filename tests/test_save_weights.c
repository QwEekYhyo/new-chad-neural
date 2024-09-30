#include <matrix.h>
#include <vector.h>
#include <neural_network.h>
#include <utils.h>

#include <stdlib.h>
#include <stdio.h>

#define INPUT_SIZE 3
#define OUTPUT_SIZE 2
#define DATASET_SIZE 100

double f(double x) {
    return -0.8 * x + 0.9;
}

int main(void) {
    // Better randomization
    srand(69);

    /*
    // Create dataset
    double data[DATASET_SIZE][INPUT_SIZE];
    double output_data[DATASET_SIZE][OUTPUT_SIZE];
    for (size_t i = 0; i < DATASET_SIZE; i++) {
        data[i][0] = (double) i / DATASET_SIZE;
        output_data[i][0] = f(data[i][0]);
    }
    */

    NeuralNetwork* nn = new_neural_network(INPUT_SIZE, 3, OUTPUT_SIZE);

    printf("Wih:\n");
    print_matrix(nn->input_hidden_weights);
    printf("Who:\n");
    print_matrix(nn->hidden_output_weights);
    printf("Bh:\n");
    print_vector(nn->hidden_biases);
    printf("Bo:\n");
    print_vector(nn->output_biases);

    free_neural_network(nn);

    return 0;
}
