#include "../include/matrix.h"
#include "../include/vector.h"
#include "../include/neural_network.h"

#include <stdio.h>

int main(void) {
    Matrix* m = new_random_matrix(3, 3);
    print_matrix(m);

    free_matrix(m);

    printf("==== Vectors ====\n");
    Vector* v = new_random_vector(4);
    print_vector(v);
    Vector* v1 = new_random_vector(4);
    print_vector(v1);

    free_vector(v);
    free_vector(v1);

    printf("==== Neural Network ====\n");
    NeuralNetwork* nn = new_neural_network(3, 2, 3);
    print_matrix(nn->input_hidden_weights);

    free_neural_network(nn);
    return 0;
}
