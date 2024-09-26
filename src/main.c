#include <matrix.h>
#include <time.h>
#include <vector.h>
#include <neural_network.h>
#include <utils.h>

#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include <stdlib.h>

void swap(int* a, int* b) {
    *a = *a ^ *b;
    *b = *a ^ *b;
    *a = *a ^ *b;
}

void shuffle(int* array, int size) {
    for (int i = size - 1; i > 0; i--) {
        int j = rand_double_range(0, i);
        swap(&array[i], &array[j]);
    }
}

double f(double x) {
    return 0.4 * x + 0.2;
}

int main(void) {
    struct timeval tm;
    gettimeofday(&tm, NULL);
    srandom(tm.tv_sec + tm.tv_usec * 1000000ul);

    size_t epochs = 1000;
    NeuralNetwork* nn = new_neural_network(1, 3, 1);
    set_batch_size(nn, 3);

    Matrix* input = new_uninitialized_matrix(1, 3);
    Matrix* expected_output = new_uninitialized_matrix(1, 3);

    int Xs[epochs];
    for (size_t i = 0; i < epochs; i++) {
        Xs[i] = i;
    }
    shuffle(Xs, epochs);
    
    double x;
    double y;
    for (size_t i = 0; i < 330; i++) {
        if (i % 100 == 0)
            printf("training epoch = %lu\n", i);

        for (size_t j = 0; j < 3; j++) {
            x = (double) Xs[i + j] / epochs;
            y = f(x);

            input->buffer[0][j] = x;
            expected_output->buffer[0][j] = y;
        }
        forward_pass(nn, input);
        back_propagation(nn, input, expected_output);
    }

    printf("testing training results:\n");
    input->buffer[0][0] = 0.1;
    input->buffer[0][1] = 0.5;
    input->buffer[0][2] = 0.73;
    forward_pass(nn, input);
    for (size_t i = 0; i < 3; i++) {
        printf("x = %f, f(x) = %f, model predicted : %f\n",
                input->buffer[0][i],
                f(input->buffer[0][i]),
                nn->output_layer->buffer[0][i]
        );
    }

    free_matrix(input);
    free_matrix(expected_output);
    free_neural_network(nn);
    return 0;
}
