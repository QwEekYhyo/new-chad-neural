#include <matrix.h>
#include <time.h>
#include <vector.h>
#include <neural_network.h>
#include <utils.h>
#include <model_trainer.h>

#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>

double f(double x) {
    return 0.4 * x + 0.2;
}

int main(void) {
    // Better randomization
    struct timeval tm;
    gettimeofday(&tm, NULL);
    srandom(tm.tv_sec + tm.tv_usec * 1000000ul);

    // Create dataset
    size_t dataset_size = 100;
    Vector* data[dataset_size];
    Vector* output_data[dataset_size];
    for (size_t i = 0; i < dataset_size; i++) {
        data[i] = new_uninitialized_vector(1);
        data[i]->buffer[0] = (double) i / dataset_size;
        output_data[i] = new_uninitialized_vector(1);
        output_data[i]->buffer[0] = f(data[i]->buffer[0]);
    }

    NeuralNetwork* nn = new_neural_network(1, 3, 1);
    ModelTrainer trainer;
    trainer.nn = nn;
    trainer.batch_size = 10;
    trainer.epochs = 2000;

    train(&trainer, data, output_data, dataset_size);

    for (size_t i = 0; i < dataset_size; i++) {
        free_vector(data[i]);
        free_vector(output_data[i]);
    }

    printf("testing training results:\n");
    Matrix* input = new_uninitialized_matrix(1, 3);
    input->buffer[0][0] = 0.1;
    input->buffer[0][1] = 0.5;
    input->buffer[0][2] = 0.73;
    set_batch_size(nn, 3);
    forward_pass(nn, input);
    for (size_t i = 0; i < 3; i++) {
        printf("x = %f, f(x) = %f, model predicted : %f\n",
                input->buffer[0][i],
                f(input->buffer[0][i]),
                nn->output_layer->buffer[0][i]
        );
    }

    free_matrix(input);
    free_neural_network(nn);

    return 0;
}
