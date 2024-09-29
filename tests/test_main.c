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
    int input_size = 1;
    int output_size = 1;
    size_t dataset_size = 100;
    double data[dataset_size][input_size];
    double output_data[dataset_size][output_size];
    for (size_t i = 0; i < dataset_size; i++) {
        data[i][0] = (double) i / dataset_size;
        output_data[i][0] = f(data[i][0]);
    }

    NeuralNetwork* nn = new_neural_network(input_size, 3, output_size);
    ModelTrainer trainer;
    trainer.nn = nn;
    trainer.batch_size = 10;
    trainer.epochs = 2000;

    double* loss_history = train_with_history(&trainer, data[0], output_data[0], dataset_size);

    printf("testing training results:\n");
    Matrix* input = new_uninitialized_matrix(1, 3);
    input->buffer[0][0] = 0.102;
    input->buffer[0][1] = 0.59;
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

    printf("loss history:\n");
    for (size_t i = 0; i < trainer.epochs; i++) {
        printf("%f, ", loss_history[i]);
    }
    printf("\n");

    free(loss_history);
    free_matrix(input);
    free_neural_network(nn);

    return 0;
}
