#include <matrix.h>
#include <time.h>
#include <vector.h>
#include <neural_network.h>
#include <utils.h>
#include <model_trainer.h>

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#define INPUT_SIZE 1
#define OUTPUT_SIZE 1
#define DATASET_SIZE 1000

double f(double x) {
    return (sin(20.0 * x) + 1) / 2.0;
}

int main(void) {
    srand(43);

    // Create dataset
    double data[DATASET_SIZE][INPUT_SIZE];
    double output_data[DATASET_SIZE][OUTPUT_SIZE];
    for (size_t i = 0; i < DATASET_SIZE; i++) {
        data[i][0] = (double) i / DATASET_SIZE;
        output_data[i][0] = f(data[i][0]);
    }

    NeuralNetwork* nn = new_neural_network(INPUT_SIZE, 10, OUTPUT_SIZE);
    set_activation_functions(nn, sigmoid, sigmoid_derivative);
    ModelTrainer trainer;
    trainer.nn = nn;
    trainer.learning_rate = 0.8;
    trainer.batch_size = 10;
    trainer.epochs = 10000;

    double* loss_history = train_with_history(&trainer, data[0], output_data[0], DATASET_SIZE);

    Matrix* input = new_uninitialized_matrix(1, 10);
    set_batch_size(nn, 10);
    FILE *gnuplot = popen("gnuplot -persist", "w");
    fprintf(gnuplot, "plot '-'\n");
    for (size_t i = 0; i < 100; i++) {
        for (size_t j = 0; j < 10; j++) {
            input->buffer[0][j] = (i * 10 + j) / 1000.0;
        }
        forward_pass(nn, input);
        for (size_t j = 0; j < 10; j++) {
            input->buffer[0][j] = (i * 10 + j) / 1000.0;
            fprintf(gnuplot, "%f %f\n",
                    input->buffer[0][j] * 20,
                    nn->output_layer->buffer[0][j] * 2 - 1
            );
        }
    }
    fprintf(gnuplot, "e\n");
    fflush(gnuplot);

    free(loss_history);
    free_matrix(input);
    free_neural_network(nn);

    return 0;
}
