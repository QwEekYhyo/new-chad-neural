/* New Chad Neural - C library to train neural networks
 * Copyright (C) 2024 Lucas Logan
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include <matrix.h>
#include <vector.h>
#include <neural_network.h>
#include <utils.h>
#include <model_trainer.h>

#include <stdlib.h>
#include <stdio.h>
#if defined(_POSIX_VERSION)
#include <sys/time.h>
#else
#include <time.h>
#endif

#define INPUT_SIZE 1
#define OUTPUT_SIZE 1
#define DATASET_SIZE 100

double f(double x) {
    return 0.4 * x + 0.2;
}

int main(void) {
    // Better randomization
#if defined(_POSIX_VERSION)
    struct timeval tm;
    gettimeofday(&tm, NULL);
    srandom(tm.tv_sec + tm.tv_usec * 1000000ul);
#else
    srand((unsigned int) time(NULL));
#endif

    // Create dataset
    double data[DATASET_SIZE][INPUT_SIZE];
    double output_data[DATASET_SIZE][OUTPUT_SIZE];
    for (size_t i = 0; i < DATASET_SIZE; i++) {
        data[i][0] = (double) i / DATASET_SIZE;
        output_data[i][0] = f(data[i][0]);
    }

    NeuralNetwork* nn = new_neural_network(INPUT_SIZE, 3, OUTPUT_SIZE);
    ModelTrainer trainer;
    trainer.nn = nn;
    trainer.batch_size = 10;
    trainer.epochs = 2000;

    double* loss_history = train_with_history(&trainer, data[0], output_data[0], DATASET_SIZE);

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
    for (size_t i = 0; i < trainer.epochs; i++)
        printf("%f, ", loss_history[i]);
    printf("\n");

    free(loss_history);
    free_matrix(input);
    free_neural_network(nn);

    return 0;
}
