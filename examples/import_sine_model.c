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
#include <neural_network.h>
#include <utils.h>

#include <stdio.h>

#define BATCH_SIZE 10

int main(void) {
    NeuralNetwork* nn = new_neural_network_from_file("sine_model.ncn");
    nn->hidden_layer_af = SIGMOID;
    nn->output_layer_af = SIGMOID;

    Matrix* input = new_uninitialized_matrix(1, BATCH_SIZE);
    set_batch_size(nn, BATCH_SIZE);
    FILE *gnuplot = popen("gnuplot -persist", "w");
    fprintf(gnuplot, "plot '-'\n");

    for (size_t i = 0; i < 100; i++) {
        for (size_t b = 0; b < BATCH_SIZE; b++) {
            input->buffer[0][b] = (i * BATCH_SIZE + b) / (100.0 * BATCH_SIZE);
        }
        forward_pass(nn, input);
        for (size_t b = 0; b < BATCH_SIZE; b++) {
            fprintf(gnuplot, "%f %f\n",
                    input->buffer[0][b] * 20,
                    nn->output_layer->buffer[0][b] * 2 - 1
            );
        }
    }
    fprintf(gnuplot, "e\n");
    fflush(gnuplot);

    free_matrix(input);
    free_neural_network(nn);

    return 0;
}
