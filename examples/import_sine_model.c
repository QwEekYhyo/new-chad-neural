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

int main(void) {
    NeuralNetwork* nn = new_neural_network_from_file("sine_model.ncn");
    set_hidden_activation_functions(nn, sigmoid, sigmoid_derivative);
    set_output_activation_functions(nn, sigmoid, sigmoid_derivative);

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

    free_matrix(input);
    free_neural_network(nn);

    return 0;
}
