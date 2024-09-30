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

#include <stdlib.h>
#include <stdio.h>

#define INPUT_SIZE 3
#define OUTPUT_SIZE 2
#define DATASET_SIZE 100

const char* FILENAME = "neural_network.ncn";

int main(void) {
    NeuralNetwork* nn = new_neural_network(INPUT_SIZE, 3, OUTPUT_SIZE);

    printf("Wih:\n");
    print_matrix(nn->input_hidden_weights);
    printf("Who:\n");
    print_matrix(nn->hidden_output_weights);
    printf("Bh:\n");
    print_vector(nn->hidden_biases);
    printf("Bo:\n");
    print_vector(nn->output_biases);

    save_neural_network(nn, FILENAME);

    free_neural_network(nn);

    nn = new_neural_network_from_file(FILENAME);
    if (!nn) {
        printf("Could not load Neural Network from file %s\n", FILENAME);
        return 1;
    }

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
