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

#include <common_defs.h>
#include <neural_network.h>
#include <utils.h>
#include <model_trainer.h>

#include <stdlib.h>
#include <math.h>

#define INPUT_SIZE 1
#define OUTPUT_SIZE 1
#define DATASET_SIZE 1000

double f(double x) {
    return (sin(20.0 * x) + 1) / 2.0;
}

ModelTrainer trainer;

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
    nn->hidden_layer_af = SIGMOID;
    nn->output_layer_af = SIGMOID;
    trainer.nn = nn;
    trainer.learning_rate = 0.8;
    trainer.batch_size = 10;
    trainer.epochs = 10000;

    double* history = train_with_history_bare(&trainer, data[0], output_data[0], DATASET_SIZE);

    save_neural_network(nn, "sine_model.ncn");

    free(history);
    free_neural_network(nn);

    return 0;
}
