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
    set_hidden_activation_functions(nn, sigmoid, sigmoid_derivative);
    set_output_activation_functions(nn, sigmoid, sigmoid_derivative);
    ModelTrainer trainer;
    trainer.nn = nn;
    trainer.learning_rate = 0.8;
    trainer.batch_size = 10;
    trainer.epochs = 10000;

    double* history = train_with_history(&trainer, data[0], output_data[0], DATASET_SIZE);

    save_neural_network(nn, "sine_model.ncn");

    free(history);
    free_neural_network(nn);

    return 0;
}
