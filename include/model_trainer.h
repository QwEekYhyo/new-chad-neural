#ifndef NCN_MODEL_TRAINER_H
#define NCN_MODEL_TRAINER_H

#include <neural_network.h>

#include <stddef.h>
#include <stdint.h>

typedef struct {
    NeuralNetwork* nn;

    double learning_rate;
    size_t epochs;
    size_t batch_size;
} ModelTrainer;

// train_data & train_output HAVE to be sized just like the neural network input & output
// or else consequences
void _train(ModelTrainer* trainer, double* train_data, double* train_output, size_t dataset_size, uint_least8_t with_history, double** loss_history);

void train(ModelTrainer* trainer, double* train_data, double* train_output, size_t dataset_size);
double* train_with_history(ModelTrainer* trainer, double* train_data, double* train_output, size_t dataset_size);

#endif // NCN_MODEL_TRAINER_H
