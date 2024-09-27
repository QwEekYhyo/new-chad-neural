#ifndef NCN_MODEL_TRAINER_H
#define NCN_MODEL_TRAINER_H

#include <common_defs.h>
#include <neural_network.h>

typedef struct {
    NeuralNetwork* nn;

    double learning_rate;
    size_t epochs;
    size_t batch_size;
} ModelTrainer;

void train(ModelTrainer* trainer, Vector** train_data, Vector** train_output, size_t dataset_size);

#endif // NCN_MODEL_TRAINER_H
