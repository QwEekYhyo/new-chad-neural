#include <matrix.h>
#include <neural_network.h>
#include <model_trainer.h>
#include <utils.h>

#include <stdio.h>
#include <stdlib.h>

/*
 * 1 epoch    = all dataset covered
 * 1 pass     = 1 forward pass + 1 backward pass
 * batch_size = number of samples used for one pass
 * iteration  = number of passes needed to perform 1 epoch
 */

// train_data and train_output should oviously be the same size (dataset_size)
void train(ModelTrainer* trainer, double* train_data, double* train_output, size_t dataset_size) {
    _train(trainer, train_data, train_output, dataset_size, 0, NULL);
}

double* train_with_history(ModelTrainer* trainer, double* train_data, double* train_output, size_t dataset_size) {
    double* loss_history = malloc(trainer->epochs * sizeof(double));
    _train(trainer, train_data, train_output, dataset_size, 1, &loss_history);
    return loss_history;
}

void _train(ModelTrainer* trainer, double* train_data, double* train_output, size_t dataset_size, uint_least8_t with_history, double** loss_history) {
    if (with_history && !loss_history) {
        printf("Called train with history without providing pointer to store loss history\n");
        return;
    }

    size_t input_size = trainer->nn->input_size;
    size_t output_size = trainer->nn->output_layer->rows;
    size_t max_size = max(input_size, output_size);

    size_t iterations = dataset_size / trainer->batch_size;
    size_t actually_trained = iterations * trainer->batch_size;
    size_t not_trained = dataset_size - actually_trained;

    Matrix* input;
    Matrix* output;
    for (size_t epoch = 0; epoch < trainer->epochs; epoch++) {
        if (epoch % 100 == 0)
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
            printf("training epoch = %zu\n", epoch);
#else
            printf("training epoch = %lu\n", epoch);
#endif

        double current_loss = 0;

        if (not_trained != 0) {
            // First train separately the small portion of the dataset left out by the iteration division
            set_batch_size(trainer->nn, not_trained);
            input  = new_uninitialized_matrix(input_size , not_trained);
            output = new_uninitialized_matrix(output_size , not_trained);

            // Fill input & output matrices
            // I feel like filling both matrices at the same time using if statements to check boundaries
            // is more efficient than looping twice ?
            for (size_t data_index = 0; data_index < not_trained; data_index++) {
                for (size_t vector_index = 0; vector_index < max_size; vector_index++) {
                    if (vector_index < input_size)
                        input->buffer[vector_index][data_index] =
                            train_data[(actually_trained + data_index) * input_size + vector_index];
                    if (vector_index < output_size)
                        output->buffer[vector_index][data_index] =
                            train_output[(actually_trained + data_index) * output_size + vector_index];
                }
            }

            // Train the small batch
            forward_pass(trainer->nn, input);
            if (with_history) {
                // Add loss
                for (size_t o = 0; o < not_trained; o++) {
                    for (size_t i = 0; i < output_size; i++) {
                        double difference =
                            trainer->nn->output_layer->buffer[i][o] - output->buffer[i][o];
                        current_loss += difference * difference;
                    }
                }
            }
            back_propagation(trainer->nn, input, output);

            free_matrix(input);
            free_matrix(output);
        }

        // Now we take care of the rest (aka normal sized batches)
        input  = new_uninitialized_matrix(input_size , trainer->batch_size);
        output = new_uninitialized_matrix(output_size , trainer->batch_size);
        for (size_t iteration = 0; iteration < iterations; iteration++) {
            // Fill input & output matrices
            // I feel like filling both matrices at the same time using if statements to check boundaries
            // is more efficient than looping twice ?
            for (size_t data_index = 0; data_index < trainer->batch_size; data_index++) {
                for (size_t vector_index = 0; vector_index < max_size; vector_index++) {
                    size_t dataset_index = iteration * trainer->batch_size + data_index;

                    if (vector_index < input_size)
                        input->buffer[vector_index][data_index] =
                            train_data[dataset_index * input_size + vector_index];
                    if (vector_index < output_size)
                        output->buffer[vector_index][data_index] =
                            train_output[dataset_index * output_size + vector_index];
                }
            }

            // Train batch
            forward_pass(trainer->nn, input);
            if (with_history) {
                // Add loss
                for (size_t o = 0; o < trainer->batch_size; o++) {
                    for (size_t i = 0; i < output_size; i++) {
                        double difference =
                            trainer->nn->output_layer->buffer[i][o] - output->buffer[i][o];
                        current_loss += difference * difference;
                    }
                }
            }
            back_propagation(trainer->nn, input, output);
        }

        if (with_history)
            (*loss_history)[epoch] = current_loss / (dataset_size * output_size);

        free_matrix(input);
        free_matrix(output);
    }
}
