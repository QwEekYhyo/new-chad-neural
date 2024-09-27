#include <matrix.h>
#include <neural_network.h>
#include <model_trainer.h>
#include <utils.h>

#include <stdio.h>

/*
 * 1 epoch    = all dataset covered
 * 1 pass     = 1 forward pass + 1 backward pass
 * batch_size = number of samples used for one pass
 * iteration  = number of passes needed to perform 1 epoch
 */

// train_data and train_output should oviously be the same size (dataset_size)
void train(ModelTrainer* trainer, Vector** train_data, Vector** train_output, size_t dataset_size) {

    size_t input_size = trainer->nn->input_size;
    size_t output_size = trainer->nn->output_layer->rows;
    size_t max_size = max(input_size, output_size);

    size_t iterations = dataset_size / trainer->batch_size;
    size_t actually_trained = iterations * trainer->batch_size;
    size_t not_trained = dataset_size - actually_trained;

    Matrix* input;
    Matrix* output;
    for (size_t epoch = 0; epoch < trainer->epochs; epoch++) {
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
                        train_data[actually_trained + data_index]->buffer[vector_index];
                if (vector_index < output_size)
                    output->buffer[vector_index][data_index] =
                        train_output[actually_trained + data_index]->buffer[vector_index];
            }
        }

        // Train the small batch here
        // ...

        printf("Created input matrix:\n");
        print_matrix(input);
        printf("Created output matrix:\n");
        print_matrix(output);

        // Now we take care of the rest
        free_matrix(input);
        free_matrix(output);
        input  = new_uninitialized_matrix(input_size , trainer->batch_size);
        output = new_uninitialized_matrix(output_size , trainer->batch_size);
        for (size_t iteration = 0; iteration < iterations; iteration++) {
            // Fill input & output matrices
            for (size_t data_index = 0; data_index < trainer->batch_size; data_index++) {
                for (size_t vector_index = 0; vector_index < max_size; vector_index++) {
                    size_t dataset_index = iteration * trainer->batch_size + data_index;

                    if (vector_index < input_size)
                        input->buffer[vector_index][data_index] =
                            train_data[dataset_index]->buffer[vector_index];
                    if (vector_index < output_size)
                        output->buffer[vector_index][data_index] =
                            train_output[dataset_index]->buffer[vector_index];
                }
            }
            // Train batch here
            // ...
            
            printf("Created input matrix:\n");
            print_matrix(input);
            printf("Created output matrix:\n");
            print_matrix(output);
        }
        free_matrix(input);
        free_matrix(output);
    }
}
