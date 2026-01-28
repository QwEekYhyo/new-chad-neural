#include <utils.h>
#include <matrix.h>
#include <common_defs.h>
#include <neural_network.h>
#include <model_trainer.h>

#include <mnist_utils.h>

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <time.h>

#define TRAIN_IMAGES_FILE "train-images.idx3-ubyte"
#define TRAIN_LABELS_FILE "train-labels.idx1-ubyte"

ModelTrainer trainer;

int main(void) {
    uint32_t num_images = 0, rows = 0, cols = 0;
    double* images = read_mnist_images(TRAIN_IMAGES_FILE, &num_images, &rows, &cols);
    if (images == NULL) {
        fprintf(stderr, "Failed to load images.\n");
        return 1;
    }

    uint32_t num_labels = 0;
    double* labels = read_mnist_labels_one_hot(TRAIN_LABELS_FILE, &num_labels);
    if (labels == NULL) {
        fprintf(stderr, "Failed to load labels.\n");
        return 1;
    }

    if (num_images != num_labels) {
        printf("Number of images and labels do not match\n");
        return 1;
    }

    // Test printing the first label and first image
    /*
    printf("First image label (one-hot encoded):\n");
    for (int i = 0; i < 10; i++) {
        printf("%d ", labels[0 * 10 + i]);
    }
    printf("\nFirst image pixels:\n");
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t c = 0; c < cols; c++) {
            printf("%.2f ", images[0 * rows * cols + r * cols + c]);
        }
        printf("\n");
    }
    */

    // Actual training
    srand((unsigned int) time(NULL));

    NeuralNetwork* nn = new_neural_network(rows * cols, 50, 10);
    nn->hidden_layer_af = SIGMOID;
    nn->output_layer_af = SOFTMAX;


    trainer.nn = nn;
    trainer.learning_rate = 0.05;
    trainer.batch_size = 32;
    trainer.epochs = 750;
    set_loss_function(&trainer, CCE);

    double* history = train_with_history(&trainer, images, (double*) labels, 1000);

    set_batch_size(nn, 1);

    forward_pass(nn, inputs_from_array(images, 1));
    printf("Label:\n");
    for (int i = 0; i < 10; i++) {
        printf("%.1lf ", labels[0 * 10 + i]);
    }
    printf("\nOutput:\n");
    print_matrix(nn->output_layer);

    forward_pass(nn, inputs_from_array(images + rows * cols * 3, 1));
    printf("\nLabel:\n");
    for (int i = 0; i < 10; i++) {
        printf("%.1lf ", labels[3 * 10 + i]);
    }
    printf("\nOutput:\n");
    print_matrix(nn->output_layer);

    free(history);
    free_neural_network(nn);

    // Free memory
    free(images);
    free(labels);

    return 0;
}
