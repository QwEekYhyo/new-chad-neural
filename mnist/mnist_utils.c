#include <mnist_utils.h>

#include <stdio.h>
#include <stdlib.h>

uint32_t swap_endian(uint32_t val) {
    return ((val << 24) & 0xFF000000) | ((val << 8)  & 0x00FF0000) |
           ((val >> 8)  & 0x0000FF00) |
           ((val >> 24) & 0x000000FF);
}

double* read_mnist_images(const char* file_name, uint32_t* number_of_images, uint32_t* rows, uint32_t* cols) {
    FILE *file = fopen(file_name, "rb");
    if (file == NULL) {
        perror("Error opening file");
        return NULL;
    }

    // Read the magic number
    uint32_t magic_number = 0;
    fread(&magic_number, sizeof(uint32_t), 1, file);
    magic_number = swap_endian(magic_number);
    if (magic_number != 2051) {
        fprintf(stderr, "Invalid MNIST image file!\n");
        fclose(file);
        return NULL;
    }

    // Read the number of images, rows, and columns
    uint32_t num_images = 0, num_rows = 0, num_cols = 0;
    fread(&num_images, sizeof(uint32_t), 1, file);
    fread(&num_rows, sizeof(uint32_t), 1, file);
    fread(&num_cols, sizeof(uint32_t), 1, file);

    num_images = swap_endian(num_images);
    num_rows = swap_endian(num_rows);
    num_cols = swap_endian(num_cols);

    *number_of_images = num_images;
    *rows = num_rows;
    *cols = num_cols;

    // Allocate memory for all images in a flat array
    double* normalized_images = malloc(num_images * num_rows * num_cols * sizeof(double));
    if (!normalized_images) {
        perror("Failed to allocate memory for images");
        fclose(file);
        return NULL;
    }

    unsigned char* images = malloc(num_images * num_rows * num_cols * sizeof(unsigned char));
    if (!images) {
        perror("Failed to allocate memory for images");
        free(normalized_images);
        fclose(file);
        return NULL;
    }
    fread(images, sizeof(unsigned char), num_images * num_rows * num_cols, file);

    for (uint32_t i = 0; i < num_images * num_rows * num_cols; i++) {
        normalized_images[i] = images[i] / 255.0;
        // The commented line below dates back from 2024
        // I have no clue why it was here nor why it's commented
        /* normalized_images[i] -= 0.15; */
    }

    fclose(file);
    free(images);
    return normalized_images;
}

double* read_mnist_labels_one_hot(const char* file_name, uint32_t* number_of_labels) {
    FILE *file = fopen(file_name, "rb");
    if (file == NULL) {
        perror("Error opening file");
        return NULL;
    }

    // Read the magic number
    uint32_t magic_number = 0;
    fread(&magic_number, sizeof(uint32_t), 1, file);
    magic_number = swap_endian(magic_number);
    if (magic_number != 2049) {
        fprintf(stderr, "Invalid MNIST label file!\n");
        fclose(file);
        return NULL;
    }

    // Read the number of labels
    uint32_t num_labels = 0;
    fread(&num_labels, sizeof(uint32_t), 1, file);
    num_labels = swap_endian(num_labels);
    *number_of_labels = num_labels;

    // Allocate memory for one-hot encoded labels (num_labels * 10)
    double* labels = calloc(num_labels * 10, sizeof(double));

    // Read each label and set the corresponding index in the one-hot array to 1
    for (uint32_t i = 0; i < num_labels; i++) {
        unsigned char label;
        fread(&label, sizeof(unsigned char), 1, file);
        labels[i * 10 + label] = 1.0;
    }

    fclose(file);
    return labels;
}
