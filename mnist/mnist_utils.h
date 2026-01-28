#include <stdint.h>

// Function to swap byte order (big-endian to little-endian)
uint32_t swap_endian(uint32_t val);

// Function to read image data
double* read_mnist_images(const char* file_name, uint32_t* number_of_images, uint32_t* rows, uint32_t* cols);

// Function to read label data and return it as a one-hot encoded 2D array
double* read_mnist_labels_one_hot(const char* file_name, uint32_t* number_of_labels);
