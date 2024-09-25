#include "../include/vector.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

Vector* new_zero_vector(size_t size) {
    Vector* new_vector = malloc(sizeof(Vector));
    new_vector->size = size;
    new_vector->buffer = calloc(size, sizeof(double));

    return new_vector;
}

Vector* new_random_vector(size_t size) {
    return NULL;
}

void free_vector(Vector* vector) {
    free(vector->buffer);
    free(vector);
}

void print_vector(Vector* vector) {
    printf("[ ");
    for (size_t i = 0; i < vector->size; i++) {
        printf("%f ", vector->buffer[i]);
    }
    printf("]\n");
}
