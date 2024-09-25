#include "../include/matrix.h"
#include "../include/utils.h"

#include <stdio.h>
#include <stdlib.h>

Matrix* new_zero_matrix(size_t rows, size_t columns) {
    Matrix* new_matrix = malloc(sizeof(Matrix));
    new_matrix->rows = rows;
    new_matrix->columns = columns;
    new_matrix->buffer = malloc(rows * sizeof(double*));

    for (size_t i = 0; i < rows; i++) {
        new_matrix->buffer[i] = calloc(columns, sizeof(double));
    }

    return new_matrix;
}

Matrix* new_random_matrix(size_t rows, size_t columns) {
    Matrix* new_matrix = malloc(sizeof(Matrix));
    new_matrix->rows = rows;
    new_matrix->columns = columns;
    new_matrix->buffer = malloc(rows * sizeof(double*));

    for (size_t i = 0; i < rows; i++) {
        new_matrix->buffer[i] = malloc(columns * sizeof(double));
        for (size_t j = 0; j < columns; j++) {
            new_matrix->buffer[i][j] = rand_double_range(0, 1);
        }
    }

    return new_matrix;
}

void free_matrix(Matrix* matrix) {
    for (size_t i = 0; i < matrix->rows; i++) {
        free(matrix->buffer[i]);
    }
    free(matrix->buffer);
    free(matrix);
}

void print_matrix(Matrix* matrix) {
    for (size_t i = 0; i < matrix->rows; i++) {
        printf("[ ");
        for (size_t j = 0; j < matrix->columns; j++) {
            printf("%f ", matrix->buffer[i][j]);
        }
        printf("]\n");
    }
}
