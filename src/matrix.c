/* New Chad Neural - C library to train neural networks
 * Copyright (C) 2024 Lucas Logan
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include <matrix.h>
#include <utils.h>

#include <stdio.h>
#include <stdlib.h>

Matrix* new_uninitialized_matrix(size_t rows, size_t columns) {
    Matrix* new_matrix = malloc(sizeof(Matrix));
    new_matrix->rows = rows;
    new_matrix->columns = columns;
    new_matrix->buffer = malloc(rows * sizeof(double*));

    for (size_t i = 0; i < rows; i++) {
        new_matrix->buffer[i] = malloc(columns * sizeof(double));
    }

    return new_matrix;
}

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
