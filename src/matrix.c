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

#define _CRT_SECURE_NO_WARNINGS

#include <matrix.h>
#include <utils.h>

#include <stdio.h>
#include <stdlib.h>

Matrix* new_uninitialized_matrix(size_t rows, size_t columns) {
    Matrix* new_matrix = _allocate_matrix(rows, columns);

    if (new_matrix) {
        new_matrix->buffer = malloc(rows * columns * sizeof(double));
    }

    return new_matrix;
}

Matrix* new_zero_matrix(size_t rows, size_t columns) {
    Matrix* new_matrix = _allocate_matrix(rows, columns);

    if (new_matrix) {
        new_matrix->buffer = calloc(rows * columns, sizeof(double));
    }

    return new_matrix;
}

Matrix* new_random_matrix(size_t rows, size_t columns) {
    Matrix* new_matrix = _allocate_matrix(rows, columns);

    if (new_matrix) {
        new_matrix->buffer = malloc(rows * columns * sizeof(double));

        for (size_t i = 0; i < rows * columns; i++) {
            new_matrix->buffer[i] = rand_double_range(0, 1);
        }
    }

    return new_matrix;
}

// Normal Xavier initialization
Matrix* new_glorot_normal_matrix(size_t rows, size_t columns) {
    Matrix* new_matrix = _allocate_matrix(rows, columns);

    if (new_matrix) {
        new_matrix->buffer = malloc(rows * columns * sizeof(double));

        double stddev = sqrt(2.0 / (columns + rows));
        double a;
        double b;

        size_t n = rows * columns;
        size_t i = 0;

        for (; i + 1 < n; i += 2) {
            rand_normal(0.0, stddev, &a, &b);
            new_matrix->buffer[i] = a;
            new_matrix->buffer[i + 1] = b;
        }

        if (i < n) {
            rand_normal(0.0, stddev, &a, &b);
            new_matrix->buffer[i] = a;
        }
    }

    return new_matrix;
}

Matrix* _allocate_matrix(size_t rows, size_t columns) {
    if (rows == 0 || columns == 0) {
        printf("Cannot allocate Matrix with size 0\n");
        return NULL;
    }
    if (rows * columns >= 100000) {
        printf("[WARNING] Trying to allocate a big Matrix (%zu, %zu)\n", rows, columns);
    }

    Matrix* new_matrix = malloc(sizeof(Matrix));
    new_matrix->rows = rows;
    new_matrix->columns = columns;
    new_matrix->_columns = columns;
    
    return new_matrix;
}

void free_matrix(Matrix* matrix) {
    free(matrix->buffer);
    free(matrix);
}

void print_matrix(Matrix* matrix) {
    for (size_t i = 0; i < matrix->rows; i++) {
        printf("[ ");
        for (size_t j = 0; j < matrix->columns; j++) {
            printf("%f ", MAT(matrix, i, j));
        }
        printf("]\n");
    }
}

int set_columns(Matrix* matrix, size_t columns) {
    if (columns == 0) {
        printf("Cannot set Matrix columns to 0\n");
        return -2;
    }
    if (columns >= 10000) {
        printf("[WARNING] Trying to reallocate a big Matrix (%zu, %zu)\n", matrix->rows, columns);
    }

    matrix->columns = columns;

    if (columns > matrix->_columns) {
        matrix->buffer = realloc(matrix->buffer, matrix->rows * columns * sizeof(double));

        if (!matrix->buffer) {
            printf("Reallocation of Matrix failed, could not allocate\n");
            return -1;
        }

        matrix->_columns = columns;
    }

    return 0;
}

int save_matrix(Matrix* matrix, FILE* file) {
    if (!matrix) {
        printf("No Matrix provided for save\n");
        return -1;
    }

    if (!file) {
        printf("No opened file provided to save Matrix\n");
        return -1;
    }

    fprintf(file, "%c ", 'M');
    fprintf(file, "%zu %zu\n", matrix->rows, matrix->columns);
    for (size_t r = 0; r < matrix->rows; r++) {
        for (size_t c = 0; c < matrix->columns; c++) {
            fprintf(file, "%.15lf ", MAT(matrix, r, c));
        }
        fprintf(file, "\n");
    }
    fprintf(file, "---\n");
    
    return 0;
}

Matrix* new_matrix_from_file(FILE* file) {
    if (!file) {
        printf("No opened file provided to load Matrix from\n");
        return NULL;
    }

    char type;
    fscanf(file, "%c", &type);
    if (type != 'M') {
        printf("Type \"%c\" is not Matrix type\n", type);
        return NULL;
    }

    size_t rows, columns;
    fscanf(file, "%zu %zu", &rows, &columns);

    Matrix* new_matrix = new_uninitialized_matrix(rows, columns);
    for (size_t r = 0; r < rows; r++) {
        for (size_t c = 0; c < columns; c++) {
            fscanf(file, "%lf", &MAT(new_matrix, r, c));
        }
    }

    char delimiter[6];
    fscanf(file, "%6c", delimiter);

    return new_matrix;
}
