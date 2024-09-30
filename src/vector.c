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

#include <vector.h>
#include <utils.h>

#include <stdio.h>
#include <stdlib.h>

Vector* new_uninitialized_vector(size_t size) {
    Vector* new_vector = malloc(sizeof(Vector));
    new_vector->size = size;
    new_vector->buffer = malloc(size * sizeof(double));

    return new_vector;
}

Vector* new_zero_vector(size_t size) {
    Vector* new_vector = malloc(sizeof(Vector));
    new_vector->size = size;
    new_vector->buffer = calloc(size, sizeof(double));

    return new_vector;
}

Vector* new_random_vector(size_t size) {
    Vector* new_vector = malloc(sizeof(Vector));
    new_vector->size = size;

    new_vector->buffer = malloc(size * sizeof(double));
    for (size_t i = 0; i < size; i++) {
        new_vector->buffer[i] = rand_double_range(0, 1);
    }

    return new_vector;
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

int save_vector(Vector* vector, const char* filename) {
    FILE* file = fopen(filename, "a");
    if (!file) {
        printf("Could not open file %s for writting\n", filename);
        return -1;
    }

    fprintf(file, "%zu\n", vector->size);
    for (size_t i = 0; i < vector->size; i++) {
        fprintf(file, "%.15lf ", vector->buffer[i]);
    }
    fprintf(file, "\n");
    fprintf(file, "---\n");
    
    fclose(file);
    return 0;
}
