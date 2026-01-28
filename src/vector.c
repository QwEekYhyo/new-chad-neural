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

#include <vector.h>
#include <utils.h>

#include <stdio.h>
#include <stdlib.h>

Vector* new_uninitialized_vector(size_t size) {
    if (size == 0) {
        printf("Cannot allocate Vector of size 0\n");
        return NULL;
    }

    Vector* new_vector = malloc(sizeof(Vector));
    new_vector->size = size;
    new_vector->buffer = malloc(size * sizeof(double));

    return new_vector;
}

Vector* new_zero_vector(size_t size) {
    if (size == 0) {
        printf("Cannot allocate Vector of size 0\n");
        return NULL;
    }

    Vector* new_vector = malloc(sizeof(Vector));
    new_vector->size = size;
    new_vector->buffer = calloc(size, sizeof(double));

    return new_vector;
}

Vector* new_random_vector(size_t size) {
    if (size == 0) {
        printf("Cannot allocate Vector of size 0\n");
        return NULL;
    }

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

int save_vector(Vector* vector, FILE* file) {
    if (!vector) {
        printf("No Vector provided for save\n");
        return -1;
    }

    if (!file) {
        printf("No opened file provided to save Vector\n");
        return -1;
    }

    fprintf(file, "%c ", 'V');
    fprintf(file, "%zu\n", vector->size);
    for (size_t i = 0; i < vector->size; i++) {
        fprintf(file, "%.15lf ", vector->buffer[i]);
    }
    fprintf(file, "\n");
    fprintf(file, "---\n");
    
    return 0;
}

Vector* new_vector_from_file(FILE* file) {
    if (!file) {
        printf("No opened file provided to load Vector from\n");
        return NULL;
    }

    char type;
    int fscanf_res = fscanf(file, "%c", &type);
    if (fscanf_res != 1) {
        printf("Tried to load Vector from empty file\n");
        return NULL;
    }
    if (type != 'V') {
        printf("Type \"%c\" is not Vector type\n", type);
        return NULL;
    }

    long long temp_signed_size;
    fscanf(file, "%lld", &temp_signed_size);
#ifdef DEBUG
    printf("Read size: %lld\n", temp_signed_size);
#endif
    if (temp_signed_size < 0) {
        printf("File contains a Vector with negative size\n");
        return NULL;
    }

    size_t size = (size_t) temp_signed_size;

    Vector* new_vector = new_uninitialized_vector(size);
    if (!new_vector) {
        printf("Failed to allocate Vector of size %zu\n", size);
        return NULL;
    }

    for (size_t i = 0; i < size; i++) {
        fscanf_res = fscanf(file, "%lf", &new_vector->buffer[i]);
        if (fscanf_res != 1) {
            printf("Expected Vector of size %zu, but only found %zu elements\n", size, i);
            free_vector(new_vector);
            return NULL;
        }
    }

    char delimiter[6];
    fscanf(file, "%6c", delimiter);

    return new_vector;
}
