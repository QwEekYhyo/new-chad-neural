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

#ifndef NCN_VECTOR_H
#define NCN_VECTOR_H

#include <stddef.h>
#include <stdio.h>

typedef struct {
    size_t size;
    double* buffer;
} Vector;

Vector* new_uninitialized_vector(size_t size);
Vector* new_zero_vector(size_t size);
Vector* new_random_vector(size_t size);
void free_vector(Vector* vector);

void print_vector(Vector* vector);

int save_vector(Vector* vector, const char* filename);
Vector* new_vector_from_file(FILE* file);

#endif // NCN_VECTOR_H
