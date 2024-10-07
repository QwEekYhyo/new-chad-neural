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

#ifndef NCN_MATRIX_H
#define NCN_MATRIX_H

#include <stddef.h>
#include <stdio.h>

typedef struct {
    size_t rows;
    size_t columns; // number of used columns
    size_t _columns; // number of actually allocated columns
    double** buffer;
} Matrix;

Matrix* new_uninitialized_matrix(size_t rows, size_t columns);
Matrix* new_zero_matrix(size_t rows, size_t columns);
Matrix* new_random_matrix(size_t rows, size_t columns);
void free_matrix(Matrix* matrix);

void print_matrix(Matrix* matrix);

int set_columns(Matrix* matrix, size_t columns);

int save_matrix(Matrix* matrix, FILE* file);
Matrix* new_matrix_from_file(FILE* file);

#endif // NCN_MATRIX_H
