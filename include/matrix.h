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

typedef struct {
    size_t rows;
    size_t columns;
    double** buffer;
} Matrix;

Matrix* new_uninitialized_matrix(size_t rows, size_t columns);
Matrix* new_zero_matrix(size_t rows, size_t columns);
Matrix* new_random_matrix(size_t rows, size_t columns);
void free_matrix(Matrix* matrix);

void print_matrix(Matrix* matrix);

int save_matrix(Matrix* matrix, const char* filename);

#endif // NCN_MATRIX_H
