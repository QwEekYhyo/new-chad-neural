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

static const char* FILENAME = "matrix.ncn";

int main(void) {
    Matrix* m = new_uninitialized_matrix(3, 4);

    for (size_t i = 0; i < m->rows; i++) {
        for (size_t j = 0; j < m->columns; j++) {
            m->buffer[i][j] = (i * 10) + j / 15.0;
            m->buffer[i][j] *= m->buffer[i][j];
        }
    }

    print_matrix(m);

    FILE* file = fopen(FILENAME, "a");
    if (!file) {
        printf("Could not open file to save Matrix\n");
        return 1;
    }

    int error = save_matrix(m, file);
    if (error != 0) {
        printf("Error while saving vector\n");
        return 1;
    }
    fclose(file);

    // Check if it was correctly saved
    file = fopen(FILENAME, "r");

    char type;
    fscanf(file, "%c", &type);
    if (type != 'M') {
        printf("Saved type is not Vector type, got: %c\n",
                type
        );
        return 1;
    }

    size_t rows, columns;
    fscanf(file, "%zu %zu", &rows, &columns);
    if (rows != m->rows) {
        printf("Saved size differs from real size, expected: %zu, got: %zu\n",
                m->rows,
                rows
        );
        return 1;
    }
    if (columns != m->columns) {
        printf("Saved size differs from real size, expected: %zu, got: %zu\n",
                m->columns,
                columns
        );
        return 1;
    }

    double current;
    for (size_t i = 0; i < m->rows; i++) {
        for (size_t j = 0; j < m->columns; j++) {
            fscanf(file, "%lf", &current);
            if (!are_double_equals(current, m->buffer[i][j])) {
                printf("Saved value differs from real value, expected: %.15lf, got: %.15lf\n",
                        m->buffer[i][j],
                        current
                );
                return 1;
            }
        }
    }
    
    fclose(file);
    free_matrix(m);

    return 0;
}
