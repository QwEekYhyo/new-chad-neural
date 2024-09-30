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

static const char* FILENAME = "test_matrix.ncn";

int main(void) {
    FILE* file = fopen(FILENAME, "r");
    if (!file) {
        printf("Error while opening file where matrix is saved\n");
        return 1;
    }

    Matrix* m = new_matrix_from_file(file);
    if (!m) {
        printf("Error while loading matrix from file\n");
        return 1;
    }

    if (m->rows != 4) {
        printf("Loaded matrix size is wrong: %zu\n", m->rows);
        return 1;
    }
    if (m->columns != 3) {
        printf("Loaded matrix size is wrong: %zu\n", m->columns);
        return 1;
    }

    for (size_t r = 0; r < 4; r++) {
        for (size_t c = 0; c < 3; c++) {
            if (!are_double_equals(m->buffer[r][c], 6.9)) {
                printf("Loaded matrix doesn't have the right value: %lf\n", m->buffer[r][c]);
                return 1;
            }
        }
    }

    print_matrix(m);
    free_matrix(m);

    m = new_matrix_from_file(file);
    if (!m) {
        printf("Error while loading matrix from file\n");
        return 1;
    }

    if (m->rows != 2) {
        printf("Loaded matrix size is wrong: %zu\n", m->rows);
        return 1;
    }
    if (m->columns != 4) {
        printf("Loaded matrix size is wrong: %zu\n", m->columns);
        return 1;
    }

    for (size_t r = 0; r < 2; r++) {
        for (size_t c = 0; c < 4; c++) {
            if (!are_double_equals(m->buffer[r][c], 1.0)) {
                printf("Loaded matrix doesn't have the right value: %lf\n", m->buffer[r][c]);
                return 1;
            }
        }
    }
    print_matrix(m);

    free_matrix(m);
    fclose(file);

    return 0;
}
