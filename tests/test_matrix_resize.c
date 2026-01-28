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

#define INITIAL_SIZE 3
#define NEW_SIZE 6

int main(void) {
    Matrix* m = new_uninitialized_matrix(4, INITIAL_SIZE);

    if (m->_columns != INITIAL_SIZE || m->columns != INITIAL_SIZE) {
        printf("Incorrect initialization of Matrix\n");
        return 1;
    }

    int error = set_columns(m, NEW_SIZE);
    if (error < 0)
        return 1;

    if (m->_columns != NEW_SIZE || m->columns != NEW_SIZE) {
        printf("Incorrect resizing of Matrix\n");
        return 1;
    }

    error = set_columns(m, INITIAL_SIZE);
    if (error < 0)
        return 1;

    if (m->_columns != NEW_SIZE || m->columns != INITIAL_SIZE) {
        printf("Incorrect resizing of Matrix, ");
        printf("allocated memory for Matrix should not change when setting size down\n");
        return 1;
    }

    free_matrix(m);

    return 0;
}
