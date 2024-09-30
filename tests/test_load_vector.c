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

static const char* FILENAME = "test_vector.ncn";

int main(void) {
    FILE* file = fopen(FILENAME, "r");
    if (!file) {
        printf("Error while opening file where vector is saved\n");
        return 1;
    }

    Vector* v = new_vector_from_file(file);
    if (!v) {
        printf("Error while loading vector from file\n");
        return 1;
    }

    if (v->size != 4) {
        printf("Loaded vector size is wrong: %zu\n", v->size);
        return 1;
    }

    for (size_t i = 0; i < 4; i++) {
        if (!are_double_equals(v->buffer[i], 6.9)) {
            printf("Loaded vector doesn't have the right value: %lf\n", v->buffer[i]);
            return 1;
        }
    }

    print_vector(v);
    free_vector(v);

    v = new_vector_from_file(file);
    if (!v) {
        printf("Error while loading vector from file\n");
        return 1;
    }

    if (v->size != 3) {
        printf("Loaded vector size is wrong: %zu\n", v->size);
        return 1;
    }

    for (size_t i = 0; i < 3; i++) {
        if (!are_double_equals(v->buffer[i], 1.0)) {
            printf("Loaded vector doesn't have the right value: %lf\n", v->buffer[i]);
            return 1;
        }
    }
    print_vector(v);

    fclose(file);
    free_vector(v);

    return 0;
}
