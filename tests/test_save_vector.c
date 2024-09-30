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

static const char* FILENAME = "vector.ncn";

int main(void) {
    Vector* v = new_uninitialized_vector(5);

    for (size_t i = 0; i < v->size; i++) {
        v->buffer[i] = i / 5.0;
        v->buffer[i] *= v->buffer[i];
    }

    print_vector(v);

    FILE* file = fopen(FILENAME, "a");
    if (!file) {
        printf("Could not open file to save Vector\n");
        return 1;
    }

    int error = save_vector(v, file);
    if (error != 0) {
        printf("Error while saving vector\n");
        return 1;
    }
    fclose(file);

    // Check if it was correctly saved
    file = fopen(FILENAME, "r");
    
    char type;
    fscanf(file, "%c", &type);
    if (type != 'V') {
        printf("Saved type is not Vector type, got: %c\n",
                type
        );
        return 1;
    }

    size_t size;
    fscanf(file, "%zu", &size);
    if (size != v->size) {
        printf("Saved size differs from real size, expected: %zu, got: %zu\n",
                v->size,
                size
        );
        return 1;
    }

    double current;
    for (size_t i = 0; i < v->size; i++) {
        fscanf(file, "%lf", &current);
        if (!are_double_equals(current, v->buffer[i])) {
            printf("Saved value differs from real value, expected: %.15lf, got: %.15lf\n",
                    v->buffer[i],
                    current
            );
            return 1;
        }
    }
    
    fclose(file);
    free_vector(v);

    return 0;
}
