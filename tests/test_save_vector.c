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

    int error = save_vector(v, FILENAME);
    if (error != 0) {
        printf("Error while saving vector\n");
        return 1;
    }

    // Check if it was correctly saved
    FILE* file = fopen(FILENAME, "r");
    if (!file) {
        printf("Error while opening file where vector is saved\n");
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
