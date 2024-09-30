#ifndef NCN_VECTOR_H
#define NCN_VECTOR_H

#include <stddef.h>

typedef struct {
    size_t size;
    double* buffer;
} Vector;

Vector* new_uninitialized_vector(size_t size);
Vector* new_zero_vector(size_t size);
Vector* new_random_vector(size_t size);
void free_vector(Vector* vector);

void print_vector(Vector* vector);

#endif // NCN_VECTOR_H
