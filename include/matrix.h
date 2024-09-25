#ifndef NCN_MATRIX_H
#define NCN_MATRIX_H

#include "common_defs.h"

typedef struct {
    size_t rows;
    size_t columns;
    double** buffer;
} Matrix;

Matrix* new_zero_matrix(size_t rows, size_t columns);
Matrix* new_random_matrix(size_t rows, size_t columns);
void free_matrix(Matrix* matrix);

void print_matrix(Matrix* matrix);

#endif // NCN_MATRIX_H
