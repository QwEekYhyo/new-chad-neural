#include "../include/matrix.h"
#include "../include/vector.h"

int main(void) {
    Matrix* m = new_zero_matrix(3, 3);
    print_matrix(m);

    free_matrix(m);

    Vector* v = new_zero_vector(4);
    print_vector(v);

    free_vector(v);
    return 0;
}
