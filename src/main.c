#include "../include/matrix.h"
#include "../include/vector.h"

int main(void) {
    Matrix* m = new_random_matrix(3, 3);
    print_matrix(m);

    free_matrix(m);

    Vector* v = new_random_vector(4);
    print_vector(v);
    Vector* v1 = new_random_vector(4);
    print_vector(v1);

    free_vector(v);
    free_vector(v1);
    return 0;
}
