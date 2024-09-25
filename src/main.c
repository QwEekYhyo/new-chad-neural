#include "../include/matrix.h"

int main(void) {
    Matrix* m = new_zero_matrix(3, 3);
    print_matrix(m);

    free_matrix(m);
    return 0;
}
