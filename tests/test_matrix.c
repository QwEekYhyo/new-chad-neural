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
#include <stdlib.h>

int main(void) {
    int error_count = 0;

    printf("Test 1: Zero-sized matrix creation...\n");
    Matrix* m_zero = new_uninitialized_matrix(0, 0);
    if (m_zero == NULL) {
        printf("  PASS: Zero-sized matrix correctly returns NULL\n");
    } else {
        printf("  FAIL: Zero-sized matrix should return NULL\n");
        error_count++;
        free_matrix(m_zero);
    }

    printf("\nTest 2: Matrix with zero rows...\n");
    Matrix* m_zero_rows = new_uninitialized_matrix(0, 5);
    if (m_zero_rows == NULL) {
        printf("  PASS: Matrix with zero rows correctly returns NULL\n");
    } else {
        printf("  FAIL: Matrix with zero rows should return NULL\n");
        error_count++;
        free_matrix(m_zero_rows);
    }

    printf("\nTest 3: Matrix with zero columns...\n");
    Matrix* m_zero_cols = new_uninitialized_matrix(5, 0);
    if (m_zero_cols == NULL) {
        printf("  PASS: Matrix with zero columns correctly returns NULL\n");
    } else {
        printf("  FAIL: Matrix with zero columns should return NULL\n");
        error_count++;
        free_matrix(m_zero_cols);
    }

    printf("\nTest 4: Single element matrix...\n");
    Matrix* m_single = new_zero_matrix(1, 1);
    if (m_single != NULL && m_single->rows == 1 && m_single->columns == 1) {
        if (are_double_equals(MAT(m_single, 0, 0), 0.0)) {
            printf("  PASS: Single element matrix created correctly\n");
        } else {
            printf("  FAIL: Single element matrix value incorrect\n");
            error_count++;
        }
        free_matrix(m_single);
    } else {
        printf("  FAIL: Single element matrix creation failed\n");
        error_count++;
    }

    printf("\nTest 5: Resize matrix to zero columns...\n");
    Matrix* m_resize = new_uninitialized_matrix(3, 5);
    if (m_resize != NULL) {
        int result = set_columns(m_resize, 0);
        if (result < 0) {
            printf("  PASS: Resizing to zero columns correctly returns error\n");
        } else {
            printf("  FAIL: Resizing to zero columns should return error\n");
            error_count++;
        }
        free_matrix(m_resize);
    }

    printf("\nTest 6: Save NULL matrix...\n");
    FILE* file = fopen("test_null.ncn", "w");
    if (file) {
        int result = save_matrix(NULL, file);
        if (result != 0) {
            printf("  PASS: Saving NULL matrix returns error\n");
        } else {
            printf("  FAIL: Saving NULL matrix should return error\n");
            error_count++;
        }
        fclose(file);
        remove("test_null.ncn");
    }

    printf("\nTest 7: Save matrix to NULL file...\n");
    Matrix* m_save = new_uninitialized_matrix(2, 2);
    if (m_save != NULL) {
        int result = save_matrix(m_save, NULL);
        if (result != 0) {
            printf("  PASS: Saving to NULL file returns error\n");
        } else {
            printf("  FAIL: Saving to NULL file should return error\n");
            error_count++;
        }
        free_matrix(m_save);
    }

    printf("\nTest 8: Load matrix from empty file...\n");
    FILE* empty_file = fopen("test_empty.ncn", "w");
    if (empty_file) {
        fclose(empty_file);
        empty_file = fopen("test_empty.ncn", "r");
        Matrix* m_empty = new_matrix_from_file(empty_file);
        if (m_empty == NULL) {
            printf("  PASS: Loading from empty file returns NULL\n");
        } else {
            printf("  FAIL: Loading from empty file should return NULL\n");
            error_count++;
            free_matrix(m_empty);
        }
        fclose(empty_file);
        remove("test_empty.ncn");
    }

    printf("\nTest 9: Load matrix from corrupted file...\n");
    FILE* corrupt_file = fopen("test_corrupt.ncn", "w");
    if (corrupt_file) {
        fprintf(corrupt_file, "X 5 5\n1.0 2.0 3.0\n"); // Wrong type marker
        fclose(corrupt_file);
        corrupt_file = fopen("test_corrupt.ncn", "r");
        Matrix* m_corrupt = new_matrix_from_file(corrupt_file);
        if (m_corrupt == NULL) {
            printf("  PASS: Loading from corrupted file returns NULL\n");
        } else {
            printf("  FAIL: Loading from corrupted file should return NULL\n");
            error_count++;
            free_matrix(m_corrupt);
        }
        fclose(corrupt_file);
        remove("test_corrupt.ncn");
    }

    printf("\nTest 10: Load matrix from NULL file...\n");
    Matrix* m_null_file = new_matrix_from_file(NULL);
    if (m_null_file == NULL) {
        printf("  PASS: Loading from NULL file returns NULL\n");
    } else {
        printf("  FAIL: Loading from NULL file should return NULL\n");
        error_count++;
        free_matrix(m_null_file);
    }

    printf("\nTest 11: Load matrix from file with negative size...\n");
    FILE* neg_file = fopen("test_matrix_neg.ncn", "w");
    if (neg_file) {
        fprintf(neg_file, "M 1 -5\n");
        fclose(neg_file);
        neg_file = fopen("test_matrix_neg.ncn", "r");
        Matrix* m_neg_size = new_matrix_from_file(neg_file);
        if (m_neg_size == NULL) {
            printf("  PASS: Loading matrix with negative size returns NULL\n");
        } else {
            printf("  FAIL: Loading matrix with negative size should return NULL\n");
            error_count++;
            free_matrix(m_neg_size);
        }
        fclose(neg_file);
        remove("test_matrix_neg.ncn");
    }

    // Summary
    printf("\n========================================\n");
    if (error_count == 0) {
        printf("All tests PASSED!\n");
    } else {
        printf("Tests completed with %d FAILURES\n", error_count);
    }
    printf("========================================\n");

    return error_count;
}
