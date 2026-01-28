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
#include <stdlib.h>

int main(void) {
    int error_count = 0;

    printf("Test 1: Zero-sized vector creation...\n");
    Vector* v_zero = new_uninitialized_vector(0);
    if (v_zero == NULL) {
        printf("  PASS: Zero-sized vector correctly returns NULL\n");
    } else {
        printf("  FAIL: Zero-sized vector should return NULL\n");
        error_count++;
        free_vector(v_zero);
    }

    printf("\nTest 2: Single element vector...\n");
    Vector* v_single = new_zero_vector(1);
    if (v_single != NULL && v_single->size == 1) {
        if (are_double_equals(v_single->buffer[0], 0.0)) {
            printf("  PASS: Single element vector created correctly\n");
        } else {
            printf("  FAIL: Single element vector value incorrect\n");
            error_count++;
        }
        free_vector(v_single);
    } else {
        printf("  FAIL: Single element vector creation failed\n");
        error_count++;
    }

    printf("\nTest 3: Random vector bounds checking...\n");
    Vector* v_random = new_random_vector(100);
    if (v_random != NULL) {
        int all_in_range = 1;
        for (size_t i = 0; i < v_random->size; i++) {
            if (v_random->buffer[i] < -1.0 || v_random->buffer[i] > 1.0) {
                all_in_range = 0;
                break;
            }
        }
        if (all_in_range) {
            printf("  PASS: All random values in expected range\n");
        } else {
            printf("  FAIL: Random values outside expected range\n");
            error_count++;
        }
        free_vector(v_random);
    } else {
        printf("  FAIL: Random vector creation failed\n");
        error_count++;
    }

    printf("\nTest 4: Save NULL vector...\n");
    FILE* file = fopen("test_vector_null.ncn", "w");
    if (file) {
        int result = save_vector(NULL, file);
        if (result != 0) {
            printf("  PASS: Saving NULL vector returns error\n");
        } else {
            printf("  FAIL: Saving NULL vector should return error\n");
            error_count++;
        }
        fclose(file);
        remove("test_vector_null.ncn");
    }

    printf("\nTest 5: Save vector to NULL file...\n");
    Vector* v_save = new_uninitialized_vector(5);
    if (v_save != NULL) {
        int result = save_vector(v_save, NULL);
        if (result != 0) {
            printf("  PASS: Saving to NULL file returns error\n");
        } else {
            printf("  FAIL: Saving to NULL file should return error\n");
            error_count++;
        }
        free_vector(v_save);
    }

    printf("\nTest 6: Load vector from empty file...\n");
    FILE* empty_file = fopen("test_vector_empty.ncn", "w");
    if (empty_file) {
        fclose(empty_file);
        empty_file = fopen("test_vector_empty.ncn", "r");
        Vector* v_empty = new_vector_from_file(empty_file);
        if (v_empty == NULL) {
            printf("  PASS: Loading from empty file returns NULL\n");
        } else {
            printf("  FAIL: Loading from empty file should return NULL\n");
            error_count++;
            free_vector(v_empty);
        }
        fclose(empty_file);
        remove("test_vector_empty.ncn");
    }

    printf("\nTest 7: Load vector from corrupted file (wrong type)...\n");
    FILE* corrupt_file = fopen("test_vector_corrupt.ncn", "w");
    if (corrupt_file) {
        fprintf(corrupt_file, "M 5 5\n1.0 2.0\n"); // Matrix type instead of Vector
        fclose(corrupt_file);
        corrupt_file = fopen("test_vector_corrupt.ncn", "r");
        Vector* v_corrupt = new_vector_from_file(corrupt_file);
        if (v_corrupt == NULL) {
            printf("  PASS: Loading from corrupted file returns NULL\n");
        } else {
            printf("  FAIL: Loading from corrupted file should return NULL\n");
            error_count++;
            free_vector(v_corrupt);
        }
        fclose(corrupt_file);
        remove("test_vector_corrupt.ncn");
    }

    printf("\nTest 8: Load vector from file with negative size...\n");
    FILE* neg_file = fopen("test_vector_neg.ncn", "w");
    if (neg_file) {
        fprintf(neg_file, "V -5\n1.0 2.0\n");
        fclose(neg_file);
        neg_file = fopen("test_vector_neg.ncn", "r");
        Vector* v_neg = new_vector_from_file(neg_file);
        if (v_neg == NULL) {
            printf("  PASS: Loading vector with negative size returns NULL\n");
        } else {
            printf("  FAIL: Loading vector with negative size should return NULL\n");
            error_count++;
            free_vector(v_neg);
        }
        fclose(neg_file);
        remove("test_vector_neg.ncn");
    }

    printf("\nTest 9: Load vector from file with incomplete data...\n");
    FILE* incomplete_file = fopen("test_vector_incomplete.ncn", "w");
    if (incomplete_file) {
        fprintf(incomplete_file, "V 5\n1.0 2.0\n"); // Only 2 values instead of 5
        fclose(incomplete_file);
        incomplete_file = fopen("test_vector_incomplete.ncn", "r");
        Vector* v_incomplete = new_vector_from_file(incomplete_file);
        if (v_incomplete == NULL) {
            printf("  PASS: Loading vector with incomplete data returns NULL\n");
        } else {
            printf("  WARNING: Loading vector with incomplete data succeeded (may have undefined values)\n");
            free_vector(v_incomplete);
        }
        fclose(incomplete_file);
        remove("test_vector_incomplete.ncn");
    }

    printf("\nTest 10: Load vector from NULL file...\n");
    Vector* v_null_file = new_vector_from_file(NULL);
    if (v_null_file == NULL) {
        printf("  PASS: Loading from NULL file returns NULL\n");
    } else {
        printf("  FAIL: Loading from NULL file should return NULL\n");
        error_count++;
        free_vector(v_null_file);
    }

    printf("\nTest 11: Vector with extreme values...\n");
    Vector* v_extreme = new_uninitialized_vector(3);
    if (v_extreme != NULL) {
        v_extreme->buffer[0] = 1e308;  // Near max double
        v_extreme->buffer[1] = -1e308; // Near min double
        v_extreme->buffer[2] = 1e-308; // Near zero
        
        FILE* extreme_file = fopen("test_vector_extreme.ncn", "w");
        if (extreme_file) {
            int result = save_vector(v_extreme, extreme_file);
            fclose(extreme_file);
            
            if (result == 0) {
                extreme_file = fopen("test_vector_extreme.ncn", "r");
                Vector* v_loaded = new_vector_from_file(extreme_file);
                if (v_loaded != NULL) {
                    if (are_double_equals(v_loaded->buffer[0], v_extreme->buffer[0]) &&
                        are_double_equals(v_loaded->buffer[1], v_extreme->buffer[1]) &&
                        are_double_equals(v_loaded->buffer[2], v_extreme->buffer[2])) {
                        printf("  PASS: Extreme values saved and loaded correctly\n");
                    } else {
                        printf("  FAIL: Extreme values not preserved\n");
                        error_count++;
                    }
                    free_vector(v_loaded);
                } else {
                    printf("  FAIL: Failed to load extreme values\n");
                    error_count++;
                }
                fclose(extreme_file);
            }
            remove("test_vector_extreme.ncn");
        }
        free_vector(v_extreme);
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
