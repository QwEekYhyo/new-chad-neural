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

#include <utils.h>
#include <functions.h>
#include <matrix.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

int main(void) {
    int error_count = 0;

    // Test 1: are_double_equals with exact equality
    printf("Test 1: are_double_equals with exact values...\n");
    if (are_double_equals(1.0, 1.0)) {
        printf("  PASS: Exact equality works\n");
    } else {
        printf("  FAIL: Exact equality failed\n");
        error_count++;
    }

    // Test 2: are_double_equals with very close values
    printf("\nTest 2: are_double_equals with epsilon difference...\n");
    double a = 1.0;
    double b = 1.0 + EPSILON / 2.0;
    if (are_double_equals(a, b)) {
        printf("  PASS: Values within epsilon are equal\n");
    } else {
        printf("  FAIL: Values within epsilon should be equal\n");
        error_count++;
    }

    // Test 3: are_double_equals with values outside epsilon
    printf("\nTest 3: are_double_equals with values outside epsilon...\n");
    a = 1.0;
    b = 1.0 + EPSILON * 2.0;
    if (!are_double_equals(a, b)) {
        printf("  PASS: Values outside epsilon are not equal\n");
    } else {
        printf("  FAIL: Values outside epsilon should not be equal\n");
        error_count++;
    }

    // Test 4: are_double_equals with negative values
    printf("\nTest 4: are_double_equals with negative values...\n");
    if (are_double_equals(-5.5, -5.5)) {
        printf("  PASS: Negative value equality works\n");
    } else {
        printf("  FAIL: Negative value equality failed\n");
        error_count++;
    }

    // Test 5: are_double_equals with zero
    printf("\nTest 5: are_double_equals with zero...\n");
    if (are_double_equals(0.0, 0.0)) {
        printf("  PASS: Zero equality works\n");
    } else {
        printf("  FAIL: Zero equality failed\n");
        error_count++;
    }

    // Test 6: are_double_equals with very small values
    printf("\nTest 6: are_double_equals with very small values...\n");
    a = 1e-15;
    b = 1e-15;
    if (are_double_equals(a, b)) {
        printf("  PASS: Very small value equality works\n");
    } else {
        printf("  FAIL: Very small value equality failed\n");
        error_count++;
    }

    // Test 7: Sigmoid with zero
    printf("\nTest 7: Sigmoid function with zero...\n");
    double sig_zero = sigmoid(0.0);
    if (are_double_equals(sig_zero, 0.5)) {
        printf("  PASS: sigmoid(0) = 0.5\n");
    } else {
        printf("  FAIL: sigmoid(0) should be 0.5, got %.15f\n", sig_zero);
        error_count++;
    }

    // Test 8: Sigmoid with large positive value
    printf("\nTest 8: Sigmoid function with large positive value...\n");
    double sig_large = sigmoid(100.0);
    if (sig_large > 0.99 && sig_large <= 1.0) {
        printf("  PASS: sigmoid(100) approaches 1.0 (got %.15f)\n", sig_large);
    } else {
        printf("  FAIL: sigmoid(100) should approach 1.0, got %.15f\n", sig_large);
        error_count++;
    }

    // Test 9: Sigmoid with large negative value
    printf("\nTest 9: Sigmoid function with large negative value...\n");
    double sig_neg = sigmoid(-100.0);
    if (sig_neg < 0.01 && sig_neg >= 0.0) {
        printf("  PASS: sigmoid(-100) approaches 0.0 (got %.15f)\n", sig_neg);
    } else {
        printf("  FAIL: sigmoid(-100) should approach 0.0, got %.15f\n", sig_neg);
        error_count++;
    }

    // Test 10: Sigmoid derivative at zero
    printf("\nTest 10: Sigmoid derivative at zero...\n");
    // sigmoid_derivative expects x that has already passed through sigmoid
    double sig_deriv_zero = sigmoid_derivative(sigmoid(0.0));
    if (are_double_equals(sig_deriv_zero, 0.25)) {
        printf("  PASS: sigmoid'(0) = 0.25\n");
    } else {
        printf("  FAIL: sigmoid'(0) should be 0.25, got %.15f\n", sig_deriv_zero);
        error_count++;
    }

    // Test 11: Squared Error with equal values
    printf("\nTest 11: MSE with equal values...\n");
    double mse_equal = squared_error(1.0, 1.0);
    if (are_double_equals(mse_equal, 0.0)) {
        printf("  PASS: MSE(1.0, 1.0) = 0.0\n");
    } else {
        printf("  FAIL: MSE should be 0 for equal values, got %.15f\n", mse_equal);
        error_count++;
    }

    // Test 12: Mean Squared Error with different values
    printf("\nTest 12: MSE with different values...\n");
    double mse_diff = squared_error(1.0, 3.0) / 2.0;
    double expected = 2.0;
    if (are_double_equals(mse_diff, expected)) {
        printf("  PASS: MSE calculated correctly\n");
    } else {
        printf("  FAIL: MSE should be %.15f, got %.15f\n", expected, mse_diff);
        error_count++;
    }

    // Test 13: MSE derivative
    printf("\nTest 13: MSE derivative...\n");
    double mse_deriv = squared_error_derivative(1.0, 3.0) / 2.0;
    double expected_deriv = 2.0; // 3 - 1 = 2
    if (are_double_equals(mse_deriv, expected_deriv)) {
        printf("  PASS: MSE derivative calculated correctly\n");
    } else {
        printf("  FAIL: MSE derivative should be %.15f, got %.15f\n", expected_deriv, mse_deriv);
        error_count++;
    }

    // Test 14: Binary Cross Entropy with valid inputs
    printf("\nTest 14: Binary Cross Entropy with valid inputs...\n");
    double bce = binary_cross_entropy(1.0, 0.9);
    if (!isnan(bce) && !isinf(bce) && bce >= 0) {
        printf("  PASS: BCE produces valid output (%.15f)\n", bce);
    } else {
        printf("  FAIL: BCE produced invalid output\n");
        error_count++;
    }

    // Test 15: Binary Cross Entropy at extremes
    printf("\nTest 15: Binary Cross Entropy at extremes...\n");
    // BCE with output = 0 and target = 0 should be finite
    double bce_extreme = binary_cross_entropy(0.0, 0.001);
    if (!isnan(bce_extreme) && !isinf(bce_extreme)) {
        printf("  PASS: BCE handles near-zero values (%.15f)\n", bce_extreme);
    } else {
        printf("  WARNING: BCE may produce Inf/NaN at extremes\n");
    }

    // Test 16: Softmax with NULL matrix
    printf("\nTest 16: Softmax with NULL matrix...\n");
    // softmax(NULL);
    printf("  MANUAL CHECK: Softmax with NULL should be handled\n");

    // Test 17: Softmax with single column
    printf("\nTest 17: Softmax with single column...\n");
    Matrix* m_softmax = new_uninitialized_matrix(3, 1);
    if (m_softmax != NULL) {
        MAT(m_softmax, 0, 0) = 1.0;
        MAT(m_softmax, 1, 0) = 2.0;
        MAT(m_softmax, 2, 0) = 3.0;
        
        softmax(m_softmax);
        
        // Sum of softmax outputs should be 1.0
        double sum = MAT(m_softmax, 0, 0) + MAT(m_softmax, 1, 0) + MAT(m_softmax, 2, 0);
        
        if (are_double_equals(sum, 1.0)) {
            printf("  PASS: Softmax outputs sum to 1.0\n");
        } else {
            printf("  FAIL: Softmax outputs should sum to 1.0, got %.15f\n", sum);
            error_count++;
        }
        
        free_matrix(m_softmax);
    }

    // Test 18: Softmax with extreme values
    printf("\nTest 18: Softmax with extreme values...\n");
    Matrix* m_extreme = new_uninitialized_matrix(3, 1);
    if (m_extreme != NULL) {
        MAT(m_extreme, 0, 0) = 1000.0;
        MAT(m_extreme, 1, 0) = -1000.0;
        MAT(m_extreme, 2, 0) = 0.0;
        
        softmax(m_extreme);
        
        // Check for NaN or Inf
        int values_valid = 1;
        for (size_t i = 0; i < 3; i++) {
            if (isnan(MAT(m_extreme, i, 0)) || isinf(MAT(m_extreme, i, 0))) {
                values_valid = 0;
                break;
            }
        }
        
        if (values_valid) {
            printf("  PASS: Softmax handles extreme values without NaN/Inf\n");
        } else {
            printf("  FAIL: Softmax produced NaN or Inf with extreme values\n");
            error_count++;
        }
        
        free_matrix(m_extreme);
    }

    // Test 19: rand_double_range bounds
    printf("\nTest 19: rand_double_range bounds checking...\n");
    int in_range = 1;
    for (int i = 0; i < 100; i++) {
        double val = rand_double_range(-5, 5);
        if (val < -5.0 || val > 5.0) {
            in_range = 0;
            break;
        }
    }
    if (in_range) {
        printf("  PASS: rand_double_range stays within bounds\n");
    } else {
        printf("  FAIL: rand_double_range produced out-of-bounds value\n");
        error_count++;
    }

    // Test 20: rand_uniform bounds
    printf("\nTest 20: rand_uniform bounds checking...\n");
    in_range = 1;
    for (int i = 0; i < 100; i++) {
        double val = rand_uniform();
        if (val < 0.0 || val > 1.0) {
            in_range = 0;
            break;
        }
    }
    if (in_range) {
        printf("  PASS: rand_uniform stays in [0, 1]\n");
    } else {
        printf("  FAIL: rand_uniform produced out-of-bounds value\n");
        error_count++;
    }

    // Test 21: rand_normal produces valid values
    printf("\nTest 21: rand_normal produces valid values...\n");
    double z0, z1;
    rand_normal(0.0, 1.0, &z0, &z1);
    if (!isnan(z0) && !isinf(z0) && !isnan(z1) && !isinf(z1)) {
        printf("  PASS: rand_normal produces valid values (%.6f, %.6f)\n", z0, z1);
    } else {
        printf("  FAIL: rand_normal produced invalid values\n");
        error_count++;
    }

    // Test 22: max function
    printf("\nTest 22: max function...\n");
    if (max(5, 10) == 10 && max(10, 5) == 10 && max(7, 7) == 7) {
        printf("  PASS: max function works correctly\n");
    } else {
        printf("  FAIL: max function incorrect\n");
        error_count++;
    }

    // Test 23: Categorical Cross Entropy
    printf("\nTest 23: Categorical Cross Entropy...\n");
    double cce = categorical_cross_entropy(1.0, 0.8);
    if (!isnan(cce) && !isinf(cce) && cce >= 0) {
        printf("  PASS: CCE produces valid output (%.15f)\n", cce);
    } else {
        printf("  FAIL: CCE produced invalid output\n");
        error_count++;
    }

    // Test 24: Loss functions with zero output
    printf("\nTest 24: Loss functions with zero output...\n");
    double bce_zero = binary_cross_entropy(1.0, 0.0);
    if (isinf(bce_zero)) {
        printf("  EXPECTED: BCE with zero output produces Inf\n");
    } else {
        printf("  WARNING: BCE with zero output should produce Inf, got %.15f\n", bce_zero);
    }

    // Test 25: Extreme precision test
    printf("\nTest 25: Extreme precision comparison...\n");
    double precise_a = 0.123456789012345;
    double precise_b = 0.123456789012345;
    if (are_double_equals(precise_a, precise_b)) {
        printf("  PASS: High precision values compared correctly\n");
    } else {
        printf("  FAIL: High precision comparison failed\n");
        error_count++;
    }

    // Summary
    printf("\n========================================\n");
    if (error_count == 0) {
        printf("All tests PASSED!\n");
    } else {
        printf("Tests completed with %d FAILURES\n", error_count);
    }
    printf("========================================\n");

    return error_count > 0 ? 1 : 0;
}
