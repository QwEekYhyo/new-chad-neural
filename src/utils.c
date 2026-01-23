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

#define _USE_MATH_DEFINES
#include <utils.h>

#include <stdlib.h>

double rand_double_range(int min, int max) {
#if defined(_POSIX_VERSION)
    int n = random();
#else
    int n = rand();
#endif
    double scale = n / (double) RAND_MAX;
    return min + scale * (max - min);
}

// Generate uniformly distributed random numbers (between 0 and 1)
double rand_uniform(void) {
    return (double) rand() / RAND_MAX;
}

// Generate normally distributed numbers using Box-Muller transform
// This generates two numbers to save 1 log and 1 square root computation every 2 generated numbers
//   mean   - mean of distribution
//   stddev - standard deviation
//   z0 and z1 are where the results are put
void rand_normal(double mean, double stddev, double* z0, double* z1) {
    const double TWO_PI = 2.0 * M_PI;

    double u1 = rand_uniform();
    double u2 = rand_uniform();
    
    double mag = stddev * sqrt(-2.0 * log(u1));
    if (z0)
        *z0 = mag * cos(TWO_PI * u2) + mean;
    if (z1)
        *z1 = mag * sin(TWO_PI * u2) + mean;
}
