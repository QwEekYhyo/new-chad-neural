#ifndef NCN_COMMON_DEFS_H
#define NCN_COMMON_DEFS_H

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
#else
typedef unsigned long size_t;
#endif
typedef double (*activation_function)(double);

#endif // NCN_COMMON_DEFS_H
