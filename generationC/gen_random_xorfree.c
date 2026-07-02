#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>

/*
 * XOR-free row-weight-corrected random matrix generator.
 *
 * For each non-empty line read from stdin, this program outputs one binary
 * matrix A of size k x nA, where
 *
 *     k  = n - m*t,
 *     nA = n - k = m*t.
 *
 * The generated matrix satisfies two constraints:
 *
 *   1. Row-weight constraint:
 *        for every row A_i,
 *        w_H(A_i) >= 2*t.
 *
 *   2. Pairwise XOR constraint:
 *        for every pair of distinct rows A_i, A_j,
 *        w_H(A_i + A_j) >= 2*t - 1.
 *
 * These constraints correspond to necessary conditions induced by the minimum
 * distance of binary Goppa codes when the full generator matrix is written in
 * systematic form G = (I_k | A).
 *
 * Output format for each generated matrix:
 *   - uint32 little-endian: k
 *   - uint32 little-endian: nA
 *   - k*nA bytes: entries of A in row-major order, each byte being 0 or 1
 */

/* Rotate a 64-bit integer left by k bits. */
static uint64_t rotl64(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

/*
 * SplitMix64 generator.
 *
 * Used to expand a single 64-bit seed into the 256-bit internal state required
 * by xoshiro256**.
 */
static uint64_t splitmix64(uint64_t *x) {
    uint64_t z = (*x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/* Internal state of the xoshiro256** pseudo-random generator. */
typedef struct {
    uint64_t s[4];
} rng_t;

/* Initialize the PRNG state from a 64-bit seed. */
static void rng_seed(rng_t *rng, uint64_t seed) {
    uint64_t x = seed;
    for (int i = 0; i < 4; i++) {
        rng->s[i] = splitmix64(&x);
    }
}

/*
 * Return the next 64 pseudo-random bits.
 *
 * This is the xoshiro256** generator. It is fast and suitable for simulation
 * and dataset generation, but it is not intended for cryptographic use.
 */
static uint64_t rng_next_u64(rng_t *rng) {
    const uint64_t result = rotl64(rng->s[1] * 5ULL, 7) * 9ULL;
    const uint64_t t = rng->s[1] << 17;

    rng->s[2] ^= rng->s[0];
    rng->s[3] ^= rng->s[1];
    rng->s[1] ^= rng->s[2];
    rng->s[0] ^= rng->s[3];

    rng->s[2] ^= t;
    rng->s[3] = rotl64(rng->s[3], 45);

    return result;
}

/*
 * Obtain a 64-bit seed from the operating system.
 *
 * The function first tries to read from /dev/urandom. If this fails, it falls
 * back to a seed derived from the current time and process id.
 */
static uint64_t os_random_seed(void) {
    uint64_t seed = 0;

    int fd = open("/dev/urandom", O_RDONLY);
    if (fd >= 0) {
        ssize_t r = read(fd, &seed, sizeof(seed));
        close(fd);

        if (r == (ssize_t)sizeof(seed)) {
            return seed;
        }
    }

    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);

    seed = ((uint64_t)ts.tv_sec << 32)
         ^ (uint64_t)ts.tv_nsec
         ^ (uint64_t)getpid();

    return seed;
}

/* Write a 32-bit unsigned integer in little-endian order. */
static int write_u32_le(FILE *out, uint32_t x) {
    unsigned char b[4];

    b[0] = (unsigned char)(x & 0xFFu);
    b[1] = (unsigned char)((x >> 8) & 0xFFu);
    b[2] = (unsigned char)((x >> 16) & 0xFFu);
    b[3] = (unsigned char)((x >> 24) & 0xFFu);

    return fwrite(b, 1, 4, out) == 4 ? 0 : -1;
}

/*
 * Compute the Hamming weight of the bitwise XOR of two binary rows.
 *
 * The rows are stored as arrays of bytes whose entries are expected to be 0 or 1.
 */
static uint32_t row_xor_weight(
    const unsigned char *a,
    const unsigned char *b,
    uint32_t nA
) {
    uint32_t w = 0;

    for (uint32_t c = 0; c < nA; c++) {
        w += (uint32_t)(a[c] ^ b[c]);
    }

    return w;
}

/*
 * Check whether the matrix contains a pairwise XOR violation.
 *
 * A violation occurs if there exist two distinct rows A_i and A_j such that
 *
 *     w_H(A_i + A_j) < 2*t - 1.
 *
 * Return 1 if a violation is found, and 0 otherwise.
 */
static int has_pair_xor_violation(
    const unsigned char *buf,
    uint32_t k,
    uint32_t nA,
    uint32_t t
) {
    const uint32_t threshold = 2u * t - 1u;

    for (uint32_t i = 0; i + 1 < k; i++) {
        const unsigned char *row_i = buf + (size_t)i * nA;

        for (uint32_t j = i + 1; j < k; j++) {
            const unsigned char *row_j = buf + (size_t)j * nA;
            uint32_t w = row_xor_weight(row_i, row_j, nA);

            if (w < threshold) {
                return 1;
            }
        }
    }

    return 0;
}

/*
 * Fill a matrix with row-weight-corrected random rows.
 *
 * Each row is sampled uniformly at random from {0,1}^nA conditioned on
 *
 *     w_H(row) >= 2*t.
 *
 * This function does not enforce the pairwise XOR constraint. That constraint
 * is checked separately on the whole matrix.
 *
 * Return 0 on success and -1 on failure.
 */
static int fill_random_matrix(
    unsigned char *buf,
    rng_t *rng,
    uint32_t k,
    uint32_t nA,
    uint32_t t
) {
    const uint32_t min_row_weight = 2u * t;

    if (min_row_weight > nA) {
        fprintf(stderr,
                "Impossible: require row weight >= %u but nA=%u\n",
                min_row_weight,
                nA);
        return -1;
    }

    for (uint32_t row = 0; row < k; row++) {
        unsigned char *row_ptr = buf + (size_t)row * nA;

        /*
         * Rejection sampling at the row level:
         * keep resampling the current row until its Hamming weight is large
         * enough.
         */
        while (1) {
            uint32_t w = 0;
            size_t i = 0;

            /* Fill the row 64 bits at a time when possible. */
            while (i + 64 <= nA) {
                uint64_t r = rng_next_u64(rng);

                for (int bit = 0; bit < 64; bit++) {
                    unsigned char b = (unsigned char)((r >> bit) & 1ULL);
                    row_ptr[i + (size_t)bit] = b;
                    w += (uint32_t)b;
                }

                i += 64;
            }

            /* Fill the remaining coordinates, if nA is not a multiple of 64. */
            if (i < nA) {
                uint64_t r = rng_next_u64(rng);

                while (i < nA) {
                    unsigned char b = (unsigned char)(r & 1ULL);
                    row_ptr[i] = b;
                    w += (uint32_t)b;
                    r >>= 1;
                    i++;
                }
            }

            if (w >= min_row_weight) {
                break;
            }
        }
    }

    return 0;
}

/*
 * Write one binary matrix to the output stream.
 *
 * The matrix is written as:
 *   - k as a 32-bit little-endian integer,
 *   - nA as a 32-bit little-endian integer,
 *   - the raw k*nA matrix entries in row-major order.
 *
 * Return 0 on success and -1 on failure.
 */
static int write_matrix(
    FILE *out,
    const unsigned char *buf,
    uint32_t k,
    uint32_t nA
) {
    const size_t size = (size_t)k * (size_t)nA;

    if (write_u32_le(out, k) != 0 || write_u32_le(out, nA) != 0) {
        return -1;
    }

    if (fwrite(buf, 1, size, out) != size) {
        return -1;
    }

    return 0;
}

/*
 * Generate and write one XOR-free row-weight-corrected random matrix.
 *
 * The generation is performed by rejection sampling at the matrix level:
 *
 *   1. Generate a row-weight-corrected random matrix.
 *   2. Check all pairwise XOR constraints.
 *   3. Reject the whole matrix if at least one violation is found.
 *
 * This can become expensive when XOR-free matrices are rare.
 *
 * Return 0 on success and -1 on failure.
 */
static int emit_random_matrix_xorfree(
    FILE *out,
    rng_t *rng,
    uint32_t k,
    uint32_t nA,
    uint32_t t
) {
    const size_t size = (size_t)k * (size_t)nA;
    unsigned char *buf = (unsigned char *)malloc(size);

    if (!buf) {
        return -1;
    }

    uint64_t attempts = 0;

    while (1) {
        attempts++;

        if (fill_random_matrix(buf, rng, k, nA, t) != 0) {
            free(buf);
            return -1;
        }

        if (!has_pair_xor_violation(buf, k, nA, t)) {
            break;
        }

    }

    if (write_matrix(out, buf, k, nA) != 0) {
        free(buf);
        return -1;
    }

    free(buf);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s m t n [ignored]\n", argv[0]);
        return 1;
    }

    /*
     * Parse parameters:
     *   m: extension degree,
     *   t: Goppa polynomial degree,
     *   n: code length.
     */
    char *end = NULL;

    long m = strtol(argv[1], &end, 10);
    if (*end != '\0' || m <= 0) {
        return 1;
    }

    long t = strtol(argv[2], &end, 10);
    if (*end != '\0' || t <= 0) {
        return 1;
    }

    long n = strtol(argv[3], &end, 10);
    if (*end != '\0' || n <= 0) {
        return 1;
    }

    /*
     * The full systematic generator matrix is G = (I_k | A).
     * This generator only outputs the non-systematic block A.
     */
    long k_long = n - m * t;
    long nA_long = n - k_long;

    if (k_long <= 0 || nA_long <= 0) {
        return 1;
    }

    uint32_t k = (uint32_t)k_long;
    uint32_t nA = (uint32_t)nA_long;

    /* Use large I/O buffers to reduce overhead when producing many samples. */
    setvbuf(stdin, NULL, _IOFBF, 1 << 20);
    setvbuf(stdout, NULL, _IOFBF, 1 << 20);

    rng_t rng;
    rng_seed(&rng, os_random_seed());

    char *line = NULL;
    size_t cap = 0;
    ssize_t len;

    /*
     * The content of stdin is used only as a trigger:
     * one non-empty input line produces one output matrix.
     */
    while ((len = getline(&line, &cap, stdin)) != -1) {
        int non_empty = 0;

        for (ssize_t i = 0; i < len; i++) {
            if (line[i] != ' '
             && line[i] != '\t'
             && line[i] != '\r'
             && line[i] != '\n') {
                non_empty = 1;
                break;
            }
        }

        if (!non_empty) {
            continue;
        }

        if (emit_random_matrix_xorfree(stdout, &rng, k, nA, (uint32_t)t) != 0) {
            free(line);
            return 1;
        }
    }

    free(line);
    fflush(stdout);

    return 0;
}