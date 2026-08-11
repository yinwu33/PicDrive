// Per-environment random number generation.
//
// The dataset-driven env got away with the global libc `rand()`, because the only
// thing it sampled was which map to load. Gigaflow-style initialization samples
// every agent's pose, size, route and reward weights, so the stream has to be
// per-env and seedable: the visualization acceptance check in viz.py is only
// meaningful if the same seed reproduces the same scene, and `rand()` is process
// global and shared across all ~34 concurrent scenes.
//
// PCG32 (O'Neill 2014). Chosen over xorshift for its much better low-bit quality --
// `sample_lane_uniform` and the agent-type draw both use the low bits directly.
#ifndef TEDDY_RANDOM_H
#define TEDDY_RANDOM_H

#include <math.h>
#include <stdint.h>

typedef struct {
    uint64_t state;
    uint64_t inc; // stream selector, must be odd
} TeddyRng;

static inline uint32_t teddy_rand_u32(TeddyRng *rng) {
    uint64_t old = rng->state;
    rng->state = old * 6364136223846793005ULL + rng->inc;
    uint32_t xorshifted = (uint32_t)(((old >> 18u) ^ old) >> 27u);
    uint32_t rot = (uint32_t)(old >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

static inline void teddy_rand_seed(TeddyRng *rng, uint64_t seed, uint64_t stream) {
    rng->state = 0u;
    rng->inc = (stream << 1u) | 1u;
    teddy_rand_u32(rng);
    rng->state += seed;
    teddy_rand_u32(rng);
}

// Uniform in [0, 1).
static inline float teddy_rand_float(TeddyRng *rng) {
    // 24 bits is the full mantissa of a float; taking more would just round.
    return (float)(teddy_rand_u32(rng) >> 8) * (1.0f / 16777216.0f);
}

// Uniform in [lo, hi).
static inline float teddy_rand_range(TeddyRng *rng, float lo, float hi) {
    return lo + (hi - lo) * teddy_rand_float(rng);
}

// Uniform integer in [lo, hi] inclusive. Debiased by rejection, so the small
// integer ranges used for agent counts and waypoint counts stay exactly uniform.
static inline int teddy_rand_int(TeddyRng *rng, int lo, int hi) {
    if (hi <= lo)
        return lo;
    uint32_t span = (uint32_t)(hi - lo) + 1u;
    uint32_t limit = UINT32_MAX - (UINT32_MAX % span) - 1u;
    uint32_t r;
    do {
        r = teddy_rand_u32(rng);
    } while (r > limit);
    return lo + (int)(r % span);
}

// Standard normal, Box-Muller. Only used for small heading jitter, so the cost of
// discarding the second variate does not matter.
static inline float teddy_rand_normal(TeddyRng *rng) {
    float u1 = teddy_rand_float(rng);
    float u2 = teddy_rand_float(rng);
    if (u1 < 1e-7f)
        u1 = 1e-7f;
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

// Gigaflow's X(a) = 0.5*U(1/a, 1) + 0.5*U(1, a), a > 1 (paper App. B.2). Generates
// as many samples below one as above it, so multiplicative dynamics coefficients
// are randomized symmetrically rather than biased upward the way a plain U(1/a, a)
// would be.
static inline float teddy_rand_xmix(TeddyRng *rng, float a) {
    if (a <= 1.0f)
        return 1.0f;
    if (teddy_rand_u32(rng) & 1u)
        return teddy_rand_range(rng, 1.0f / a, 1.0f);
    return teddy_rand_range(rng, 1.0f, a);
}

#endif // TEDDY_RANDOM_H
