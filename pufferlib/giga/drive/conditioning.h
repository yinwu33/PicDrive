// Per-agent conditioning: the reward weights and dynamics coefficients that make one
// agent behave differently from another.
//
// This is the mechanism Gigaflow uses instead of a policy pool. A single network at
// current parameters drives every agent, so without conditioning all traffic would be
// homogeneous and the policy would learn to expect a mirror of itself. Each agent
// instead draws its own reward weights and embodiment, and -- the part that matters --
// "the policy only observes the conditioning C of the agent it controls" (Sec. 2). The
// goals, conditioning and momentary accelerations of *other* agents are hidden, so the
// policy has to cope with traffic whose intentions it cannot read. That is what
// produces cautious drivers next to aggressive ones, and it is why there is no league
// or opponent sampling anywhere in the paper.
//
// Values are Gigaflow Table A2. Two deviations, both deliberate:
//   * alpha_stop_line and alpha_overspeed are absent. They weight terms that need
//     stop lines and lane speed limits, and neither exists in the WOMD map format
//     nor in the intermediate JSON, so the terms are not implemented either.
//   * v_goal is randomized over U(0,20) rather than fixed at 3 m/s, following the
//     value recorded for this project in Agents.md Sec. 3.
#ifndef GIGA_CONDITIONING_H
#define GIGA_CONDITIONING_H

#include "giga_random.h"

// Reward weights (randomized, observed).
#define COND_DELTA_GOAL 0        // goal collection radius, m
#define COND_V_GOAL 1            // speed below which the final goal counts, m/s
#define COND_ALPHA_COLLISION 2   // risk tolerance: aggressive .. conservative
#define COND_ALPHA_BOUNDARY 3    // off-road aversion
#define COND_ALPHA_COMFORT 4     // jerk/acceleration aversion
#define COND_ALPHA_L_ALIGN 5     // lane heading alignment
#define COND_ALPHA_VEL_ALIGN 6   // penalty for travelling against lane heading
#define COND_ALPHA_L_CENTER 7    // lane centering
#define COND_ALPHA_CENTER_BIAS 8 // preferred offset within the lane, m (left/right)
#define COND_ALPHA_REVERSE 9     // reversing aversion
// Dynamics coefficients (randomized, observed; the paper puts these in the ego state).
#define COND_C_THROTTLE 10
#define COND_C_STEER 11
#define COND_C_ACC 12

#define GIGA_NUM_COND 13
#define GIGA_NUM_COND_UNIFORM 10 // the first ten are plain uniforms

// Fixed weights. Constant across agents, so conditioning on them would carry no
// information and they are left out of the observation.
#define GIGA_ALPHA_VELOCITY 0.0025f
#define GIGA_ALPHA_TIMESTEP 0.000025f

static const float GIGA_COND_LO[GIGA_NUM_COND] = {
    2.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.00025f, 0.0f, 0.00025f, -0.5f, 0.00025f,
    0.8f,           // C_throttle, X(1.25)
    0.8f,           // C_steer,    X(1.25)
    2.0f / 3.0f,    // C_acc,      X(1.5)
};
static const float GIGA_COND_HI[GIGA_NUM_COND] = {
    12.0f, 20.0f, 3.0f, 3.0f, 0.1f, 0.025f, 1.0f, 0.0075f, 0.5f, 0.0075f,
    1.25f, 1.25f, 1.5f,
};

// Drawn per agent per life -- at episode reset and again on every respawn, since a
// respawned agent is a new road user rather than the same one continuing.
static void giga_sample_conditioning(GigaRng *rng, float *cond) {
    for (int i = 0; i < GIGA_NUM_COND_UNIFORM; i++)
        cond[i] = giga_rand_range(rng, GIGA_COND_LO[i], GIGA_COND_HI[i]);
    // X(a) rather than U(1/a, a): equally many samples below and above 1, so the
    // randomization does not bias the fleet toward being more capable than nominal.
    cond[COND_C_THROTTLE] = giga_rand_xmix(rng, 1.25f);
    cond[COND_C_STEER] = giga_rand_xmix(rng, 1.25f);
    cond[COND_C_ACC] = giga_rand_xmix(rng, 1.5f);
}

// To [0, 1] for the observation. The raw values span four orders of magnitude
// (2.5e-5 to 20), which no shared scale would survive.
static inline float giga_cond_norm(int i, float v) {
    float span = GIGA_COND_HI[i] - GIGA_COND_LO[i];
    return (span > 0.0f) ? (v - GIGA_COND_LO[i]) / span : 0.0f;
}

#endif // GIGA_CONDITIONING_H
