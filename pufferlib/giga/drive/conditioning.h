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
// Values are Gigaflow Table A2. One deviation: alpha_stop_line is absent, because
// WOMD carries no stop lines and R_stop-line is not implemented either.
#ifndef GIGA_CONDITIONING_H
#define GIGA_CONDITIONING_H

#include "giga_random.h"

// Reward weights (randomized, observed).
#define COND_DELTA_GOAL 0        // goal collection radius, m
#define COND_SLOT_IS_FINAL 1     // not conditioning: carries the is_final waypoint flag
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
#define GIGA_V_GOAL 3.0f

// Saturation distance for R_l-center, metres. Not a tuning knob so much as the
// limit of what the simulator can see: find_closest_lane only sweeps the 5x5 grid
// neighbourhood (GRID_CELL_SIZE = 5 m), so past roughly 10 m there is no nearest
// centerline to measure against at all. It must stay above the largest offset the
// 4 m lane_valid gate can produce -- |offset| <= 4 m plus |center_bias| <= 0.5 m,
// so d <= 4.5 -- which is what keeps every state that was already being rewarded
// unchanged to the bit, and below the search horizon so the cap is actually
// reached before the reference disappears.
#define GIGA_LANE_D_CAP 6.0f

// The three indicator penalties (collision, off-road, comfort) are the paper's values
// divided by 3. They are not scaled by dt, so at dt=0.1 they would be charged three
// times as often per second as at the paper's dt=0.3. Note R_collision's 0.1*|v| term
// lives in drive.h and is NOT scaled: at 20 m/s it contributes -2.0 against
// alpha_collision's -1.0, so speed now outweighs the risk-tolerance knob.
//
// Slot 1 held v_goal, which is now the constant GIGA_V_GOAL. The row stays because
// both arrays are indexed positionally, and LO == HI keeps giga_cond_norm at 0 so
// drive.h can write the is_final flag over it. Give it a range and that breaks.
//
// R_l-align and R_l-center are at their paper ranges. They used to sit inside
// `if (a->lane_valid)` in drive.h, which drops 4 m from the nearest centerline, and
// that made leaving the lane cheaper than driving imperfectly inside one: measured
// at -0.0039/step outside against -0.0073/step inside, and a trained policy duly
// collapsed to lane_alignment_rate 0.035 for 700 epochs. Both terms now extend past
// the gate -- see the reward block in drive.h -- so these bounds are live as
// written rather than describing a term the agent can switch off by leaving.
static const float GIGA_COND_LO[GIGA_NUM_COND] = {
    2.0f,        // delta_goal: goal-reach radius lower bound, m
    0.0f,        // COND_SLOT_IS_FINAL placeholder: must equal HI
    0.0f,        // alpha_collision: collision penalty weight lower bound
    0.0f,        // alpha_boundary: off-road penalty weight lower bound
    0.0f,        // alpha_comfort: acceleration/jerk penalty weight lower bound
    0.00025f,    // alpha_l_align: lane-heading alignment weight
    0.0f,        // alpha_vel_align: wrong-way motion multiplier
    0.00025f,    // alpha_l_center: lane-centering weight
    -0.5f,       // alpha_center_bias: preferred lateral offset lower bound, m (right)
    0.00025f,    // alpha_reverse: reversing penalty weight lower bound
    0.8f,        // C_throttle normalization bound; sampled with X(1.25)
    0.8f,        // C_steer normalization bound; sampled with X(1.25)
    2.0f / 3.0f, // C_acc normalization bound; sampled with X(1.5)
};

static const float GIGA_COND_HI[GIGA_NUM_COND] = {
    12.0f,   // delta_goal: goal-reach radius upper bound, m
    0.0f,    // COND_SLOT_IS_FINAL placeholder: must equal LO
    1.0f,    // alpha_collision: paper 3.0 / 3, for dt=0.1 (paper trains at 0.3)
    1.0f,    // alpha_boundary: paper 3.0 / 3, same reason
    0.0025f,   // alpha_comfort: paper 0.1 / 3, same reason
    0.0025f,  // alpha_l_align: lane-heading alignment weight
    1.0f,    // alpha_vel_align: wrong-way motion multiplier
    0.0075f, // alpha_l_center: lane-centering weight
    0.5f,    // alpha_center_bias: preferred lateral offset upper bound, m (left)
    0.0075f, // alpha_reverse: reversing penalty weight upper bound
    1.25f,   // C_throttle normalization bound; sampled with X(1.25)
    1.25f,   // C_steer normalization bound; sampled with X(1.25)
    1.5f,    // C_acc normalization bound; sampled with X(1.5)
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
