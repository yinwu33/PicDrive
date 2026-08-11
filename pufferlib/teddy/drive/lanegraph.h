// Lane graph over the map's ROAD_LANE centerlines.
//
// Must be included from drive.h *after* the Entity struct, which it reads.
//
// Why this exists: Gigaflow spawns agents at random points on the drivable surface
// and gives them goals sampled over the map. Neither the .bin map format nor the
// intermediate WOMD JSON carries lane connectivity -- the upstream converter drops
// `entry_lanes`/`exit_lanes` -- so without a graph there is nothing to sample along
// and nothing to route over.
//
// Connectivity is recovered by endpoint matching, which is sound here because WOMD
// lane polylines share endpoints exactly: measured over 40 training maps, a 0.5 m
// tolerance already finds every join a 2 m tolerance does (70-92% of lanes get a
// successor; the rest genuinely run off the edge of the ~274 m map crop). 2.51% of
// those endpoint matches are the *oncoming* lane rather than a successor, which is
// why the tangent filter below is mandatory and not a refinement.
//
// Gigaflow instead samples uniformly over drivable *polygons* and then corrects a
// bias toward wide road sections. Sampling by arclength along centerlines has no
// such bias to correct, and it also places agents in lanes rather than anywhere on
// the asphalt, which is what we want for a policy meant to transfer to real driving.
#ifndef TEDDY_LANEGRAPH_H
#define TEDDY_LANEGRAPH_H

#include <math.h>
#include <stdlib.h>

#include "teddy_random.h"

#define LANE_JOIN_EPS 0.5f      // metres; see note above on why 0.5 suffices
#define LANE_SUCC_MIN_COS 0.2588f // cos(75 deg): rejects oncoming-lane endpoint matches
#define MAX_LANE_SUCC 8
#define MIN_LANE_LENGTH 1.0f    // shorter polylines are map artifacts, not drivable lanes

typedef struct {
    int entity_idx; // index into env->entities
    int num_points;
    float *cum_s; // [num_points], cumulative arclength, cum_s[0] = 0
    float length;
    int succ[MAX_LANE_SUCC];
    int num_succ;
} Lane;

typedef struct {
    Lane *lanes;
    int num_lanes;
    float *cum_length; // [num_lanes + 1] prefix sum, for arclength-weighted sampling
    float total_length;
} LaneGraph;

// Pose at arclength `s` along `lane_idx`. `s` is clamped into range.
static void lane_pose_at(const LaneGraph *lg, const Entity *entities, int lane_idx, float s, float *out_x,
                         float *out_y, float *out_heading) {
    const Lane *lane = &lg->lanes[lane_idx];
    const Entity *e = &entities[lane->entity_idx];
    if (s < 0.0f)
        s = 0.0f;
    if (s > lane->length)
        s = lane->length;

    // Binary search for the segment containing s.
    int lo = 0, hi = lane->num_points - 1;
    while (hi - lo > 1) {
        int mid = (lo + hi) / 2;
        if (lane->cum_s[mid] <= s)
            lo = mid;
        else
            hi = mid;
    }

    float seg = lane->cum_s[hi] - lane->cum_s[lo];
    float t = (seg > 1e-6f) ? (s - lane->cum_s[lo]) / seg : 0.0f;
    float x0 = e->traj_x[lo], y0 = e->traj_y[lo];
    float x1 = e->traj_x[hi], y1 = e->traj_y[hi];
    *out_x = x0 + t * (x1 - x0);
    *out_y = y0 + t * (y1 - y0);
    *out_heading = atan2f(y1 - y0, x1 - x0);
}

// Uniform by arclength over the whole drivable network, so agent density per metre
// of lane is flat rather than per lane (which would crowd short intersection stubs).
static int sample_lane_point(const LaneGraph *lg, TeddyRng *rng, int *out_lane, float *out_s) {
    if (lg->num_lanes == 0 || lg->total_length <= 0.0f)
        return 0;

    float target = teddy_rand_float(rng) * lg->total_length;
    int lo = 0, hi = lg->num_lanes;
    while (hi - lo > 1) {
        int mid = (lo + hi) / 2;
        if (lg->cum_length[mid] <= target)
            lo = mid;
        else
            hi = mid;
    }
    *out_lane = lo;
    *out_s = target - lg->cum_length[lo];
    if (*out_s > lg->lanes[lo].length)
        *out_s = lg->lanes[lo].length;
    return 1;
}

// Random forward walk of `target_dist` metres along successors. Returns the distance
// actually covered, which is less than requested at a dead end -- callers use that to
// relax their spacing constraints rather than failing, as the paper does.
static float lane_walk_forward(const LaneGraph *lg, TeddyRng *rng, int lane_idx, float s, float target_dist,
                               int *out_lane, float *out_s) {
    float travelled = 0.0f;
    int guard = 0;
    while (travelled < target_dist && guard++ < 64) {
        const Lane *lane = &lg->lanes[lane_idx];
        float avail = lane->length - s;
        float need = target_dist - travelled;
        if (need <= avail) {
            s += need;
            travelled = target_dist;
            break;
        }
        travelled += avail;
        s = lane->length; // consumed the rest of this lane; advance before leaving it
        if (lane->num_succ == 0)
            break; // dead end (usually the map crop boundary)
        lane_idx = lane->succ[teddy_rand_int(rng, 0, lane->num_succ - 1)];
        s = 0.0f;
    }
    *out_lane = lane_idx;
    *out_s = s;
    return travelled;
}

static void build_lane_graph(LaneGraph *lg, const Entity *entities, int num_objects, int num_entities,
                             int road_lane_type) {
    lg->lanes = NULL;
    lg->cum_length = NULL;
    lg->num_lanes = 0;
    lg->total_length = 0.0f;

    int cap = 0;
    for (int i = num_objects; i < num_entities; i++)
        if (entities[i].type == road_lane_type && entities[i].array_size >= 2)
            cap++;
    if (cap == 0)
        return;

    lg->lanes = (Lane *)calloc(cap, sizeof(Lane));

    for (int i = num_objects; i < num_entities; i++) {
        const Entity *e = &entities[i];
        if (e->type != road_lane_type || e->array_size < 2)
            continue;

        int n = e->array_size;
        float *cum = (float *)malloc(n * sizeof(float));
        cum[0] = 0.0f;
        for (int j = 1; j < n; j++) {
            float dx = e->traj_x[j] - e->traj_x[j - 1];
            float dy = e->traj_y[j] - e->traj_y[j - 1];
            cum[j] = cum[j - 1] + sqrtf(dx * dx + dy * dy);
        }
        if (cum[n - 1] < MIN_LANE_LENGTH) {
            free(cum);
            continue;
        }
        Lane *lane = &lg->lanes[lg->num_lanes++];
        lane->entity_idx = i;
        lane->num_points = n;
        lane->cum_s = cum;
        lane->length = cum[n - 1];
        lane->num_succ = 0;
    }

    if (lg->num_lanes == 0) {
        free(lg->lanes);
        lg->lanes = NULL;
        return;
    }

    // Successors: end of i coincides with start of j, and the tangents agree. O(L^2)
    // with L ~ 135 lanes per map, i.e. ~18k distance tests, paid once per map load.
    float eps2 = LANE_JOIN_EPS * LANE_JOIN_EPS;
    for (int i = 0; i < lg->num_lanes; i++) {
        Lane *a = &lg->lanes[i];
        const Entity *ea = &entities[a->entity_idx];
        int an = a->num_points;
        float ax = ea->traj_x[an - 1], ay = ea->traj_y[an - 1];
        float atx = ax - ea->traj_x[an - 2], aty = ay - ea->traj_y[an - 2];
        float alen = sqrtf(atx * atx + aty * aty);
        if (alen < 1e-6f)
            continue;
        atx /= alen;
        aty /= alen;

        for (int j = 0; j < lg->num_lanes && a->num_succ < MAX_LANE_SUCC; j++) {
            if (i == j)
                continue;
            Lane *b = &lg->lanes[j];
            const Entity *eb = &entities[b->entity_idx];
            float bx = eb->traj_x[0], by = eb->traj_y[0];
            float dx = bx - ax, dy = by - ay;
            if (dx * dx + dy * dy > eps2)
                continue;

            float btx = eb->traj_x[1] - bx, bty = eb->traj_y[1] - by;
            float blen = sqrtf(btx * btx + bty * bty);
            if (blen < 1e-6f)
                continue;
            btx /= blen;
            bty /= blen;
            if (atx * btx + aty * bty < LANE_SUCC_MIN_COS)
                continue; // oncoming lane, not a successor

            a->succ[a->num_succ++] = j;
        }
    }

    lg->cum_length = (float *)malloc((lg->num_lanes + 1) * sizeof(float));
    lg->cum_length[0] = 0.0f;
    for (int i = 0; i < lg->num_lanes; i++)
        lg->cum_length[i + 1] = lg->cum_length[i] + lg->lanes[i].length;
    lg->total_length = lg->cum_length[lg->num_lanes];
}

static void free_lane_graph(LaneGraph *lg) {
    if (lg->lanes != NULL) {
        for (int i = 0; i < lg->num_lanes; i++)
            free(lg->lanes[i].cum_s);
        free(lg->lanes);
        lg->lanes = NULL;
    }
    free(lg->cum_length);
    lg->cum_length = NULL;
    lg->num_lanes = 0;
    lg->total_length = 0.0f;
}

#endif // TEDDY_LANEGRAPH_H
