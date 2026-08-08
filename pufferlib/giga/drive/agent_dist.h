// Empirical agent state distribution, read from resources/drive/agent_dist.bin
// (built by data_utils/womd/build_agent_dist.py).
//
// Gigaflow draws size from independent uniforms and has no agent type. We resample
// whole (length, width, height) triples observed in WOMD instead, because the three
// extents are strongly correlated -- corr(l,w) = 0.86 for vehicles -- and sampling
// the marginals independently yields bodies that do not exist on the road. Since the
// camera policy is meant to be aligned against real Waymo imagery later, the
// rendered silhouettes have to come from the real distribution.
//
// The table is process global: it is a few hundred KB, identical for every scene,
// and loaded once.
#ifndef GIGA_AGENT_DIST_H
#define GIGA_AGENT_DIST_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "giga_random.h"

#define AGENT_DIST_MAX_TYPES 8

typedef struct {
    int type;
    float prob;
    float cum_prob;
    int num_rows;
    float *rows; // [num_rows * 3] = length, width, height
} AgentDistType;

typedef struct {
    AgentDistType types[AGENT_DIST_MAX_TYPES];
    int num_types;
    int loaded;
    char path[512];
} AgentDist;

static AgentDist g_agent_dist = {{{0}}, 0, 0, {0}};

// Returns 1 on success. Reloads only if a different path is requested, so the
// common case of ~34 envs sharing one table costs a single read.
static int agent_dist_load(const char *path) {
    if (g_agent_dist.loaded && strcmp(g_agent_dist.path, path) == 0)
        return 1;

    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        fprintf(stderr, "[giga] cannot open agent distribution '%s'\n", path);
        return 0;
    }

    char magic[8];
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "GIGADST1", 8) != 0) {
        fprintf(stderr, "[giga] '%s' is not a GIGADST1 file\n", path);
        fclose(f);
        return 0;
    }

    int num_types = 0;
    if (fread(&num_types, sizeof(int), 1, f) != 1 || num_types <= 0 || num_types > AGENT_DIST_MAX_TYPES) {
        fprintf(stderr, "[giga] '%s' has a bad type count (%d)\n", path, num_types);
        fclose(f);
        return 0;
    }

    // Free any previously loaded table before overwriting it.
    for (int i = 0; i < g_agent_dist.num_types; i++)
        free(g_agent_dist.types[i].rows);
    memset(&g_agent_dist, 0, sizeof(g_agent_dist));

    g_agent_dist.num_types = num_types;
    for (int i = 0; i < num_types; i++) {
        AgentDistType *t = &g_agent_dist.types[i];
        if (fread(&t->type, sizeof(int), 1, f) != 1 || fread(&t->prob, sizeof(float), 1, f) != 1 ||
            fread(&t->num_rows, sizeof(int), 1, f) != 1 || t->num_rows <= 0) {
            fprintf(stderr, "[giga] '%s' truncated in the type header\n", path);
            fclose(f);
            return 0;
        }
    }
    float acc = 0.0f;
    for (int i = 0; i < num_types; i++) {
        AgentDistType *t = &g_agent_dist.types[i];
        t->rows = (float *)malloc((size_t)t->num_rows * 3 * sizeof(float));
        if (fread(t->rows, sizeof(float), (size_t)t->num_rows * 3, f) != (size_t)t->num_rows * 3) {
            fprintf(stderr, "[giga] '%s' truncated in the row payload\n", path);
            fclose(f);
            return 0;
        }
        acc += t->prob;
        t->cum_prob = acc;
    }
    fclose(f);

    // Normalize so the final bucket is exactly 1.0; float accumulation of the stored
    // probabilities otherwise leaves a sliver that sampling would never reach.
    if (acc > 0.0f)
        for (int i = 0; i < num_types; i++)
            g_agent_dist.types[i].cum_prob /= acc;
    g_agent_dist.types[num_types - 1].cum_prob = 1.0f;

    snprintf(g_agent_dist.path, sizeof(g_agent_dist.path), "%s", path);
    g_agent_dist.loaded = 1;
    return 1;
}

// Draws a type from the categorical, then a whole row from that type's table, which
// is what preserves the joint distribution over the three extents.
static void agent_dist_sample(GigaRng *rng, int *out_type, float *out_length, float *out_width, float *out_height) {
    if (!g_agent_dist.loaded) {
        // Only reachable if a caller skipped agent_dist_load; keep the sim running
        // with a plausible sedan rather than a zero-size box.
        *out_type = 1;
        *out_length = 4.70f;
        *out_width = 2.09f;
        *out_height = 1.72f;
        return;
    }
    float u = giga_rand_float(rng);
    int ti = g_agent_dist.num_types - 1;
    for (int i = 0; i < g_agent_dist.num_types; i++) {
        if (u < g_agent_dist.types[i].cum_prob) {
            ti = i;
            break;
        }
    }
    AgentDistType *t = &g_agent_dist.types[ti];
    int row = giga_rand_int(rng, 0, t->num_rows - 1);
    *out_type = t->type;
    *out_length = t->rows[row * 3 + 0];
    *out_width = t->rows[row * 3 + 1];
    *out_height = t->rows[row * 3 + 2];
}

#endif // GIGA_AGENT_DIST_H
