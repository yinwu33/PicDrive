#include <signal.h>
#include <sys/types.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <unistd.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"
#include <time.h>
#include "error.h"
#include "teddy_random.h"

// Render modes
#define RENDER_WINDOW 0
#define RENDER_HEADLESS 1

// View modes
#define VIEW_MODE_SIM_STATE 0
#define VIEW_MODE_BEV_AGENT_OBS 1
#define VIEW_MODE_AGENT_PERSP 2

// Order of entities in rendering (lower is rendered first)
#define Z_ROAD_SURFACE 0.0f
#define Z_ROAD_MARKINGS 0.05f // Lane lines, road lines, traces
#define Z_AGENT_DETAILS 0.4f  // Arrow, goal markers, obs overlays
#define Z_AGENTS 0.6f         // Vehicles, cyclists, pedestrians

// Entity Types
#define NONE 0
#define VEHICLE 1
#define PEDESTRIAN 2
#define CYCLIST 3
#define ROAD_LANE 4
#define ROAD_LINE 5
#define ROAD_EDGE 6
#define STOP_SIGN 7
#define CROSSWALK 8
#define SPEED_BUMP 9
#define DRIVEWAY 10

#define INVALID_POSITION -10000.0f

// Trajectory Length
#define TRAJECTORY_LENGTH 91

// Initialization modes
#define INIT_ALL_VALID 0
#define INIT_ONLY_CONTROLLABLE_AGENTS 1

// Control modes
#define CONTROL_VEHICLES 0
#define CONTROL_AGENTS 1
#define CONTROL_WOSAC 2
#define CONTROL_SDC_ONLY 3
#define CONTROL_MIXED_PLAY 4

// Minimum distance to goal position
#define MIN_DISTANCE_TO_GOAL 2.0f

// Actions
#define NOOP 0

// Dynamics Models
#define CLASSIC 0
#define JERK 1

// Collision state
#define NO_COLLISION 0
#define VEHICLE_COLLISION 1
#define OFFROAD 2

// Metrics array indices
#define COLLISION_IDX 0
#define OFFROAD_IDX 1
#define REACHED_GOAL_IDX 2
#define LANE_ALIGNED_IDX 3

// Grid cell size
#define GRID_CELL_SIZE 5.0f
#define MAX_ENTITIES_PER_CELL                                                                                          \
    30 // Depends on resolution of data Formula: 3 * (2 + GRID_CELL_SIZE*sqrt(2)/resolution) => For each entity type in
       // gridmap, diagonal poly-lines -> sqrt(2), include diagonal ends -> 2

// Observation constants
#define MAX_ROAD_SEGMENT_OBSERVATIONS 128

// Default partner observation radius in metres, used when the config omits the key
// or gives a non-positive value. 50 m is what both envs hardcoded before it became
// configurable, so an old config keeps its behaviour exactly.
#define PARTNER_OBS_RADIUS_DEFAULT 50.0f


// Up to 3 intermediate waypoints (Gigaflow N_wp ~ U{0,3}) plus the final goal.
#define MAX_WAYPOINTS 4

// Maximum number of agents per scene
#ifndef MAX_AGENTS
#define MAX_AGENTS 128
#endif
// How many *other* agents reach the vector observation. Decoupled from MAX_AGENTS so
// the observation width does not follow the traffic density -- giga uses Gigaflow's
// N_o = 20 here as its partial-observability mechanism.
//
// 31 rather than 20 so that the observation is the same width as ocean's, which is
// MAX_AGENTS - 1 = 31 there. That is the whole reason for the number: with both envs
// on jerk dynamics, a policy trained on puffer_drive loads into puffer_teddy and back
// without adapting a single tensor. Change this and that stops being true.
//
// The camera env observes no partners at all, so this only affects puffer_teddy.
#ifndef MAX_PARTNER_OBS
#define MAX_PARTNER_OBS 31
#endif
#define STOP_AGENT 1
#define REMOVE_AGENT 2

#define ROAD_FEATURES 7
#define ROAD_FEATURES_ONEHOT 13
#define PARTNER_FEATURES 7

// Ego features depend on dynamics model. Plain ocean widths: `teddy` has no
// per-agent conditioning to append, because its reward weights are the same fixed
// config values for every agent.
#define EGO_FEATURES_CLASSIC 8
#define EGO_FEATURES_JERK 11

// Observation normalization constants
#define MAX_SPEED 100.0f
#define MAX_VEH_LEN 30.0f
#define MAX_VEH_WIDTH 15.0f
#define MAX_VEH_HEIGHT 10.0f
#define MIN_REL_GOAL_COORD -1000.0f
#define MAX_REL_GOAL_COORD 1000.0f
#define MIN_REL_AGENT_POS -1000.0f
#define MAX_REL_AGENT_POS 1000.0f
#define MAX_ORIENTATION_RAD 2 * PI
#define MIN_RG_COORD -1000.0f
#define MAX_RG_COORD 1000.0f
#define MAX_ROAD_SCALE 100.0f
#define MAX_ROAD_SEGMENT_LENGTH 100.0f

// Goal behavior
#define GOAL_RESPAWN 0
#define GOAL_GENERATE_NEW 1
#define GOAL_STOP 2
#define GOAL_REMOVE 3

// Observation mode. VECTOR is the legacy privileged entity-set observation.
// RENDER_STATE writes only the ego vector to the observation buffer and emits the
// scene into a separate world-frame RenderState buffer, which the perspective
// rasterizer consumes on the GPU. The policy never sees the scene buffer.
#define OBS_MODE_VECTOR 0
#define OBS_MODE_RENDER_STATE 1

// RenderState layout. All primitives are world-frame; the rasterizer transforms
// them per ego. Agents and egos are rewritten every step; roads are static for
// the lifetime of a map and are filled once at init.
#define RENDER_AGENT_FEATURES 8 // x, y, cos_h, sin_h, length, width, height, type
#define RENDER_ROAD_FEATURES 6  // x0, y0, x1, y1, width, type
// x, y, cos_h, sin_h, self_index. `self_index` points at this ego's own entry in
// the agent array so the rasterizer can skip it: a camera does not see the car it
// is mounted on, and the paper renders the surroundings only.
#define RENDER_EGO_FEATURES 5

// Painted-marking width in meters, used for road primitives that carry no width.
#define RENDER_ROAD_MARKING_WIDTH 0.15f

// Which road entity types are drawn into the perspective view, as a type bitmask.
// Lane centerlines (ROAD_LANE) are a map abstraction with no painted counterpart
// on real asphalt, so drawing them would put privileged structure into the image.
// They are off by default. STOP_SIGN is a point feature, not a polyline.
#define RENDER_ROAD_TYPES_DEFAULT ((1 << ROAD_LINE) | (1 << ROAD_EDGE) | (1 << CROSSWALK) | (1 << SPEED_BUMP))

// Jerk action space (for JERK dynamics model)
static const float JERK_LONG[4] = {-15.0f, -4.0f, 0.0f, 4.0f};
static const float JERK_LAT[3] = {-4.0f, 0.0f, 4.0f};

// Classic action space (for CLASSIC dynamics model)
static const float ACCELERATION_VALUES[7] = {-4.0000f, -2.6670f, -1.3330f, -0.0000f, 1.3330f, 2.6670f, 4.0000f};
static const float STEERING_VALUES[13] = {-1.000f, -0.833f, -0.667f, -0.500f, -0.333f, -0.167f, 0.000f,
                                          0.167f,  0.333f,  0.500f,  0.667f,  0.833f,  1.000f};

static const float offsets[4][2] = {
    {-1, 1}, // top-left
    {1, 1},  // top-right
    {1, -1}, // bottom-right
    {-1, -1} // bottom-left
};

static const int collision_offsets[25][2] = {
    {-2, -2}, {-1, -2}, {0, -2}, {1, -2}, {2, -2}, // Top row
    {-2, -1}, {-1, -1}, {0, -1}, {1, -1}, {2, -1}, // Second row
    {-2, 0},  {-1, 0},  {0, 0},  {1, 0},  {2, 0},  // Middle row (including center)
    {-2, 1},  {-1, 1},  {0, 1},  {1, 1},  {2, 1},  // Fourth row
    {-2, 2},  {-1, 2},  {0, 2},  {1, 2},  {2, 2}   // Bottom row
};

const Color STONE_GRAY = (Color){80, 80, 80, 255};
const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};
const Color PUFF_BACKGROUND2 = (Color){18, 72, 72, 255};
const Color LIGHTGREEN = (Color){152, 255, 152, 255};
const Color LIGHTYELLOW = (Color){255, 255, 152, 255};
const Color SOFT_YELLOW = (Color){245, 245, 220, 255};
const Color ROAD_COLOR = (Color){35, 35, 37, 255};
const Color LIGHTBLUE = (Color){167, 204, 255, 255};
const Color DEEPBLUE = (Color){45, 112, 226, 255};
const Color EXPERT_REPLAY = (Color){162, 220, 183, 255};
const Color EXPERT_REPLAY_SMALL = (Color){95, 112, 93, 255};
const Color LIGHT_ORANGE = (Color){255, 160, 80, 255};
const Color LIGHT_PURPLE = (Color){204, 204, 255, 255};

struct timespec ts;

typedef struct Drive Drive;
typedef struct Client Client;
typedef struct Log Log;

struct Log {
    float episode_return;
    float episode_length;
    float score;
    float goals_reached_this_episode;
    float goals_sampled_this_episode;
    float offroad_rate;
    float collision_rate;
    float completion_rate;
    float offroad_per_agent;
    float collisions_per_agent;
    float dnf_rate;
    float n;
    float lane_alignment_rate;
    float speed_at_goal;
    float active_agent_count;
    float expert_static_agent_count;
    float static_agent_count;
    float perc_controlled;
    float perc_other;
};

typedef struct Entity Entity;
struct Entity {
    int scenario_id;
    int type;
    int id;
    int array_size;
    float *traj_x;
    float *traj_y;
    float *traj_z;
    float *traj_vx;
    float *traj_vy;
    float *traj_vz;
    float *traj_heading;
    int *traj_valid;
    float width;
    float length;
    float height;
    float goal_position_x;
    float goal_position_y;
    float goal_position_z;
    float init_goal_x;
    float init_goal_y;
    int mark_as_expert;
    int collision_state;
    float metrics_array[5]; // metrics_array: [collision, offroad, reached_goal, lane_aligned
    float x;
    float y;
    float z;
    float vx;
    float vy;
    float vz;
    float heading;
    float heading_x;
    float heading_y;
    int current_lane_idx;
    int valid;
    // The step a recycled agent was re-placed on, or -1 if it never was. In `teddy`
    // this is a record, not a state: unlike `ocean`, nothing gates visibility or
    // collision on it, because a respawned agent here is a full road user from the
    // next step onward. Its only reader is the ego observation, which raises a flag
    // for exactly the one step on which the pose jumped.
    int respawn_timestep;
    int respawn_count;
    int collided_before_goal;
    float goals_reached_this_episode;
    float goals_sampled_this_episode;
    int current_goal_reached;
    int active_agent;
    int stopped;
    int removed;

    // Frenet state against the nearest lane centerline, filled by
    // compute_agent_metrics: signed heading error and signed lateral offset (positive
    // to the left of travel). No reward term reads these -- `teddy` pays only for
    // collisions, off-road and goals -- but the debug trace does, and "how far from
    // the lane centre was it when it hit something" is the first question a trace has
    // to answer. One atan2 per agent per step, against a neighbour sweep that already
    // costs far more.
    float lane_heading_error;
    float lane_lateral_offset;
    int lane_valid;

    // Gigaflow route: waypoints[0..num_waypoints-1] are visited in order, the last
    // one being the final goal. goal_position_x/y always mirrors the *current*
    // target so every existing consumer (observations, reward, renderer) keeps
    // working unchanged.
    float waypoints[MAX_WAYPOINTS][2];
    int num_waypoints;
    int current_waypoint;
    // Where this agent was spawned, kept so a respawn can start a fresh route from
    // the lane graph rather than from the dataset trajectory.
    int spawn_lane;
    float spawn_s;

    // Jerk dynamics
    float a_long;
    float a_lat;
    float jerk_long;
    float jerk_lat;
    float steering_angle;
    float wheelbase;
};

#include "teddy_random.h"
#include "lanegraph.h"
#include "agent_dist.h"

void free_entity(Entity *entity) {
    // free trajectory arrays
    free(entity->traj_x);
    free(entity->traj_y);
    free(entity->traj_z);
    free(entity->traj_vx);
    free(entity->traj_vy);
    free(entity->traj_vz);
    free(entity->traj_heading);
    free(entity->traj_valid);
}

// Utility functions
float relative_distance(float a, float b) {
    float distance = sqrtf(powf(a - b, 2));
    return distance;
}

float relative_distance_2d(float x1, float y1, float x2, float y2) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    float distance = sqrtf(dx * dx + dy * dy);
    return distance;
}

float clip(float value, float min, float max) {
    if (value < min)
        return min;
    if (value > max)
        return max;
    return value;
}

typedef struct GridMapEntity GridMapEntity;
struct GridMapEntity {
    int entity_idx;
    int geometry_idx;
};

typedef struct GridMap GridMap;
struct GridMap {
    float top_left_x;
    float top_left_y;
    float bottom_right_x;
    float bottom_right_y;
    int grid_cols;
    int grid_rows;
    int cell_size_x;
    int cell_size_y;
    int *cell_entities_count; // number of entities in each cell of the GridMap
    GridMapEntity **cells;    // list of gridEntities in each cell of the GridMap
    // Extras/Optimizations
    int vision_range;
    int *neighbor_cache_count;               // number of entities in each cells neighbor cache
    GridMapEntity **neighbor_cache_entities; // preallocated array to hold neighbor entities
};

// ---------------------------------------------------------------------------
// Per-step reward trace (debug only)
//
// The four reward terms are summed into one float before they reach `rewards[]`, so
// a run that drives badly says nothing about *which* term it is answering -- which
// is exactly the question when goal progress and collision rate rise together. When
// Python binds a `debug_terms` buffer through env_put, the reward loop writes one
// row per agent per step: every term separately, plus the state each term is
// computed from, so the sum can be checked against the reward the trainer actually
// saw. Unbound (the default, including in training) the whole mechanism costs one
// NULL test per step.
//
// The lane columns are state, not reward: nothing in `teddy` pays for lane
// discipline, but a trace that cannot say where in the lane the agent was is not
// worth reading.
// ---------------------------------------------------------------------------
#define TEDDY_DBG_REWARD 0 // total for this step; equals rewards[i]
#define TEDDY_DBG_R_COLLISION 1
#define TEDDY_DBG_R_OFFROAD 2
#define TEDDY_DBG_R_GOAL 3
#define TEDDY_DBG_R_JERK 4 // classic dynamics only; 0 under jerk
#define TEDDY_DBG_X 5
#define TEDDY_DBG_Y 6
#define TEDDY_DBG_HEADING 7
#define TEDDY_DBG_SPEED 8
#define TEDDY_DBG_SIGNED_V 9
#define TEDDY_DBG_A_LONG 10
#define TEDDY_DBG_A_LAT 11
#define TEDDY_DBG_JERK_LONG 12
#define TEDDY_DBG_JERK_LAT 13
#define TEDDY_DBG_STEERING 14
#define TEDDY_DBG_COLLISION_STATE 15 // 0 none, 1 vehicle, 2 offroad
#define TEDDY_DBG_LANE_VALID 16
#define TEDDY_DBG_LANE_HEADING_ERR 17
#define TEDDY_DBG_LANE_LATERAL_OFFSET 18
#define TEDDY_DBG_LANE_ALIGNED 19
#define TEDDY_DBG_DIST_TO_GOAL 20
#define TEDDY_DBG_CURRENT_WAYPOINT 21
#define TEDDY_DBG_NUM_WAYPOINTS 22
#define TEDDY_DBG_GOALS_REACHED 23 // cumulative over the agent's life
#define TEDDY_DBG_REACHED_FINAL 24 // final goal hit on this step
#define TEDDY_DBG_RESPAWN_COUNT 25
#define TEDDY_DBG_ACTION 26
#define TEDDY_DBG_TIMESTEP 27
#define TEDDY_DEBUG_FEATURES (TEDDY_DBG_TIMESTEP + 1)

struct Drive {
    Client *client;
    float *observations;
    float *actions;
    float *rewards;
    unsigned char *terminals;
    unsigned char *truncations;
    Log log;
    Log *logs;
    int num_agents;
    int active_agent_count;
    int *active_agent_indices;
    int action_type;
    int human_agent_idx;
    Entity *entities;
    int num_entities;
    int num_actors;
    int num_objects;
    int num_roads;
    int static_agent_count;
    int *static_agent_indices;
    int expert_static_agent_count;
    int *expert_static_agent_indices;
    int timestep;
    int init_steps;
    int dynamics_model;
    GridMap *grid_map;
    int *neighbor_offsets;
    int episode_length;
    int termination_mode;
    float reward_vehicle_collision;
    float reward_offroad_collision;
    char *map_name;
    float world_mean_x;
    float world_mean_y;
    float dt;
    float reward_goal;
    float reward_goal_post_respawn;
    float goal_radius;
    float goal_speed;
    int logs_capacity;
    int goal_behavior;
    float goal_target_distance;
    char *ini_file;
    char scenario_id[16];
    int collision_behavior;
    int offroad_behavior;
    int sdc_track_index;
    int num_tracks_to_predict;
    int *tracks_to_predict_indices;
    int init_mode;
    int control_mode;
    int max_controlled_agents;
    int render_mode;

    // How far a partner can be and still reach the vector observation, in metres.
    // Was hardcoded at 50 m. Configurable because it is the single knob that decides
    // how much of the scene the policy is allowed to see, and because the answer
    // differs by sensor: a camera policy cannot see 50 m of cross traffic through a
    // building, so a vector baseline that does is not the same task. 0 or less in the
    // config means "use the 50 m default" -- see PARTNER_OBS_RADIUS_DEFAULT.
    float partner_obs_radius;

    // Gigaflow-style random initialization. Unlike ocean's env this is not optional:
    // agents are always placed by sampling the lane graph, never from logged tracks.
    LaneGraph lane_graph;
    TeddyRng rng;
    int agents_per_map_min;
    int agents_per_map_max;
    float spawn_speed_max;
    float spawn_heading_jitter_deg;
    float wrong_way_frac;
    int num_waypoints_max;
    float waypoint_min_dist;
    float waypoint_max_dist;

    // Perspective rendering (Pictura). Buffers are owned by Python and handed in
    // through the binding, so the rasterizer reads them without a copy.
    int obs_mode;
    int render_road_types;  // bitmask over entity types; see render_type_enabled()
    float *render_agents;   // [render_max_agents * RENDER_AGENT_FEATURES]
    float *render_egos;     // [active_agent_count * RENDER_EGO_FEATURES]
    float *render_roads;    // [render_max_roads * RENDER_ROAD_FEATURES]
    int *render_counts;     // [2] = {agents written this step, roads written at init}
    int render_max_agents;
    int render_max_roads;

    // Rendered camera views for the selected agent, so the viewer can show what
    // the policy actually sees. Laid out as [num_cameras * height, width, 3],
    // i.e. the views stacked vertically, in the order they should be displayed
    // left to right. Python sorts the rig by mounting yaw before filling this, so
    // a three-camera Waymo rig arrives as front_left, front, front_right. Owned by
    // Python and filled from the rasterizer; NULL when perspective rendering is off.
    unsigned char *render_camera_rgb;
    int render_camera_count;
    int render_camera_width;
    int render_camera_height;
    // Camera names for the panel labels, NUL-padded to a fixed stride and in the
    // same order as the views above. NULL when Python did not supply them, in
    // which case the panels fall back to an index.
    char *render_camera_names;
    int render_camera_name_stride;

    // Per-step reward trace. Owned by Python, NULL unless env_put was handed a
    // `debug_terms` array; see the TEDDY_DBG_* block above for the row layout.
    float *debug_terms;
    int debug_max_rows;
};

// Single source of truth for the observation stride. In RENDER_STATE mode the
// observation carries only the ego vector; the scene leaves through RenderState.
static inline int drive_ego_dim(Drive *env) {
    return (env->dynamics_model == JERK) ? EGO_FEATURES_JERK : EGO_FEATURES_CLASSIC;
}

static inline int drive_obs_size(Drive *env) {
    int ego_dim = drive_ego_dim(env);
    if (env->obs_mode == OBS_MODE_RENDER_STATE)
        return ego_dim;
    return ego_dim + PARTNER_FEATURES * MAX_PARTNER_OBS + ROAD_FEATURES * MAX_ROAD_SEGMENT_OBSERVATIONS;
}

void add_log(Drive *env) {
    for (int i = 0; i < env->active_agent_count; i++) {
        Entity *e = &env->entities[env->active_agent_indices[i]];

        env->log.goals_reached_this_episode += e->goals_reached_this_episode;
        env->log.goals_sampled_this_episode += e->goals_sampled_this_episode;

        int offroad = env->logs[i].offroad_rate;
        env->log.offroad_rate += offroad;
        int collided = env->logs[i].collision_rate;
        env->log.collision_rate += collided;
        float offroad_per_agent = env->logs[i].offroad_per_agent;
        env->log.offroad_per_agent += offroad_per_agent;
        float collisions_per_agent = env->logs[i].collisions_per_agent;
        env->log.collisions_per_agent += collisions_per_agent;

        float frac_goal_reached = e->goals_reached_this_episode / e->goals_sampled_this_episode;

        // Update score, which is an aggregate measure whether the agent fully solved its task
        float threshold = 1.0f; // Default threshold for 1 goal (must complete it)
        if (e->goals_sampled_this_episode > 1) {
            // For multiple goals, require n-1 goals to be reached
            threshold = (e->goals_sampled_this_episode - 1.0f) / e->goals_sampled_this_episode;
        }

        int collision_occurred =
            (env->goal_behavior == GOAL_RESPAWN) ? e->collided_before_goal : env->logs[i].collision_rate;
        if (frac_goal_reached >= threshold && !collision_occurred) {
            env->log.score += 1.0f;
        }
        if (!offroad && !collided && frac_goal_reached < 1.0f) {
            env->log.dnf_rate += 1.0f;
        }
        // Time average over the agent's own step count, which is the episode length
        // for every agent here (respawn recycles an agent, it does not end it).
        float steps = env->logs[i].episode_length;
        if (steps > 0.0f)
            env->log.lane_alignment_rate += env->logs[i].lane_alignment_rate / steps;
        env->log.speed_at_goal += env->logs[i].speed_at_goal;
        env->log.episode_length += env->logs[i].episode_length;
        env->log.episode_return += env->logs[i].episode_return;
        // Log composition counts per agent so vec_log averaging recovers the per-env value
        env->log.active_agent_count += env->active_agent_count;
        env->log.expert_static_agent_count += env->expert_static_agent_count;
        env->log.static_agent_count += env->static_agent_count;
        int total = env->active_agent_count + env->static_agent_count;
        env->log.perc_controlled += (float)env->active_agent_count / (float)total;
        env->log.perc_other += (float)env->static_agent_count / (float)total;
        env->log.n += 1;
    }
}

Entity *load_map_binary(const char *filename, Drive *env) {
    FILE *file = fopen(filename, "rb");
    if (!file)
        return NULL;

    // Read scenario_id
    fread(env->scenario_id, sizeof(char), 16, file);

    // Read sdc_track_index
    fread(&env->sdc_track_index, sizeof(int), 1, file);

    // Read tracks_to_predict
    fread(&env->num_tracks_to_predict, sizeof(int), 1, file);
    if (env->num_tracks_to_predict > 0) {
        env->tracks_to_predict_indices = (int *)malloc(env->num_tracks_to_predict * sizeof(int));

        for (int i = 0; i < env->num_tracks_to_predict; i++) {
            fread(&env->tracks_to_predict_indices[i], sizeof(int), 1, file);
        }
    } else {
        env->tracks_to_predict_indices = NULL;
    }

    fread(&env->num_objects, sizeof(int), 1, file);
    fread(&env->num_roads, sizeof(int), 1, file);
    env->num_entities = env->num_objects + env->num_roads;
    Entity *entities = (Entity *)malloc(env->num_entities * sizeof(Entity));
    for (int i = 0; i < env->num_entities; i++) {
        // Read base entity data
        fread(&entities[i].scenario_id, sizeof(int), 1, file);
        fread(&entities[i].type, sizeof(int), 1, file);
        fread(&entities[i].id, sizeof(int), 1, file);
        fread(&entities[i].array_size, sizeof(int), 1, file);
        // Allocate arrays based on type
        int size = entities[i].array_size;
        entities[i].traj_x = (float *)malloc(size * sizeof(float));
        entities[i].traj_y = (float *)malloc(size * sizeof(float));
        entities[i].traj_z = (float *)malloc(size * sizeof(float));
        if (entities[i].type == VEHICLE || entities[i].type == PEDESTRIAN ||
            entities[i].type == CYCLIST) { // Object type
            // Allocate arrays for object-specific data
            entities[i].traj_vx = (float *)malloc(size * sizeof(float));
            entities[i].traj_vy = (float *)malloc(size * sizeof(float));
            entities[i].traj_vz = (float *)malloc(size * sizeof(float));
            entities[i].traj_heading = (float *)malloc(size * sizeof(float));
            entities[i].traj_valid = (int *)malloc(size * sizeof(int));
        } else {
            // Roads don't use these arrays
            entities[i].traj_vx = NULL;
            entities[i].traj_vy = NULL;
            entities[i].traj_vz = NULL;
            entities[i].traj_heading = NULL;
            entities[i].traj_valid = NULL;
        }
        // Read array data
        fread(entities[i].traj_x, sizeof(float), size, file);
        fread(entities[i].traj_y, sizeof(float), size, file);
        fread(entities[i].traj_z, sizeof(float), size, file);
        if (entities[i].type == VEHICLE || entities[i].type == PEDESTRIAN ||
            entities[i].type == CYCLIST) { // Object type
            fread(entities[i].traj_vx, sizeof(float), size, file);
            fread(entities[i].traj_vy, sizeof(float), size, file);
            fread(entities[i].traj_vz, sizeof(float), size, file);
            fread(entities[i].traj_heading, sizeof(float), size, file);
            fread(entities[i].traj_valid, sizeof(int), size, file);
        }
        // Read remaining scalar fields
        fread(&entities[i].width, sizeof(float), 1, file);
        fread(&entities[i].length, sizeof(float), 1, file);
        fread(&entities[i].height, sizeof(float), 1, file);
        fread(&entities[i].goal_position_x, sizeof(float), 1, file);
        fread(&entities[i].goal_position_y, sizeof(float), 1, file);
        fread(&entities[i].goal_position_z, sizeof(float), 1, file);
        fread(&entities[i].mark_as_expert, sizeof(int), 1, file);
    }

    fclose(file);
    return entities;
}

void set_start_position(Drive *env) {
    for (int i = 0; i < env->num_entities; i++) {
        int is_active = 0;
        for (int j = 0; j < env->active_agent_count; j++) {
            if (env->active_agent_indices[j] == i) {
                is_active = 1;
                break;
            }
        }
        Entity *e = &env->entities[i];

        // Clamp init_steps to ensure we don't go out of bounds
        int step = env->init_steps;
        if (step >= e->array_size)
            step = e->array_size - 1;
        if (step < 0)
            step = 0;

        e->x = e->traj_x[step];
        e->y = e->traj_y[step];
        e->z = e->traj_z[step];
        if (e->type > CYCLIST || e->type == 0) {
            continue;
        }
        if (is_active == 0) {
            e->vx = 0;
            e->vy = 0;
            e->vz = 0;
            e->collided_before_goal = 0;
        } else {
            e->vx = e->traj_vx[env->init_steps];
            e->vy = e->traj_vy[env->init_steps];
            e->vz = e->traj_vz[env->init_steps];
        }
        e->heading = e->traj_heading[env->init_steps];
        e->heading_x = cosf(e->heading);
        e->heading_y = sinf(e->heading);
        e->valid = e->traj_valid[env->init_steps];
        e->collision_state = 0;
        e->metrics_array[COLLISION_IDX] = 0.0f;    // vehicle collision
        e->metrics_array[OFFROAD_IDX] = 0.0f;      // offroad
        e->metrics_array[REACHED_GOAL_IDX] = 0.0f; // reached goal
        e->metrics_array[LANE_ALIGNED_IDX] = 0.0f; // lane aligned
        e->respawn_timestep = -1;
        e->stopped = 0;
        e->removed = 0;
        e->respawn_count = 0;

        // Dynamics
        e->a_long = 0.0f;
        e->a_lat = 0.0f;
        e->jerk_long = 0.0f;
        e->jerk_lat = 0.0f;
        e->steering_angle = 0.0f;
        e->wheelbase = 0.6f * e->length;
    }
}

int getGridIndex(Drive *env, float x1, float y1) {
    if (env->grid_map->top_left_x >= env->grid_map->bottom_right_x ||
        env->grid_map->bottom_right_y >= env->grid_map->top_left_y) {
        return -1; // Invalid grid coordinates
    }

    float relativeX = x1 - env->grid_map->top_left_x;     // Distance from left
    float relativeY = y1 - env->grid_map->bottom_right_y; // Distance from bottom
    int gridX = (int)(relativeX / GRID_CELL_SIZE);        // Column index
    int gridY = (int)(relativeY / GRID_CELL_SIZE);        // Row index
    if (gridX < 0 || gridX >= env->grid_map->grid_cols || gridY < 0 || gridY >= env->grid_map->grid_rows) {
        return -1; // Return -1 for out of bounds
    }
    int index = (gridY * env->grid_map->grid_cols) + gridX;
    return index;
}

void add_entity_to_grid(Drive *env, int grid_index, int entity_idx, int geometry_idx, int *cell_entities_insert_index) {
    if (grid_index == -1) {
        return;
    }

    int count = cell_entities_insert_index[grid_index];
    if (count >= env->grid_map->cell_entities_count[grid_index]) {
        printf("Error: Exceeded precomputed entity count for grid cell %d. Current count: %d, Max count(Precomputed): "
               "%d\n",
               grid_index, count, env->grid_map->cell_entities_count[grid_index]);
        return;
    }

    env->grid_map->cells[grid_index][count].entity_idx = entity_idx;
    env->grid_map->cells[grid_index][count].geometry_idx = geometry_idx;
    cell_entities_insert_index[grid_index] = count + 1;
}

void init_grid_map(Drive *env) {
    // Allocate memory for the grid map structure
    env->grid_map = (GridMap *)malloc(sizeof(GridMap));

    // Find top left and bottom right points of the map
    float top_left_x;
    float top_left_y;
    float bottom_right_x;
    float bottom_right_y;
    int first_valid_point = 0;
    for (int i = 0; i < env->num_entities; i++) {
        if (env->entities[i].type > 3 && env->entities[i].type < 7) {
            // Check all points in the trajectory for road elements
            Entity *e = &env->entities[i];
            for (int j = 0; j < e->array_size; j++) {
                if (e->traj_x[j] == INVALID_POSITION)
                    continue;
                if (e->traj_y[j] == INVALID_POSITION)
                    continue;
                if (!first_valid_point) {
                    top_left_x = bottom_right_x = e->traj_x[j];
                    top_left_y = bottom_right_y = e->traj_y[j];
                    first_valid_point = true;
                    continue;
                }
                if (e->traj_x[j] < top_left_x)
                    top_left_x = e->traj_x[j];
                if (e->traj_x[j] > bottom_right_x)
                    bottom_right_x = e->traj_x[j];
                if (e->traj_y[j] > top_left_y)
                    top_left_y = e->traj_y[j];
                if (e->traj_y[j] < bottom_right_y)
                    bottom_right_y = e->traj_y[j];
            }
        }
    }

    env->grid_map->top_left_x = top_left_x;
    env->grid_map->top_left_y = top_left_y;
    env->grid_map->bottom_right_x = bottom_right_x;
    env->grid_map->bottom_right_y = bottom_right_y;
    env->grid_map->cell_size_x = GRID_CELL_SIZE;
    env->grid_map->cell_size_y = GRID_CELL_SIZE;

    // Calculate grid dimensions
    float grid_width = bottom_right_x - top_left_x;
    float grid_height = top_left_y - bottom_right_y;
    env->grid_map->grid_cols = ceil(grid_width / GRID_CELL_SIZE);
    env->grid_map->grid_rows = ceil(grid_height / GRID_CELL_SIZE);
    int grid_cell_count = env->grid_map->grid_cols * env->grid_map->grid_rows;
    env->grid_map->cells = (GridMapEntity **)calloc(grid_cell_count, sizeof(GridMapEntity *));
    env->grid_map->cell_entities_count = (int *)calloc(grid_cell_count, sizeof(int));

    // Calculate number of entities in each grid cell
    for (int i = 0; i < env->num_entities; i++) {
        if (env->entities[i].type > 3 && env->entities[i].type < 7) {
            for (int j = 0; j < env->entities[i].array_size - 1; j++) {
                float x_center = (env->entities[i].traj_x[j] + env->entities[i].traj_x[j + 1]) / 2;
                float y_center = (env->entities[i].traj_y[j] + env->entities[i].traj_y[j + 1]) / 2;
                int grid_index = getGridIndex(env, x_center, y_center);
                env->grid_map->cell_entities_count[grid_index]++;
            }
        }
    }
    int cell_entities_insert_index[grid_cell_count]; // Helper array for insertion index
    memset(cell_entities_insert_index, 0, grid_cell_count * sizeof(int));

    // Initialize grid cells
    for (int grid_index = 0; grid_index < grid_cell_count; grid_index++) {
        env->grid_map->cells[grid_index] =
            (GridMapEntity *)calloc(env->grid_map->cell_entities_count[grid_index], sizeof(GridMapEntity));
    }
    for (int i = 0; i < grid_cell_count; i++) {
        if (cell_entities_insert_index[i] != 0) {
            printf("Error: cell_entities_insert_index[%d] not zero during initialization.\n", i);
            cell_entities_insert_index[i] = 0;
        }
    }

    // Populate grid cells
    for (int i = 0; i < env->num_entities; i++) {
        if (env->entities[i].type > 3 &&
            env->entities[i].type < 7) { // NOTE: Only Road Edges, Lines, and Lanes in grid map
            for (int j = 0; j < env->entities[i].array_size - 1; j++) {
                float x_center = (env->entities[i].traj_x[j] + env->entities[i].traj_x[j + 1]) / 2;
                float y_center = (env->entities[i].traj_y[j] + env->entities[i].traj_y[j + 1]) / 2;
                int grid_index = getGridIndex(env, x_center, y_center);
                add_entity_to_grid(env, grid_index, i, j, cell_entities_insert_index);
            }
        }
    }
}

void init_neighbor_offsets(Drive *env) {
    // Allocate memory for the offsets
    env->neighbor_offsets = (int *)calloc(env->grid_map->vision_range * env->grid_map->vision_range * 2, sizeof(int));
    // neighbor offsets in a spiral pattern
    int dx[] = {1, 0, -1, 0};
    int dy[] = {0, 1, 0, -1};
    int x = 0;                  // Current x offset
    int y = 0;                  // Current y offset
    int dir = 0;                // Current direction (0: right, 1: up, 2: left, 3: down)
    int steps_to_take = 1;      // Number of steps in current direction
    int steps_taken = 0;        // Steps taken in current direction
    int segments_completed = 0; // Count of direction segments completed
    int total = 0;              // Total offsets added
    int max_offsets = env->grid_map->vision_range * env->grid_map->vision_range;
    // Start at center (0,0)
    int curr_idx = 0;
    env->neighbor_offsets[curr_idx++] = 0; // x offset
    env->neighbor_offsets[curr_idx++] = 0; // y offset
    total++;
    // Generate spiral pattern
    while (total < max_offsets) {
        // Move in current direction
        x += dx[dir];
        y += dy[dir];
        // Only add if within vision range bounds
        if (abs(x) <= env->grid_map->vision_range / 2 && abs(y) <= env->grid_map->vision_range / 2) {
            env->neighbor_offsets[curr_idx++] = x;
            env->neighbor_offsets[curr_idx++] = y;
            total++;
        }
        steps_taken++;
        // Check if we need to change direction
        if (steps_taken != steps_to_take)
            continue;
        steps_taken = 0;     // Reset steps taken
        dir = (dir + 1) % 4; // Change direction (clockwise: right->up->left->down)
        segments_completed++;
        // Increase step length every two direction changes
        if (segments_completed % 2 == 0) {
            steps_to_take++;
        }
    }
}

void cache_neighbor_offsets(Drive *env) {
    int count = 0;
    int cell_count = env->grid_map->grid_cols * env->grid_map->grid_rows;
    env->grid_map->neighbor_cache_entities = (GridMapEntity **)calloc(cell_count, sizeof(GridMapEntity *));
    env->grid_map->neighbor_cache_count = (int *)calloc(cell_count + 1, sizeof(int));
    for (int i = 0; i < cell_count; i++) {
        int cell_x = i % env->grid_map->grid_cols; // Convert to 2D coordinates
        int cell_y = i / env->grid_map->grid_cols;
        int current_cell_neighbor_count = 0;
        for (int j = 0; j < env->grid_map->vision_range * env->grid_map->vision_range; j++) {
            int x = cell_x + env->neighbor_offsets[j * 2];
            int y = cell_y + env->neighbor_offsets[j * 2 + 1];
            int grid_index = env->grid_map->grid_cols * y + x;
            if (x < 0 || x >= env->grid_map->grid_cols || y < 0 || y >= env->grid_map->grid_rows)
                continue;
            int grid_count = env->grid_map->cell_entities_count[grid_index];
            current_cell_neighbor_count += grid_count;
        }
        env->grid_map->neighbor_cache_count[i] = current_cell_neighbor_count;
        count += current_cell_neighbor_count;
        if (current_cell_neighbor_count == 0) {
            env->grid_map->neighbor_cache_entities[i] = NULL;
            continue;
        }
        env->grid_map->neighbor_cache_entities[i] =
            (GridMapEntity *)calloc(current_cell_neighbor_count, sizeof(GridMapEntity));
    }

    env->grid_map->neighbor_cache_count[cell_count] = count;
    for (int i = 0; i < cell_count; i++) {
        int cell_x = i % env->grid_map->grid_cols; // Convert to 2D coordinates
        int cell_y = i / env->grid_map->grid_cols;
        int base_index = 0;
        for (int j = 0; j < env->grid_map->vision_range * env->grid_map->vision_range; j++) {
            int x = cell_x + env->neighbor_offsets[j * 2];
            int y = cell_y + env->neighbor_offsets[j * 2 + 1];
            int grid_index = env->grid_map->grid_cols * y + x;
            if (x < 0 || x >= env->grid_map->grid_cols || y < 0 || y >= env->grid_map->grid_rows)
                continue;
            int grid_count = env->grid_map->cell_entities_count[grid_index];

            // Skip if no entities or source is NULL
            if (grid_count == 0 || env->grid_map->cells[grid_index] == NULL) {
                continue;
            }

            int src_idx = grid_index;
            int dst_idx = base_index;
            // Copy grid_count pairs (entity_idx, geometry_idx) at once
            memcpy(&env->grid_map->neighbor_cache_entities[i][dst_idx], env->grid_map->cells[src_idx],
                   grid_count * sizeof(GridMapEntity));
            base_index += grid_count;
        }
    }
}

int get_neighbor_cache_entities(Drive *env, int cell_idx, GridMapEntity *entities, int max_entities) {
    GridMap *grid_map = env->grid_map;
    if (cell_idx < 0 || cell_idx >= (grid_map->grid_cols * grid_map->grid_rows)) {
        return 0; // Invalid cell index
    }

    int count = grid_map->neighbor_cache_count[cell_idx];
    // Limit to available space
    if (count > max_entities) {
        count = max_entities;
    }
    memcpy(entities, grid_map->neighbor_cache_entities[cell_idx], count * sizeof(GridMapEntity));
    return count;
}

void set_means(Drive *env) {
    float mean_x = 0.0f;
    float mean_y = 0.0f;
    int64_t point_count = 0;

    // Compute single mean for all entities (vehicles and roads)
    for (int i = 0; i < env->num_entities; i++) {
        if (env->entities[i].type == VEHICLE || env->entities[i].type == PEDESTRIAN ||
            env->entities[i].type == CYCLIST) {
            for (int j = 0; j < env->entities[i].array_size; j++) {
                // Assume a validity flag exists (e.g., valid[j]); adjust if not available
                if (env->entities[i].traj_valid[j]) { // Add validity check if applicable
                    point_count++;
                    mean_x += (env->entities[i].traj_x[j] - mean_x) / point_count;
                    mean_y += (env->entities[i].traj_y[j] - mean_y) / point_count;
                }
            }
        } else if (env->entities[i].type >= 4) {
            for (int j = 0; j < env->entities[i].array_size; j++) {
                point_count++;
                mean_x += (env->entities[i].traj_x[j] - mean_x) / point_count;
                mean_y += (env->entities[i].traj_y[j] - mean_y) / point_count;
            }
        }
    }
    env->world_mean_x = mean_x;
    env->world_mean_y = mean_y;
    for (int i = 0; i < env->num_entities; i++) {
        if (env->entities[i].type == VEHICLE || env->entities[i].type == PEDESTRIAN ||
            env->entities[i].type == CYCLIST || env->entities[i].type >= 4) {
            for (int j = 0; j < env->entities[i].array_size; j++) {
                if (env->entities[i].traj_x[j] == INVALID_POSITION)
                    continue;
                env->entities[i].traj_x[j] -= mean_x;
                env->entities[i].traj_y[j] -= mean_y;
            }
            env->entities[i].goal_position_x -= mean_x;
            env->entities[i].goal_position_y -= mean_y;
        }
    }
}

void move_expert(Drive *env, float *actions, int agent_idx) {
    Entity *agent = &env->entities[agent_idx];
    int t = env->timestep;
    if (t < 0 || t >= agent->array_size) {
        agent->x = INVALID_POSITION;
        agent->y = INVALID_POSITION;
        agent->z = 0.0f;
        agent->heading = 0.0f;
        agent->heading_x = 1.0f;
        agent->heading_y = 0.0f;
        return;
    }
    if (agent->traj_valid && agent->traj_valid[t] == 0) {
        agent->x = INVALID_POSITION;
        agent->y = INVALID_POSITION;
        agent->z = 0.0f;
        agent->heading = 0.0f;
        agent->heading_x = 1.0f;
        agent->heading_y = 0.0f;
        return;
    }
    agent->x = agent->traj_x[t];
    agent->y = agent->traj_y[t];
    agent->z = agent->traj_z[t];
    agent->heading = agent->traj_heading[t];
    agent->heading_x = cosf(agent->heading);
    agent->heading_y = sinf(agent->heading);
}

bool check_line_intersection(float p1[2], float p2[2], float q1[2], float q2[2]) {
    if (fmax(p1[0], p2[0]) < fmin(q1[0], q2[0]) || fmin(p1[0], p2[0]) > fmax(q1[0], q2[0]) ||
        fmax(p1[1], p2[1]) < fmin(q1[1], q2[1]) || fmin(p1[1], p2[1]) > fmax(q1[1], q2[1]))
        return false;

    // Calculate vectors
    float dx1 = p2[0] - p1[0];
    float dy1 = p2[1] - p1[1];
    float dx2 = q2[0] - q1[0];
    float dy2 = q2[1] - q1[1];

    // Calculate cross products
    float cross = dx1 * dy2 - dy1 * dx2;

    // If lines are parallel
    if (cross == 0)
        return false;

    // Calculate relative vectors between start points
    float dx3 = p1[0] - q1[0];
    float dy3 = p1[1] - q1[1];

    // Calculate parameters for intersection point
    float s = (dx1 * dy3 - dy1 * dx3) / cross;
    float t = (dx2 * dy3 - dy2 * dx3) / cross;

    // Check if intersection point lies within both line segments
    return (s >= 0 && s <= 1 && t >= 0 && t <= 1);
}

int checkNeighbors(Drive *env, float x, float y, GridMapEntity *entity_list, int max_size,
                   const int (*local_offsets)[2], int offset_size) {
    // Get the grid index for the given position (x, y)
    int index = getGridIndex(env, x, y);
    if (index == -1)
        return 0; // Return 0 size if position invalid
    // Calculate 2D grid coordinates
    int cellsX = env->grid_map->grid_cols;
    int gridX = index % cellsX;
    int gridY = index / cellsX;
    int entity_list_count = 0;
    // Fill the provided array
    for (int i = 0; i < offset_size; i++) {
        int nx = gridX + local_offsets[i][0];
        int ny = gridY + local_offsets[i][1];
        // Ensure the neighbor is within grid bounds
        if (nx < 0 || nx >= env->grid_map->grid_cols || ny < 0 || ny >= env->grid_map->grid_rows)
            continue;
        int neighborIndex = ny * env->grid_map->grid_cols + nx;
        int count = env->grid_map->cell_entities_count[neighborIndex];
        // Add entities from this cell to the list
        for (int j = 0; j < count && entity_list_count < max_size; j++) {
            int entityId = env->grid_map->cells[neighborIndex][j].entity_idx;
            int geometry_idx = env->grid_map->cells[neighborIndex][j].geometry_idx;
            entity_list[entity_list_count].entity_idx = entityId;
            entity_list[entity_list_count].geometry_idx = geometry_idx;
            entity_list_count += 1;
        }
    }
    return entity_list_count;
}

int check_aabb_collision(Entity *car1, Entity *car2) {
    // Get car corners in world space
    float cos1 = car1->heading_x;
    float sin1 = car1->heading_y;
    float cos2 = car2->heading_x;
    float sin2 = car2->heading_y;

    // Calculate half dimensions
    float half_len1 = car1->length * 0.5f;
    float half_width1 = car1->width * 0.5f;
    float half_len2 = car2->length * 0.5f;
    float half_width2 = car2->width * 0.5f;

    // Calculate car1's corners in world space
    float car1_corners[4][2] = {
        {car1->x + (half_len1 * cos1 - half_width1 * sin1), car1->y + (half_len1 * sin1 + half_width1 * cos1)},
        {car1->x + (half_len1 * cos1 + half_width1 * sin1), car1->y + (half_len1 * sin1 - half_width1 * cos1)},
        {car1->x + (-half_len1 * cos1 - half_width1 * sin1), car1->y + (-half_len1 * sin1 + half_width1 * cos1)},
        {car1->x + (-half_len1 * cos1 + half_width1 * sin1), car1->y + (-half_len1 * sin1 - half_width1 * cos1)}};

    // Calculate car2's corners in world space
    float car2_corners[4][2] = {
        {car2->x + (half_len2 * cos2 - half_width2 * sin2), car2->y + (half_len2 * sin2 + half_width2 * cos2)},
        {car2->x + (half_len2 * cos2 + half_width2 * sin2), car2->y + (half_len2 * sin2 - half_width2 * cos2)},
        {car2->x + (-half_len2 * cos2 - half_width2 * sin2), car2->y + (-half_len2 * sin2 + half_width2 * cos2)},
        {car2->x + (-half_len2 * cos2 + half_width2 * sin2), car2->y + (-half_len2 * sin2 - half_width2 * cos2)}};

    // Get the axes to check (normalized vectors perpendicular to each edge)
    float axes[4][2] = {
        {cos1, sin1},  // Car1's length axis
        {-sin1, cos1}, // Car1's width axis
        {cos2, sin2},  // Car2's length axis
        {-sin2, cos2}  // Car2's width axis
    };

    // Check each axis
    for (int i = 0; i < 4; i++) {
        float min1 = INFINITY, max1 = -INFINITY;
        float min2 = INFINITY, max2 = -INFINITY;

        // Project car1's corners onto the axis
        for (int j = 0; j < 4; j++) {
            float proj = car1_corners[j][0] * axes[i][0] + car1_corners[j][1] * axes[i][1];
            min1 = fminf(min1, proj);
            max1 = fmaxf(max1, proj);
        }

        // Project car2's corners onto the axis
        for (int j = 0; j < 4; j++) {
            float proj = car2_corners[j][0] * axes[i][0] + car2_corners[j][1] * axes[i][1];
            min2 = fminf(min2, proj);
            max2 = fmaxf(max2, proj);
        }

        // If there's a gap on this axis, the boxes don't intersect
        if (max1 < min2 || min1 > max2) {
            return 0; // No collision
        }
    }

    // If we get here, there's no separating axis, so the boxes intersect
    return 1; // Collision
}

int collision_check(Drive *env, int agent_idx) {
    Entity *agent = &env->entities[agent_idx];

    if (agent->x == INVALID_POSITION)
        return -1;

    // ocean skips pedestrians here because dataset-driven init drops them onto
    // sidewalks already overlapping other entities. Gigaflow init has no such
    // pedestrians: every controlled agent is rejection-sampled onto the lane graph,
    // routed over it and driven with the same bicycle model, so the type only
    // selects a silhouette. Exempting it produced ~15% of traffic that could drive
    // through cars and off the road for free while still collecting the lane and
    // goal rewards -- and, because pedestrians stayed valid collision *targets*,
    // charging the vehicles they hit for it.

    int car_collided_with_index = -1;

    for (int i = 0; i < env->num_actors; i++) {
        int index = -1;
        if (i < env->active_agent_count) {
            index = env->active_agent_indices[i];
        } else if (i < env->num_actors && env->static_agent_count > 0) {
            index = env->static_agent_indices[i - env->active_agent_count];
        }
        if (index == -1)
            continue;
        if (index == agent_idx)
            continue;
        Entity *entity = &env->entities[index];
        float x1 = entity->x;
        float y1 = entity->y;
        float dist = ((x1 - agent->x) * (x1 - agent->x) + (y1 - agent->y) * (y1 - agent->y));
        if (dist > 225.0f)
            continue;
        if (check_aabb_collision(agent, entity)) {
            car_collided_with_index = index;
            break;
        }
    }

    return car_collided_with_index;
}

int check_lane_aligned(Entity *car, Entity *lane, int geometry_idx) {
    // Validate lane geometry length
    if (!lane || lane->array_size < 2)
        return 0;

    // Clamp geometry index to valid segment range [0, array_size-2]
    if (geometry_idx < 0)
        geometry_idx = 0;
    if (geometry_idx >= lane->array_size - 1)
        geometry_idx = lane->array_size - 2;

    // Compute local lane segment heading
    float heading_x1, heading_y1;
    if (geometry_idx > 0) {
        heading_x1 = lane->traj_x[geometry_idx] - lane->traj_x[geometry_idx - 1];
        heading_y1 = lane->traj_y[geometry_idx] - lane->traj_y[geometry_idx - 1];
    } else {
        // For first segment, just use the forward direction
        heading_x1 = lane->traj_x[geometry_idx + 1] - lane->traj_x[geometry_idx];
        heading_y1 = lane->traj_y[geometry_idx + 1] - lane->traj_y[geometry_idx];
    }

    float heading_x2 = lane->traj_x[geometry_idx + 1] - lane->traj_x[geometry_idx];
    float heading_y2 = lane->traj_y[geometry_idx + 1] - lane->traj_y[geometry_idx];

    float heading_1 = atan2f(heading_y1, heading_x1);
    float heading_2 = atan2f(heading_y2, heading_x2);
    float heading = (heading_1 + heading_2) / 2.0f;

    // Normalize to [-pi, pi]
    if (heading > M_PI)
        heading -= 2.0f * M_PI;
    if (heading < -M_PI)
        heading += 2.0f * M_PI;

    // Compute heading difference
    float car_heading = car->heading; // radians
    float heading_diff = fabsf(car_heading - heading);

    if (heading_diff > M_PI)
        heading_diff = 2.0f * M_PI - heading_diff;

    // within 15 degrees
    return (heading_diff < (M_PI / 12.0f)) ? 1 : 0;
}

void reset_agent_metrics(Drive *env, int agent_idx) {
    Entity *agent = &env->entities[agent_idx];
    agent->metrics_array[COLLISION_IDX] = 0.0f;    // vehicle collision
    agent->metrics_array[OFFROAD_IDX] = 0.0f;      // offroad
    agent->metrics_array[LANE_ALIGNED_IDX] = 0.0f; // lane aligned
    agent->collision_state = 0;
}

float point_to_segment_distance_2d(float px, float py, float x1, float y1, float x2, float y2) {
    float dx = x2 - x1;
    float dy = y2 - y1;

    if (dx == 0 && dy == 0) {
        // The segment is a point
        return sqrtf((px - x1) * (px - x1) + (py - y1) * (py - y1));
    }

    // Calculate the t that minimizes the distance
    float t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy);

    // Clamp t to the segment
    if (t < 0)
        t = 0;
    else if (t > 1)
        t = 1;

    // Find the closest point on the segment
    float closestX = x1 + t * dx;
    float closestY = y1 + t * dy;

    // Return the distance from p to the closest point
    return sqrtf((px - closestX) * (px - closestX) + (py - closestY) * (py - closestY));
}

void compute_agent_metrics(Drive *env, int agent_idx) {
    Entity *agent = &env->entities[agent_idx];

    reset_agent_metrics(env, agent_idx);

    if (agent->x == INVALID_POSITION)
        return; // invalid agent position

    int collided = 0;
    float half_length = agent->length / 2.0f;
    float half_width = agent->width / 2.0f;
    float cos_heading = cosf(agent->heading);
    float sin_heading = sinf(agent->heading);
    float min_distance = (float)INT16_MAX;

    int closest_lane_entity_idx = -1;
    int closest_lane_geometry_idx = -1;

    float corners[4][2];
    for (int i = 0; i < 4; i++) {
        corners[i][0] =
            agent->x + (offsets[i][0] * half_length * cos_heading - offsets[i][1] * half_width * sin_heading);
        corners[i][1] =
            agent->y + (offsets[i][0] * half_length * sin_heading + offsets[i][1] * half_width * cos_heading);
    }

    GridMapEntity entity_list[MAX_ENTITIES_PER_CELL * 25]; // Array big enough for all neighboring cells
    int list_size =
        checkNeighbors(env, agent->x, agent->y, entity_list, MAX_ENTITIES_PER_CELL * 25, collision_offsets, 25);
    for (int i = 0; i < list_size; i++) {
        if (entity_list[i].entity_idx == -1)
            continue;
        if (entity_list[i].entity_idx == agent_idx)
            continue;
        Entity *entity;
        entity = &env->entities[entity_list[i].entity_idx];

        // Check for offroad collision with road edges. Applies to every controlled
        // agent, pedestrians included: see the note in collision_check.
        if (entity->type == ROAD_EDGE) {
            int geometry_idx = entity_list[i].geometry_idx;
            float start[2] = {entity->traj_x[geometry_idx], entity->traj_y[geometry_idx]};
            float end[2] = {entity->traj_x[geometry_idx + 1], entity->traj_y[geometry_idx + 1]};
            for (int k = 0; k < 4; k++) { // Check each edge of the bounding box
                int next = (k + 1) % 4;
                if (check_line_intersection(corners[k], corners[next], start, end)) {
                    collided = OFFROAD;
                    break;
                }
            }
        }

        if (collided == OFFROAD)
            break;

        // Find closest point on the road centerline to the agent
        if (entity->type == ROAD_LANE) {
            int entity_idx = entity_list[i].entity_idx;
            int geometry_idx = entity_list[i].geometry_idx;

            float start[2] = {entity->traj_x[geometry_idx], entity->traj_y[geometry_idx]};
            float end[2] = {entity->traj_x[geometry_idx + 1], entity->traj_y[geometry_idx + 1]};

            float dist = point_to_segment_distance_2d(agent->x, agent->y, start[0], start[1], end[0], end[1]);
            float heading_diff = fabsf(atan2f(end[1] - start[1], end[0] - start[0]) - agent->heading);

            // Normalize heading difference to [0, pi]
            if (heading_diff > M_PI)
                heading_diff = 2.0f * M_PI - heading_diff;

            // Penalize if heading differs by more than 30 degrees
            if (heading_diff > (M_PI / 6.0f))
                dist += 3.0f;

            if (dist < min_distance) {
                min_distance = dist;
                closest_lane_entity_idx = entity_idx;
                closest_lane_geometry_idx = geometry_idx;
            }
        }
    }

    // check if aligned with closest lane and set current lane
    // 4.0m threshold: agents more than 4 meters from any lane are considered off-road
    if (min_distance > 4.0f || closest_lane_entity_idx == -1) {
        agent->metrics_array[LANE_ALIGNED_IDX] = 0.0f;
        agent->current_lane_idx = -1;
        agent->lane_valid = 0;
        agent->lane_heading_error = 0.0f;
        agent->lane_lateral_offset = 0.0f;
    } else {
        agent->current_lane_idx = closest_lane_entity_idx;
        int lane_aligned =
            check_lane_aligned(agent, &env->entities[closest_lane_entity_idx], closest_lane_geometry_idx);
        agent->metrics_array[LANE_ALIGNED_IDX] = lane_aligned;

        // Frenet frame of the closest segment: heading error wrapped to [-pi, pi] and
        // lateral offset signed by which side of the lane the agent sits on (positive
        // left). The sign matters -- alpha_center_bias picks a side, so an unsigned
        // offset would make the left and right halves of the lane indistinguishable.
        Entity *lane = &env->entities[closest_lane_entity_idx];
        int g = closest_lane_geometry_idx;
        float sx = lane->traj_x[g], sy = lane->traj_y[g];
        float dx = lane->traj_x[g + 1] - sx, dy = lane->traj_y[g + 1] - sy;
        float seg_len = sqrtf(dx * dx + dy * dy);
        if (seg_len > 1e-6f) {
            dx /= seg_len;
            dy /= seg_len;
            float err = agent->heading - atan2f(dy, dx);
            while (err > (float)M_PI)
                err -= 2.0f * (float)M_PI;
            while (err < -(float)M_PI)
                err += 2.0f * (float)M_PI;
            agent->lane_heading_error = err;
            agent->lane_lateral_offset = -dy * (agent->x - sx) + dx * (agent->y - sy);
            agent->lane_valid = 1;
        } else {
            agent->lane_valid = 0;
            agent->lane_heading_error = 0.0f;
            agent->lane_lateral_offset = 0.0f;
        }
    }

    // Check for vehicle collisions
    int car_collided_with_index = collision_check(env, agent_idx);
    if (car_collided_with_index != -1)
        collided = VEHICLE_COLLISION;

    agent->collision_state = collided;

    if (collided == VEHICLE_COLLISION) {
        if (env->collision_behavior == STOP_AGENT && !agent->stopped) {
            agent->stopped = 1;
            agent->vx = agent->vy = 0.0f;
        } else if (env->collision_behavior == REMOVE_AGENT && !agent->removed) {
            Entity *agent_collided = &env->entities[car_collided_with_index];
            agent->removed = 1;
            agent_collided->removed = 1;
            agent->x = agent->y = -10000.0f;
            agent_collided->x = agent_collided->y = -10000.0f;
        }
    }
    if (collided == OFFROAD) {
        agent->metrics_array[OFFROAD_IDX] = 1.0f;
        if (env->offroad_behavior == STOP_AGENT && !agent->stopped) {
            agent->stopped = 1;
            agent->vx = agent->vy = 0.0f;
        } else if (env->offroad_behavior == REMOVE_AGENT && !agent->removed) {
            agent->removed = 1;
            agent->x = agent->y = -10000.0f;
        }
    }

    return;
}

// ---------------------------------------------------------------------------
// Gigaflow-style random initialization
//
// Replaces the dataset-driven placement entirely: no logged track supplies a pose,
// a size or a goal. Agents are drawn from the WOMD state distribution, placed by
// rejection sampling along the lane graph, and routed over it.
// ---------------------------------------------------------------------------

#define TEDDY_SPAWN_ATTEMPTS 48
#define TEDDY_SPAWN_GAP 1.5f // metres of clear space required between footprints

// True if the agent's box crosses a road edge. Spawning happens on lane centerlines,
// so "outside the road" is not a failure mode that needs testing; what this catches
// is a footprint wide or long enough to straddle a curb, which is a real outcome
// once sizes are sampled independently of the lane that hosts them.
static int teddy_is_offroad(Drive *env, Entity *agent) {
    if (agent->type == PEDESTRIAN)
        return 0; // pedestrians are exempt from offroad everywhere else too
    float half_length = agent->length / 2.0f;
    float half_width = agent->width / 2.0f;
    float ch = agent->heading_x;
    float sh = agent->heading_y;
    float corners[4][2];
    for (int i = 0; i < 4; i++) {
        corners[i][0] = agent->x + (offsets[i][0] * half_length * ch - offsets[i][1] * half_width * sh);
        corners[i][1] = agent->y + (offsets[i][0] * half_length * sh + offsets[i][1] * half_width * ch);
    }
    GridMapEntity entity_list[MAX_ENTITIES_PER_CELL * 25];
    int list_size =
        checkNeighbors(env, agent->x, agent->y, entity_list, MAX_ENTITIES_PER_CELL * 25, collision_offsets, 25);
    for (int i = 0; i < list_size; i++) {
        if (entity_list[i].entity_idx == -1)
            continue;
        Entity *e = &env->entities[entity_list[i].entity_idx];
        if (e->type != ROAD_EDGE)
            continue;
        int g = entity_list[i].geometry_idx;
        float start[2] = {e->traj_x[g], e->traj_y[g]};
        float end[2] = {e->traj_x[g + 1], e->traj_y[g + 1]};
        for (int k = 0; k < 4; k++)
            if (check_line_intersection(corners[k], corners[(k + 1) % 4], start, end))
                return 1;
    }
    return 0;
}

// Slack in metres between this footprint at (x, y) and the nearest already-placed
// agent: negative means they overlap. Circumscribed radii keep it heading-agnostic,
// which is what lets the caller score a candidate before committing to it.
static float teddy_clearance(Drive *env, int agent_idx, float x, float y, int count) {
    Entity *a = &env->entities[agent_idx];
    float ra = 0.5f * sqrtf(a->length * a->length + a->width * a->width);
    float best = 1e30f;
    for (int j = 0; j < count; j++) {
        if (j == agent_idx)
            continue;
        Entity *o = &env->entities[j];
        if (o->removed)
            continue;
        float dx = o->x - x, dy = o->y - y;
        float ro = 0.5f * sqrtf(o->length * o->length + o->width * o->width);
        float slack = sqrtf(dx * dx + dy * dy) - (ra + ro + TEDDY_SPAWN_GAP);
        if (slack < best)
            best = slack;
    }
    return best;
}

// True if this agent's box, at its current pose, intersects any already-placed one.
//
// teddy_clearance uses circumscribed circles, which is the right thing to *rank*
// candidates by but the wrong thing to accept on: two cars abreast in neighbouring
// lanes have intersecting circles and perfectly disjoint boxes. Accepting on circles
// makes dense maps unplaceable and forces the best-effort fallback, which is where the
// real overlaps came from. The circle test survives as a cheap reject in front of the
// same oriented-box check the simulator's own collision detection uses.
static int teddy_overlaps_any(Drive *env, int agent_idx, int count) {
    Entity *a = &env->entities[agent_idx];
    float ra = 0.5f * sqrtf(a->length * a->length + a->width * a->width);
    for (int j = 0; j < count; j++) {
        if (j == agent_idx)
            continue;
        Entity *o = &env->entities[j];
        if (o->removed)
            continue;
        float dx = o->x - a->x, dy = o->y - a->y;
        float ro = 0.5f * sqrtf(o->length * o->length + o->width * o->width);
        float reach = ra + ro;
        if (dx * dx + dy * dy > reach * reach)
            continue;
        if (check_aabb_collision(a, o))
            return 1;
    }
    return 0;
}

// Waypoint chain: N_wp ~ U{0, num_waypoints_max} intermediate points plus a final
// goal, each one a random forward walk further along the lane graph.
//
// Gigaflow spaces waypoints 20-200 m apart. That upper bound assumes CARLA towns
// with 4-40 km of lane; a WOMD crop is ~274 m across and a random walk covers 168 m
// on average, so a 200 m draw would land in the paper's constraint-relaxation branch
// most of the time. The bound is configurable and defaults to 80 m instead. Walks
// that dead-end short (usually at the map boundary) end the chain, which is the
// relaxation behaviour the paper describes.
static void teddy_sample_route(Drive *env, int agent_idx) {
    Entity *a = &env->entities[agent_idx];
    LaneGraph *lg = &env->lane_graph;
    a->num_waypoints = 0;
    a->current_waypoint = 0;

    if (lg->num_lanes == 0) {
        a->waypoints[0][0] = a->x;
        a->waypoints[0][1] = a->y;
        a->num_waypoints = 1;
    } else {
        int n_wp = teddy_rand_int(&env->rng, 0, env->num_waypoints_max) + 1; // + final goal
        if (n_wp > MAX_WAYPOINTS)
            n_wp = MAX_WAYPOINTS;
        int lane = a->spawn_lane;
        float s = a->spawn_s;
        for (int i = 0; i < n_wp; i++) {
            float want = teddy_rand_range(&env->rng, env->waypoint_min_dist, env->waypoint_max_dist);
            int next_lane;
            float next_s;
            float got = lane_walk_forward(lg, &env->rng, lane, s, want, &next_lane, &next_s);
            // Commit only if the walk actually moved. Appending regardless would stack
            // waypoints on the same spot every time a walk starts at a dead end -- and
            // a final goal coincident with the waypoint before it is reached for free.
            if (got < 1.0f)
                break;
            float x, y, h;
            lane_pose_at(lg, env->entities, next_lane, next_s, &x, &y, &h);
            a->waypoints[a->num_waypoints][0] = x;
            a->waypoints[a->num_waypoints][1] = y;
            a->num_waypoints++;
            lane = next_lane;
            s = next_s;
        }

        // Spawned somewhere with no room to move forward at all (the end of a lane
        // that leaves the map crop). Fall back to a goal elsewhere on the network
        // rather than leaving the agent with none: Gigaflow samples the first goal
        // uniformly over the map anyway, so a goal that is not lane-reachable from
        // here is within the spirit of the design.
        if (a->num_waypoints == 0) {
            int lane2;
            float s2, x, y, h;
            if (sample_lane_point(lg, &env->rng, &lane2, &s2)) {
                lane_pose_at(lg, env->entities, lane2, s2, &x, &y, &h);
                a->waypoints[0][0] = x;
                a->waypoints[0][1] = y;
            } else {
                a->waypoints[0][0] = a->x;
                a->waypoints[0][1] = a->y;
            }
            a->num_waypoints = 1;
        }
    }

    // goal_position_* always mirrors the *current* waypoint, so every existing
    // consumer -- the ego observation, the goal reward, the renderer -- keeps working
    // without knowing routes exist.
    a->goal_position_x = a->waypoints[0][0];
    a->goal_position_y = a->waypoints[0][1];
    a->goal_position_z = 0.0f;
    a->init_goal_x = a->waypoints[0][0];
    a->init_goal_y = a->waypoints[0][1];
    a->goals_sampled_this_episode += 1.0f;
}

// Draws this agent's embodiment and pose. `count` is how many entries of
// env->entities are already placed and should be avoided.
static void teddy_place_agent(Drive *env, int agent_idx, int count) {
    Entity *a = &env->entities[agent_idx];
    LaneGraph *lg = &env->lane_graph;

    // Size first: the clearance test needs the footprint.
    agent_dist_sample(&env->rng, &a->type, &a->length, &a->width, &a->height);
    a->lane_valid = 0;
    a->lane_heading_error = 0.0f;
    a->lane_lateral_offset = 0.0f;
    a->wheelbase = 0.6f * a->length;
    a->removed = 0;
    a->stopped = 0;

    if (lg->num_lanes == 0) {
        // Cannot happen on the WOMD corpus (all 10000 maps carry >= 2 lanes), but a
        // silently stacked scene would be far worse than an obviously absent agent.
        a->removed = 1;
        a->x = a->y = INVALID_POSITION;
        a->num_waypoints = 0;
        return;
    }

    float jitter = env->spawn_heading_jitter_deg * (float)M_PI / 180.0f;
    float best_x = 0.0f, best_y = 0.0f, best_h = 0.0f, best_s = 0.0f, best_score = -1e30f;
    int best_lane = 0;

    for (int attempt = 0; attempt < TEDDY_SPAWN_ATTEMPTS; attempt++) {
        int lane;
        float s;
        if (!sample_lane_point(lg, &env->rng, &lane, &s))
            break;
        float x, y, h;
        lane_pose_at(lg, env->entities, lane, s, &x, &y, &h);
        h += teddy_rand_normal(&env->rng) * jitter;
        if (teddy_rand_float(&env->rng) < env->wrong_way_frac)
            h += (float)M_PI;

        float clear = teddy_clearance(env, agent_idx, x, y, count);

        // Prefer somewhere with road ahead. An agent dropped within a metre of a
        // terminal lane end -- the map crop boundary -- has no route to generate, and
        // teddy_sample_route then has to fall back to a goal it cannot reach by
        // following lanes. Scoring it down here is cheaper than repairing it later,
        // and it reuses the rejection loop that is already running.
        const Lane *cand = &lg->lanes[lane];
        int dead_end = (cand->num_succ == 0 && (cand->length - s) < env->waypoint_min_dist);

        // Provisional placement so the offroad test sees the real oriented box.
        a->x = x;
        a->y = y;
        a->heading = h;
        a->heading_x = cosf(h);
        a->heading_y = sinf(h);
        // Ranking for the fallback, when no attempt turns out fully legal. The
        // penalties are ordered by how bad the outcome actually is: starting inside a
        // wall is unrecoverable, starting inside another car costs an immediate
        // collision, and a dead-end spawn merely yields a goal that has to be sampled
        // elsewhere. They are far larger than any clearance value (bounded by a few
        // metres) so the ordering is strict and clearance only breaks ties.
        int offroad = teddy_is_offroad(env, a);
        int overlap = teddy_overlaps_any(env, agent_idx, count);
        float score = clear - (offroad ? 1000.0f : 0.0f) - (overlap ? 200.0f : 0.0f) - (dead_end ? 50.0f : 0.0f);

        if (score > best_score) {
            best_score = score;
            best_x = x;
            best_y = y;
            best_h = h;
            best_lane = lane;
            best_s = s;
        }
        // Accept as soon as the pose is legal: on the road, with road ahead, and not
        // inside anyone.
        if (!offroad && !overlap && !dead_end)
            break;
    }

    // The agent count per scene is fixed by my_shared before any map is read, and
    // Python has already sized the observation slice to match, so a map too small to
    // hold them all must still yield exactly that many agents. Falling back to the
    // roomiest candidate degrades density rather than breaking the contract.
    // Gigaflow can instead shrink the set, because it picks the count after placing.
    a->x = best_x;
    a->y = best_y;
    a->z = 0.0f;
    a->heading = best_h;
    a->heading_x = cosf(best_h);
    a->heading_y = sinf(best_h);
    a->spawn_lane = best_lane;
    a->spawn_s = best_s;

    float speed = teddy_rand_range(&env->rng, 0.0f, env->spawn_speed_max);
    a->vx = speed * a->heading_x;
    a->vy = speed * a->heading_y;
    a->vz = 0.0f;

    a->valid = 1;
    a->collision_state = 0;
    a->respawn_timestep = -1;
    a->current_goal_reached = 0;
    a->collided_before_goal = 0;
    a->a_long = 0.0f;
    a->a_lat = 0.0f;
    a->jerk_long = 0.0f;
    a->jerk_lat = 0.0f;
    a->steering_angle = 0.0f;
    for (int m = 0; m < 4; m++)
        a->metrics_array[m] = 0.0f;

    // Keep the one-sample trajectory in step with the spawn pose; a few helpers
    // (renderer traces, the WOSAC export) still read traj_*[0].
    a->traj_x[0] = a->x;
    a->traj_y[0] = a->y;
    a->traj_z[0] = a->z;
    a->traj_vx[0] = a->vx;
    a->traj_vy[0] = a->vy;
    a->traj_vz[0] = a->vz;
    a->traj_heading[0] = a->heading;
    a->traj_valid[0] = 1;

    teddy_sample_route(env, agent_idx);
}

// Discards the logged tracks and puts `n_agents` blank synthetic agents in their
// place. Roads are carried over untouched, and the "objects first, roads second"
// layout is preserved so every downstream loop keeps working unchanged.
static void teddy_build_agents(Drive *env, int n_agents) {
    if (n_agents < 1)
        n_agents = 1;
    if (n_agents > MAX_AGENTS)
        n_agents = MAX_AGENTS;

    int num_roads = env->num_entities - env->num_objects;
    Entity *fresh = (Entity *)calloc((size_t)(n_agents + num_roads), sizeof(Entity));
    memcpy(&fresh[n_agents], &env->entities[env->num_objects], (size_t)num_roads * sizeof(Entity));

    // Only the logged tracks are freed; the roads' polyline arrays are now owned by
    // `fresh`, so freeing the whole old array here would be a double free.
    for (int i = 0; i < env->num_objects; i++)
        free_entity(&env->entities[i]);
    free(env->entities);

    for (int i = 0; i < n_agents; i++) {
        Entity *a = &fresh[i];
        a->id = i;
        a->type = VEHICLE; // replaced by agent_dist_sample when placed
        // A one-sample trajectory rather than NULL: several helpers still index
        // traj_*[0], and a length-1 array lets them read the spawn pose instead of
        // dereferencing NULL.
        a->array_size = 1;
        a->traj_x = (float *)calloc(1, sizeof(float));
        a->traj_y = (float *)calloc(1, sizeof(float));
        a->traj_z = (float *)calloc(1, sizeof(float));
        a->traj_vx = (float *)calloc(1, sizeof(float));
        a->traj_vy = (float *)calloc(1, sizeof(float));
        a->traj_vz = (float *)calloc(1, sizeof(float));
        a->traj_heading = (float *)calloc(1, sizeof(float));
        a->traj_valid = (int *)calloc(1, sizeof(int));
        a->traj_valid[0] = 1;
        a->valid = 1;
        a->respawn_timestep = -1;
        a->current_lane_idx = -1;
        a->length = 4.7f;
        a->width = 2.1f;
        a->height = 1.7f;
    }

    env->entities = fresh;
    env->num_objects = n_agents;
    env->num_roads = num_roads;
    env->num_entities = n_agents + num_roads;
    env->sdc_track_index = -1; // a synthetic scene has no logged ego
}

// Every synthetic agent is policy-controlled. There are no static or expert-replay
// agents in a Gigaflow scene: nothing is replaying anything.
static void teddy_set_active_agents(Drive *env) {
    int n = env->num_objects;
    env->active_agent_count = n;
    env->num_actors = n;
    env->static_agent_count = 0;
    env->expert_static_agent_count = 0;
    env->active_agent_indices = (int *)malloc((size_t)n * sizeof(int));
    for (int i = 0; i < n; i++) {
        env->active_agent_indices[i] = i;
        env->entities[i].active_agent = 1;
    }
    // c_close frees these unconditionally.
    env->static_agent_indices = (int *)calloc(1, sizeof(int));
    env->expert_static_agent_indices = (int *)calloc(1, sizeof(int));
}

// Sequential rejection sampling, as in Gigaflow App. A.1: place agent i avoiding the
// i-1 already placed. Sampling a hundred poses jointly and hoping none collide has a
// vanishing acceptance rate on a map this size.
static void teddy_spawn_all(Drive *env) {
    for (int i = 0; i < env->num_objects; i++)
        teddy_place_agent(env, i, i);
}

// DEAD IN TEDDY. Reached only from set_active_agents below, which nothing calls:
// teddy_set_active_agents replaces it and makes every synthesised entity active
// regardless of type. Kept so the file still reads against ocean's, but note the
// consequence -- `control_mode` selects nothing here, so CONTROL_VEHICLES does NOT
// restrict teddy to type == VEHICLE the way it does in ocean.
bool should_control_agent(Drive *env, int agent_idx, int control_limit) {
    // Check if we have room for more agents or are already at capacity
    if (env->active_agent_count >= control_limit) {
        return false;
    }

    Entity *entity = &env->entities[agent_idx];

    if (env->control_mode == CONTROL_SDC_ONLY) {
        return agent_idx == env->sdc_track_index;
    }

    bool is_vehicle = (entity->type == VEHICLE);
    bool is_ped_or_bike = (entity->type == PEDESTRIAN || entity->type == CYCLIST);
    bool type_is_valid = false;

    switch (env->control_mode) {
    case CONTROL_WOSAC:
        // Valid types only, ignore expert flag and goal distance
        return (is_vehicle || is_ped_or_bike);

    case CONTROL_VEHICLES:
        type_is_valid = is_vehicle;
        break;

    default:
        type_is_valid = (is_vehicle || is_ped_or_bike);
        break;
    }

    // Filter invalid types or experts
    if (!type_is_valid || entity->mark_as_expert) {
        return false;
    }

    // Check distance to goal in agent's local frame
    float cos_heading = cosf(entity->traj_heading[0]);
    float sin_heading = sinf(entity->traj_heading[0]);
    float goal_dx = entity->goal_position_x - entity->traj_x[0];
    float goal_dy = entity->goal_position_y - entity->traj_y[0];

    // Transform to agent's local frame
    float local_goal_x = goal_dx * cos_heading + goal_dy * sin_heading;
    float local_goal_y = -goal_dx * sin_heading + goal_dy * cos_heading;
    float distance_to_goal = relative_distance_2d(0, 0, local_goal_x, local_goal_y);

    return distance_to_goal >= MIN_DISTANCE_TO_GOAL;
}

// DEAD IN TEDDY: superseded by teddy_set_active_agents. This is the dataset-driven
// selector; it reads traj_valid and the SDC index, neither of which a synthesised
// Gigaflow scene has.
void set_active_agents(Drive *env) {

    // Initialize
    env->active_agent_count = 0;        // Policy-controlled agents
    env->static_agent_count = 0;        // Non-moving background agents
    env->expert_static_agent_count = 0; // Expert replay agents (non-controlled)
    env->num_actors = 0;                // Total agents created (there is always the SDC)

    int active_agent_indices[MAX_AGENTS];
    int static_agent_indices[MAX_AGENTS];
    int expert_static_agent_indices[MAX_AGENTS];

    if (env->num_agents == 0) {
        env->num_agents = MAX_AGENTS;
    }

    int control_limit;
    if (env->control_mode == CONTROL_MIXED_PLAY) {
        control_limit = (env->max_controlled_agents < env->num_agents) ? env->max_controlled_agents : env->num_agents;
    } else {
        control_limit = env->num_agents;
    }

    // If we have a SDC index (WOMD), initialize it first:
    int sdc_index = env->sdc_track_index;

    if (sdc_index >= 0) {
        active_agent_indices[0] = sdc_index;
        env->num_actors++;
        env->active_agent_count++;
        env->entities[sdc_index].active_agent = 1;
    }

    // Iterate through entities to find agents to create and/or control
    for (int i = 0; i < env->num_objects && env->num_actors < MAX_AGENTS; i++) {

        // Skip if its the SDC
        if (i == sdc_index) {
            continue;
        }

        Entity *entity = &env->entities[i];

        // Skip if not valid at initialization
        if (entity->traj_valid[env->init_steps] != 1) {
            continue;
        }

        // Determine if entity should be created
        bool should_create = false;
        if (env->init_mode == INIT_ALL_VALID) {
            should_create = true; // All valid entities
        } else if (env->control_mode == CONTROL_VEHICLES) {
            should_create = (entity->type == VEHICLE);
        } else { // Control all agents
            should_create = (entity->type == VEHICLE || entity->type == PEDESTRIAN || entity->type == CYCLIST);
        }

        if (!should_create)
            continue;

        env->num_actors++;

        // Determine if this agent should be policy-controlled
        bool is_controlled = false;

        is_controlled = should_control_agent(env, i, control_limit);

        if (is_controlled) {
            active_agent_indices[env->active_agent_count] = i;
            env->active_agent_count++;
            env->entities[i].active_agent = 1;
        } else if (env->init_mode != INIT_ONLY_CONTROLLABLE_AGENTS) {
            static_agent_indices[env->static_agent_count] = i;
            env->static_agent_count++; // Includes expert replay and static agents
            env->entities[i].active_agent = 0;
            if (env->entities[i].mark_as_expert == 1 || env->active_agent_count == control_limit) {
                expert_static_agent_indices[env->expert_static_agent_count] = i;
                env->expert_static_agent_count++;
                env->entities[i].mark_as_expert = 1;
            }
        }
    }

    // Set up initial active agents
    env->active_agent_indices = (int *)malloc(env->active_agent_count * sizeof(int));
    env->static_agent_indices = (int *)malloc(env->static_agent_count * sizeof(int));
    env->expert_static_agent_indices = (int *)malloc(env->expert_static_agent_count * sizeof(int));
    for (int i = 0; i < env->active_agent_count; i++) {
        env->active_agent_indices[i] = active_agent_indices[i];
    };
    for (int i = 0; i < env->static_agent_count; i++) {
        env->static_agent_indices[i] = static_agent_indices[i];
    }
    for (int i = 0; i < env->expert_static_agent_count; i++) {
        env->expert_static_agent_indices[i] = expert_static_agent_indices[i];
    }
    // printf("Total actors: %d, Active agents: %d, Static agents: %d, Expert static agents: %d\n", env->num_actors,
    //        env->active_agent_count, env->static_agent_count, env->expert_static_agent_count);
    // printf("Control mode: %d, max controlled agents: %d\n", env->control_mode, env->max_controlled_agents);

    return;
}

// DEAD IN TEDDY: no caller, and it returns immediately unless control_mode is
// CONTROL_WOSAC, which teddy never uses.
void remove_bad_trajectories(Drive *env) {

    if (env->control_mode != CONTROL_WOSAC) {
        return; // Leave all trajectories in WOSAC control mode
    }

    set_start_position(env);
    int collided_agents[env->active_agent_count];
    int collided_with_indices[env->active_agent_count];
    memset(collided_agents, 0, env->active_agent_count * sizeof(int));
    for (int i = 0; i < env->active_agent_count; ++i) {
        collided_with_indices[i] = -1;
    }
    // move experts through trajectories to check for collisions and remove as illegal agents
    for (int t = 0; t < env->episode_length; t++) {
        for (int i = 0; i < env->active_agent_count; i++) {
            int agent_idx = env->active_agent_indices[i];
            move_expert(env, env->actions, agent_idx);
        }
        for (int i = 0; i < env->expert_static_agent_count; i++) {
            int expert_idx = env->expert_static_agent_indices[i];
            if (env->entities[expert_idx].x == INVALID_POSITION)
                continue;
            move_expert(env, env->actions, expert_idx);
        }
        // check collisions
        for (int i = 0; i < env->active_agent_count; i++) {
            int agent_idx = env->active_agent_indices[i];
            env->entities[agent_idx].collision_state = 0;
            int collided_with_index = collision_check(env, agent_idx);
            if ((collided_with_index >= 0) && collided_agents[i] == 0) {
                collided_agents[i] = 1;
                collided_with_indices[i] = collided_with_index;
            }
        }
        env->timestep++;
    }

    for (int i = 0; i < env->active_agent_count; i++) {
        if (collided_with_indices[i] == -1)
            continue;
        for (int j = 0; j < env->static_agent_count; j++) {
            int static_agent_idx = env->static_agent_indices[j];
            if (static_agent_idx != collided_with_indices[i])
                continue;
            env->entities[static_agent_idx].traj_x[0] = INVALID_POSITION;
            env->entities[static_agent_idx].traj_y[0] = INVALID_POSITION;
        }
    }
    env->timestep = 0;
}

void init_goal_positions(Drive *env) {
    for (int x = 0; x < env->active_agent_count; x++) {
        int agent_idx = env->active_agent_indices[x];
        env->entities[agent_idx].init_goal_x = env->entities[agent_idx].goal_position_x;
        env->entities[agent_idx].init_goal_y = env->entities[agent_idx].goal_position_y;
    }
}

void init(Drive *env) {
    env->human_agent_idx = 0;
    env->timestep = 0;
    if (env->render_road_types == 0)
        env->render_road_types = RENDER_ROAD_TYPES_DEFAULT;
    env->entities = load_map_binary(env->map_name, env);
    // set_means still centres the map using the logged tracks, before they are
    // discarded, so teddy and ocean put the same map in the same place.
    set_means(env);
    teddy_build_agents(env, env->num_agents);
    // After teddy_build_agents: the graph stores entity indices, and rebuilding the
    // array moves every road. After set_means: it stores interpolated poses, so it
    // must be in the same map-centered frame as everything else.
    build_lane_graph(&env->lane_graph, env->entities, env->num_objects, env->num_entities, ROAD_LANE);
    init_grid_map(env);
    env->grid_map->vision_range = 21; // TODO: Why is this hardcoded?
    init_neighbor_offsets(env);
    cache_neighbor_offsets(env);
    env->logs_capacity = 0;
    teddy_set_active_agents(env);
    env->logs_capacity = env->active_agent_count;
    // Must follow init_grid_map: spawn rejection queries the grid for road edges.
    teddy_spawn_all(env);
    env->logs = (Log *)calloc(env->active_agent_count, sizeof(Log));
}

void close_client(Client *client);

void c_close(Drive *env) {
    if (env->client != NULL) {
        close_client(env->client);
        env->client = NULL;
    }
    for (int i = 0; i < env->num_entities; i++) {
        free_entity(&env->entities[i]);
    }
    free(env->entities);
    free(env->active_agent_indices);
    free(env->logs);
    // GridMap cleanup
    int grid_cell_count = env->grid_map->grid_cols * env->grid_map->grid_rows;
    for (int grid_index = 0; grid_index < grid_cell_count; grid_index++) {
        free(env->grid_map->cells[grid_index]);
    }
    free(env->grid_map->cells);
    free(env->grid_map->cell_entities_count);
    free(env->neighbor_offsets);

    for (int i = 0; i < grid_cell_count; i++) {
        free(env->grid_map->neighbor_cache_entities[i]);
    }
    free(env->grid_map->neighbor_cache_entities);
    free(env->grid_map->neighbor_cache_count);
    free(env->grid_map);
    free_lane_graph(&env->lane_graph);
    free(env->static_agent_indices);
    free(env->expert_static_agent_indices);
    free(env->ini_file);
    free(env->tracks_to_predict_indices);
    env->tracks_to_predict_indices = NULL;
}

void allocate(Drive *env) {
    init(env);
    int max_obs = drive_obs_size(env);
    env->observations = (float *)calloc(env->active_agent_count * max_obs, sizeof(float));
    env->actions = (float *)calloc(env->active_agent_count * 2, sizeof(float));
    env->rewards = (float *)calloc(env->active_agent_count, sizeof(float));
    env->terminals = (unsigned char *)calloc(env->active_agent_count, sizeof(unsigned char));
    env->truncations = (unsigned char *)calloc(env->active_agent_count, sizeof(unsigned char));
}

void free_allocated(Drive *env) {
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    free(env->truncations);
    c_close(env);
}

float clipSpeed(float speed) {
    const float maxSpeed = MAX_SPEED;
    if (speed > maxSpeed)
        return maxSpeed;
    if (speed < -maxSpeed)
        return -maxSpeed;
    return speed;
}

float normalize_heading(float heading) {
    if (heading > M_PI)
        heading -= 2 * M_PI;
    if (heading < -M_PI)
        heading += 2 * M_PI;
    return heading;
}

float normalize_value(float value, float min, float max) { return (value - min) / (max - min); }

void move_dynamics(Drive *env, int action_idx, int agent_idx) {
    Entity *agent = &env->entities[agent_idx];
    if (agent->removed)
        return;

    if (agent->stopped) {
        agent->vx = 0.0f;
        agent->vy = 0.0f;
        return;
    }

    if (env->dynamics_model == CLASSIC) {
        // Classic dynamics model
        float acceleration = 0.0f;
        float steering = 0.0f;

        if (env->action_type == 1) { // continuous
            float (*action_array_f)[2] = (float (*)[2])env->actions;
            acceleration = action_array_f[action_idx][0];
            steering = action_array_f[action_idx][1];

            acceleration *= ACCELERATION_VALUES[6];
            steering *= STEERING_VALUES[12];
        } else { // discrete
            // Interpret action as a single integer: a = accel_idx * num_steer + steer_idx
            int *action_array = (int *)env->actions;
            int num_steer = sizeof(STEERING_VALUES) / sizeof(STEERING_VALUES[0]);
            int action_val = action_array[action_idx];
            int acceleration_index = action_val / num_steer;
            int steering_index = action_val % num_steer;
            acceleration = ACCELERATION_VALUES[acceleration_index];
            steering = STEERING_VALUES[steering_index];
        }

        // Every agent shares one actuation response, as in ocean. Gigaflow randomizes
        // it per agent (C_throttle / C_steer / C_acc) and hands the draw to the policy
        // in the observation; here there is no conditioning block to put it in, so a
        // randomized response would be an unobservable disturbance rather than an
        // embodiment the policy can adapt to.
        acceleration = clip(acceleration, -5.0f, 2.5f);

        // Current state
        float x = agent->x;
        float y = agent->y;
        float heading = agent->heading;
        float vx = agent->vx;
        float vy = agent->vy;

        // Calculate current speed (signed based on direction relative to heading)
        float speed_magnitude = sqrtf(vx * vx + vy * vy);
        float v_dot_heading = vx * agent->heading_x + vy * agent->heading_y;
        float signed_speed = copysignf(speed_magnitude, v_dot_heading);

        // Update speed with acceleration
        signed_speed = signed_speed + acceleration * env->dt;
        signed_speed = clipSpeed(signed_speed);
        // Compute yaw rate
        float beta = tanh(.5 * tanf(steering));

        // New heading
        float yaw_rate = (signed_speed * cosf(beta) * tanf(steering)) / agent->length;

        // New velocity
        float new_vx = signed_speed * cosf(heading + beta);
        float new_vy = signed_speed * sinf(heading + beta);

        // Update position
        x = x + (new_vx * env->dt);
        y = y + (new_vy * env->dt);
        heading = heading + yaw_rate * env->dt;

        // Realized accelerations. The classic model commands acceleration directly
        // rather than integrating it, but R_comfort reads a_long/a_lat/jerk_* the
        // same way in both models, so they have to be maintained here too -- left at
        // zero the comfort term is identically zero and alpha_comfort is dead.
        // a_lat is the centripetal term v * yaw_rate, the classic-model counterpart
        // of the jerk model's v^2 * curvature.
        float a_lat_new = signed_speed * yaw_rate;
        agent->jerk_long = (acceleration - agent->a_long) / env->dt;
        agent->jerk_lat = (a_lat_new - agent->a_lat) / env->dt;
        agent->a_long = acceleration;
        agent->a_lat = a_lat_new;

        // Apply updates to the agent's state
        agent->x = x;
        agent->y = y;
        agent->heading = heading;
        agent->heading_x = cosf(heading);
        agent->heading_y = sinf(heading);
        agent->vx = new_vx;
        agent->vy = new_vy;
    } else {
        // JERK dynamics model
        // Extract action components
        float a_long, a_lat;
        if (env->action_type == 1) { // continuous
            float (*action_array_f)[2] = (float (*)[2])env->actions;

            // Asymmetric scaling for longitudinal jerk to match discrete action space
            // Discrete: JERK_LONG = [-15, -4, 0, 4] (more braking than acceleration)
            float a_long_action = action_array_f[action_idx][0]; // [-1, 1]
            if (a_long_action < 0) {
                a_long = a_long_action * (-JERK_LONG[0]); // Negative: [-1, 0] → [-15, 0] (braking)
            } else {
                a_long = a_long_action * JERK_LONG[3]; // Positive: [0, 1] → [0, 4] (acceleration)
            }

            // Symmetric scaling for lateral jerk
            a_lat = action_array_f[action_idx][1] * JERK_LAT[2];
        } else { // discrete
            // Interpret action as a single integer: a = long_idx * num_lat + lat_idx
            int *action_array = (int *)env->actions;
            int num_lat = sizeof(JERK_LAT) / sizeof(JERK_LAT[0]);
            int action_val = action_array[action_idx];
            int a_long_idx = action_val / num_lat;
            int a_lat_idx = action_val % num_lat;
            a_long = JERK_LONG[a_long_idx];
            a_lat = JERK_LAT[a_lat_idx];
        }

        // Calculate new acceleration
        float a_long_new = agent->a_long + a_long * env->dt;
        float a_lat_new = agent->a_lat + a_lat * env->dt;

        // Make it easy to stop with 0 accel
        if (agent->a_long * a_long_new < 0) {
            a_long_new = 0.0f;
        } else {
            a_long_new = clip(a_long_new, -5.0f, 2.5f);
        }

        if (agent->a_lat * a_lat_new < 0) {
            a_lat_new = 0.0f;
        } else {
            a_lat_new = clip(a_lat_new, -4.0f, 4.0f);
        }

        // Calculate new velocity
        float v_dot_heading = agent->vx * agent->heading_x + agent->vy * agent->heading_y;
        float signed_v = copysignf(sqrtf(agent->vx * agent->vx + agent->vy * agent->vy), v_dot_heading);
        float v_target = signed_v + 0.5f * (a_long_new + agent->a_long) * env->dt;
        float v_new = v_target;

        // Make it easy to stop with 0 vel
        if (signed_v * v_target < 0) {
            v_new = 0.0f;
        } else {
            v_new = clip(v_target, -2.0f, 20.0f);
        }

        // A speed the limiter refused is a speed change that did not happen, so the
        // acceleration behind it did not happen either. Carrying a_long forward
        // unchanged made it a claim about the vehicle that nothing else agreed with:
        // held at the -2 m/s reverse floor with a_long = -5, the speed was constant
        // and jerk_long was zero, yet R_comfort charged the |a_long| > 3 indicator on
        // every one of those steps and the ego observation reported hard braking. The
        // state was also absorbing -- the zero-jerk action leaves a_long untouched, so
        // only ~13 consecutive full-throttle steps could unwind it, and both a random
        // and a 1.7B-step policy sat in it for 96% of an episode. Nothing accelerates
        // a vehicle pinned against a limit; the realized value is zero.
        if (v_new != v_target) {
            a_long_new = 0.0f;
        }

        // Calculate new steering angle
        float signed_curvature = a_lat_new / fmaxf(v_new * v_new, 1e-5f);
        signed_curvature = copysignf(fmaxf(fabsf(signed_curvature), 1e-5f), signed_curvature);
        float steering_angle = atanf(signed_curvature * agent->wheelbase);
        float delta_steer = clip(steering_angle - agent->steering_angle, -0.6f * env->dt, 0.6f * env->dt);
        float new_steering_angle = clip(agent->steering_angle + delta_steer, -0.55f, 0.55f);

        // Update curvature and accel to account for limited steering
        signed_curvature = tanf(new_steering_angle) / agent->wheelbase;
        a_lat_new = v_new * v_new * signed_curvature;

        // Calculate resulting movement using bicycle dynamics
        float d = 0.5f * (v_new + signed_v) * env->dt;
        float theta = d * signed_curvature;
        float dx_local, dy_local;

        if (fabsf(signed_curvature) < 1e-5f || fabsf(theta) < 1e-5f) {
            dx_local = d;
            dy_local = 0.0f;
        } else {
            dx_local = sinf(theta) / signed_curvature;
            dy_local = (1.0f - cosf(theta)) / signed_curvature;
        }

        float dx = dx_local * agent->heading_x - dy_local * agent->heading_y;
        float dy = dx_local * agent->heading_y + dy_local * agent->heading_x;

        // Update everything
        agent->x += dx;
        agent->y += dy;
        agent->jerk_long = (a_long_new - agent->a_long) / env->dt;
        agent->jerk_lat = (a_lat_new - agent->a_lat) / env->dt;
        agent->a_long = a_long_new;
        agent->a_lat = a_lat_new;
        agent->heading = normalize_heading(agent->heading + theta);
        agent->heading_x = cosf(agent->heading);
        agent->heading_y = sinf(agent->heading);
        agent->vx = v_new * agent->heading_x;
        agent->vy = v_new * agent->heading_y;
        agent->steering_angle = new_steering_angle;
    }

    return;
}

static inline int is_in_track_to_predicts(Drive *env, int agent_idx) {
    if (env->tracks_to_predict_indices == NULL || env->num_tracks_to_predict == 0) {
        return 0;
    }
    for (int k = 0; k < env->num_tracks_to_predict; k++) {
        if (env->tracks_to_predict_indices[k] == agent_idx) {
            return 1;
        }
    }
    return 0;
}

void c_get_global_agent_state(Drive *env, float *x_out, float *y_out, float *z_out, float *heading_out, int *id_out,
                              float *length_out, float *width_out) {
    for (int i = 0; i < env->active_agent_count; i++) {
        int agent_idx = env->active_agent_indices[i];
        Entity *agent = &env->entities[agent_idx];

        // For WOSAC, we need the original world coordinates, so we add the world means back
        x_out[i] = agent->x + env->world_mean_x;
        y_out[i] = agent->y + env->world_mean_y;
        z_out[i] = agent->z;
        heading_out[i] = agent->heading;
        id_out[i] = agent->id;
        length_out[i] = agent->length;
        width_out[i] = agent->width;
    }
}

void c_get_global_ground_truth_trajectories(Drive *env, float *x_out, float *y_out, float *z_out, float *heading_out,
                                            int *valid_out, int *id_out, bool *is_vehicle_out,
                                            bool *is_track_to_predict_out, char *scenario_id_out) {
    for (int i = 0; i < env->active_agent_count; i++) {
        int agent_idx = env->active_agent_indices[i];
        Entity *agent = &env->entities[agent_idx];
        id_out[i] = agent->id;
        is_vehicle_out[i] = agent->type == VEHICLE;
        is_track_to_predict_out[i] = is_in_track_to_predicts(env, agent_idx);

        // The scenario_id is an array of 16 char
        memcpy(scenario_id_out + (i * 16), env->scenario_id, 16);

        for (int t = env->init_steps; t < agent->array_size; t++) {
            int out_idx = i * (agent->array_size - env->init_steps) + (t - env->init_steps);
            // Add world means back to get original world coordinates
            x_out[out_idx] = agent->traj_x[t] + env->world_mean_x;
            y_out[out_idx] = agent->traj_y[t] + env->world_mean_y;
            z_out[out_idx] = agent->traj_z[t];
            heading_out[out_idx] = agent->traj_heading[t];
            valid_out[out_idx] = agent->traj_valid[t];
        }
    }
}

void c_get_road_edge_counts(Drive *env, int *num_polylines_out, int *total_points_out) {
    int count = 0, points = 0;
    for (int i = env->num_objects; i < env->num_entities; i++) {
        if (env->entities[i].type == ROAD_EDGE) {
            count++;
            points += env->entities[i].array_size;
        }
    }
    *num_polylines_out = count;
    *total_points_out = points;
}

// Scene snapshot for the visualization acceptance check (pufferlib/teddy/drive/viz.py).
//
// Deliberately dumps roads *and* agents from the same env in one call: the simulator
// works in a map-centered frame (set_means subtracts the map centroid), so a viz that
// re-parsed the .bin for road geometry would draw the road in a different frame than
// the agents and the mismatch would look like a spawn bug. Reading both from here
// makes the two frames the same by construction.
#define TEDDY_SNAPSHOT_AGENT_FEATURES (11 + 2 * MAX_WAYPOINTS)

// out = {num_agents, num_road_polylines, num_road_points}
void c_teddy_scene_sizes(Drive *env, int *out) {
    int polys = 0;
    int pts = 0;
    for (int i = env->num_objects; i < env->num_entities; i++) {
        polys++;
        pts += env->entities[i].array_size;
    }
    out[0] = env->active_agent_count;
    out[1] = polys;
    out[2] = pts;
}

// agents:    float[num_agents][TEDDY_SNAPSHOT_AGENT_FEATURES]
//            x, y, heading, length, width, height, type, vx, vy,
//            num_waypoints, current_waypoint, then MAX_WAYPOINTS pairs of (x, y)
// road_xy:   float[num_road_points][2]
// road_meta: int[num_road_polylines][2] = {entity type, point count}
void c_teddy_scene_dump(Drive *env, float *agents, float *road_xy, int *road_meta) {
    for (int i = 0; i < env->active_agent_count; i++) {
        Entity *a = &env->entities[env->active_agent_indices[i]];
        float *row = &agents[(size_t)i * TEDDY_SNAPSHOT_AGENT_FEATURES];
        row[0] = a->x;
        row[1] = a->y;
        row[2] = a->heading;
        row[3] = a->length;
        row[4] = a->width;
        row[5] = a->height;
        row[6] = (float)a->type;
        row[7] = a->vx;
        row[8] = a->vy;
        row[9] = (float)a->num_waypoints;
        row[10] = (float)a->current_waypoint;
        for (int w = 0; w < MAX_WAYPOINTS; w++) {
            // Pad unused slots with the agent position so a careless consumer draws a
            // zero-length segment rather than a line to the map origin.
            int have = w < a->num_waypoints;
            row[11 + 2 * w] = have ? a->waypoints[w][0] : a->x;
            row[12 + 2 * w] = have ? a->waypoints[w][1] : a->y;
        }
    }

    int poly = 0;
    size_t pt = 0;
    for (int i = env->num_objects; i < env->num_entities; i++) {
        Entity *e = &env->entities[i];
        road_meta[poly * 2 + 0] = e->type;
        road_meta[poly * 2 + 1] = e->array_size;
        poly++;
        for (int j = 0; j < e->array_size; j++) {
            road_xy[pt * 2 + 0] = e->traj_x[j];
            road_xy[pt * 2 + 1] = e->traj_y[j];
            pt++;
        }
    }
}

void c_get_road_edge_polylines(Drive *env, float *x_out, float *y_out, int *lengths_out, char *scenario_ids_out) {
    int poly_idx = 0, pt_idx = 0;
    for (int i = env->num_objects; i < env->num_entities; i++) {
        Entity *e = &env->entities[i];
        if (e->type == ROAD_EDGE) {
            lengths_out[poly_idx] = e->array_size;

            char *scenario_id_ptr = scenario_ids_out + poly_idx * 16;
            memcpy(scenario_id_ptr, env->scenario_id, 16);

            for (int j = 0; j < e->array_size; j++) {
                x_out[pt_idx] = e->traj_x[j] + env->world_mean_x;
                y_out[pt_idx] = e->traj_y[j] + env->world_mean_y;
                pt_idx++;
            }
            poly_idx++;
        }
    }
}

// ---------------------------------------------------------------------------
// RenderState: world-frame scene primitives for the perspective rasterizer.
//
// This buffer is privileged and never reaches the policy. The vecenv wrapper
// consumes it on the GPU, produces camera images, and discards it; only the
// rendered images and the ego vector are handed to the network.
// ---------------------------------------------------------------------------

static inline int render_type_enabled(Drive *env, int type) {
    if (type < 0 || type > 31)
        return 0;
    return (env->render_road_types >> type) & 1;
}

// Painted width in meters per road feature. Perspective alone makes near markings
// read thicker than far ones; this only sets the world-space width.
static inline float render_road_width(int type) {
    switch (type) {
    case ROAD_EDGE:
        return 0.25f;
    case CROSSWALK:
        return 0.50f;
    case SPEED_BUMP:
        return 0.40f;
    default:
        return RENDER_ROAD_MARKING_WIDTH;
    }
}

// Number of road segments the enabled types would emit. Python calls this to size
// the road buffer before handing it back through the binding.
int count_render_roads(Drive *env) {
    int n = 0;
    for (int i = 0; i < env->num_entities; i++) {
        Entity *e = &env->entities[i];
        if (e->type < ROAD_LANE || !render_type_enabled(env, e->type))
            continue;
        if (e->array_size > 1)
            n += e->array_size - 1;
    }
    return n;
}

// Static road geometry. Filled once per map, not per step.
void fill_render_roads(Drive *env) {
    if (env->render_roads == NULL)
        return;
    int n = 0;
    int cap = env->render_max_roads;
    for (int i = 0; i < env->num_entities && n < cap; i++) {
        Entity *e = &env->entities[i];
        if (e->type < ROAD_LANE || !render_type_enabled(env, e->type))
            continue;
        float width = render_road_width(e->type);
        for (int j = 0; j + 1 < e->array_size && n < cap; j++) {
            float *out = &env->render_roads[n * RENDER_ROAD_FEATURES];
            out[0] = e->traj_x[j];
            out[1] = e->traj_y[j];
            out[2] = e->traj_x[j + 1];
            out[3] = e->traj_y[j + 1];
            out[4] = width;
            out[5] = (float)e->type;
            n++;
        }
    }
    if (env->render_counts != NULL)
        env->render_counts[1] = n;
}

// Dynamic scene: every drawable agent, plus the pose of each controlled ego.
// Called every step from compute_observations.
void fill_render_state(Drive *env) {
    if (env->render_agents == NULL || env->render_egos == NULL)
        return;

    int n = 0;
    int cap = env->render_max_agents;
    // Entity index behind each primitive, so each ego can find its own box below.
    int prim_entity[MAX_AGENTS];
    // Same iteration and validity rules as the partner observations above, so the
    // vectorized and perspective modalities see the same set of agents.
    for (int j = 0; j < env->num_actors && n < cap; j++) {
        int index = -1;
        if (j < env->active_agent_count) {
            index = env->active_agent_indices[j];
        } else if (j < env->num_actors && env->static_agent_count > 0) {
            index = env->static_agent_indices[j - env->active_agent_count];
        }
        if (index == -1)
            continue;
        Entity *e = &env->entities[index];
        if (e->type > CYCLIST || e->type == NONE)
            continue;
        if (e->removed)
            continue;

        float *out = &env->render_agents[n * RENDER_AGENT_FEATURES];
        out[0] = e->x;
        out[1] = e->y;
        out[2] = e->heading_x;
        out[3] = e->heading_y;
        out[4] = e->length;
        out[5] = e->width;
        out[6] = e->height;
        out[7] = (float)e->type;
        prim_entity[n] = index;
        n++;
    }
    if (env->render_counts != NULL)
        env->render_counts[0] = n;

    for (int i = 0; i < env->active_agent_count; i++) {
        int index = env->active_agent_indices[i];
        Entity *e = &env->entities[index];
        float *out = &env->render_egos[i * RENDER_EGO_FEATURES];
        out[0] = e->x;
        out[1] = e->y;
        out[2] = e->heading_x;
        out[3] = e->heading_y;
        // -1 when this ego has no primitive this step (removed or respawning),
        // in which case there is nothing to skip.
        out[4] = -1.0f;
        for (int k = 0; k < n; k++) {
            if (prim_entity[k] == index) {
                out[4] = (float)k;
                break;
            }
        }
    }
}

void compute_observations(Drive *env) {
    int ego_dim = drive_ego_dim(env);
    int max_obs = drive_obs_size(env);
    memset(env->observations, 0, max_obs * env->active_agent_count * sizeof(float));
    float (*observations)[max_obs] = (float (*)[max_obs])env->observations;
    for (int i = 0; i < env->active_agent_count; i++) {
        float *obs = &observations[i][0];
        Entity *ego_entity = &env->entities[env->active_agent_indices[i]];
        if (ego_entity->type > 3)
            break;

        float cos_heading = ego_entity->heading_x;
        float sin_heading = ego_entity->heading_y;
        float speed_magnitude = sqrtf(ego_entity->vx * ego_entity->vx + ego_entity->vy * ego_entity->vy);
        float v_dot_heading = ego_entity->vx * ego_entity->heading_x + ego_entity->vy * ego_entity->heading_y;
        float signed_speed = copysignf(speed_magnitude, v_dot_heading);

        // Set goal distances
        float goal_x = ego_entity->goal_position_x - ego_entity->x;
        float goal_y = ego_entity->goal_position_y - ego_entity->y;

        // Rotate to ego vehicle's frame
        float rel_goal_x = goal_x * cos_heading + goal_y * sin_heading;
        float rel_goal_y = -goal_x * sin_heading + goal_y * cos_heading;

        obs[0] = rel_goal_x * 0.005f;
        obs[1] = rel_goal_y * 0.005f;
        obs[2] = signed_speed / MAX_SPEED;
        obs[3] = ego_entity->width / MAX_VEH_WIDTH;
        obs[4] = ego_entity->length / MAX_VEH_LEN;
        obs[5] = (ego_entity->collision_state > 0) ? 1.0f : 0.0f;

        if (env->dynamics_model == JERK) {
            obs[6] = ego_entity->steering_angle / M_PI;
            // Asymmetric normalization for a_long to match action space
            obs[7] =
                (ego_entity->a_long < 0) ? ego_entity->a_long / (-JERK_LONG[0]) : ego_entity->a_long / JERK_LONG[3];
            obs[8] = ego_entity->a_lat / JERK_LAT[2];
            obs[9] = (ego_entity->respawn_timestep == env->timestep) ? 1 : 0;
            // Add normalized entity type (VEHICLE=1, PEDESTRIAN=2, CYCLIST=3)
            obs[10] = ego_entity->type / 3.0f;
        } else {
            obs[6] = (ego_entity->respawn_timestep == env->timestep) ? 1 : 0;
            obs[7] = ego_entity->type / 3.0f;
        }

        // In perspective mode the ego vector is the whole observation: the scene
        // reaches the policy only as rendered pixels, never as entity sets.
        if (env->obs_mode == OBS_MODE_RENDER_STATE)
            continue;

        // Relative Pos of other cars, nearest first.
        //
        // The slot budget is smaller than the traffic a scene can hold, so *which*
        // partners get in matters. Scanning in index order and taking the first that
        // fit hands the policy an arbitrary subset the moment the budget binds --
        // and the ones it drops are as likely to be the car it is about to hit as
        // one behind it. Ranking by distance makes the truncation the only sensible
        // one, and makes the observation mean the same thing at every traffic
        // density.
        int obs_idx = ego_dim;
        int cars_seen = 0;
        // Candidates within the observation radius, gathered before any is written:
        // the ranking needs to see all of them.
        float cand_dist[MAX_AGENTS];
        int cand_idx[MAX_AGENTS];
        int num_cand = 0;
        float radius_sq = env->partner_obs_radius * env->partner_obs_radius;
        for (int j = 0; j < env->num_actors; j++) {
            int index = -1;
            if (j < env->active_agent_count) {
                index = env->active_agent_indices[j];
            } else if (j < env->num_actors && env->static_agent_count > 0) {
                index = env->static_agent_indices[j - env->active_agent_count];
            }
            if (index == -1)
                continue;
            if (env->entities[index].type > 3)
                break;
            if (index == env->active_agent_indices[i])
                continue; // Skip self
            Entity *other_entity = &env->entities[index];
            float dx = other_entity->x - ego_entity->x;
            float dy = other_entity->y - ego_entity->y;
            float dist = (dx * dx + dy * dy);
            if (dist > radius_sq)
                continue;
            cand_dist[num_cand] = dist;
            cand_idx[num_cand] = index;
            num_cand++;
        }

        // Partial selection sort: only the MAX_PARTNER_OBS nearest need ordering, and
        // after the radius filter num_cand is small (~14 at the default 50 m), so this
        // costs less than a full sort of the candidate list.
        int n_take = (num_cand < MAX_PARTNER_OBS) ? num_cand : MAX_PARTNER_OBS;
        for (int k = 0; k < n_take; k++) {
            int best = k;
            for (int m = k + 1; m < num_cand; m++) {
                if (cand_dist[m] < cand_dist[best])
                    best = m;
            }
            if (best != k) {
                float td = cand_dist[k];
                cand_dist[k] = cand_dist[best];
                cand_dist[best] = td;
                int ti = cand_idx[k];
                cand_idx[k] = cand_idx[best];
                cand_idx[best] = ti;
            }

            Entity *other_entity = &env->entities[cand_idx[k]];
            float dx = other_entity->x - ego_entity->x;
            float dy = other_entity->y - ego_entity->y;
            // Rotate to ego vehicle's frame
            float rel_x = dx * cos_heading + dy * sin_heading;
            float rel_y = -dx * sin_heading + dy * cos_heading;
            obs[obs_idx] = rel_x * 0.02f;
            obs[obs_idx + 1] = rel_y * 0.02f;
            obs[obs_idx + 2] = other_entity->width / MAX_VEH_WIDTH;
            obs[obs_idx + 3] = other_entity->length / MAX_VEH_LEN;
            // relative heading
            float rel_heading_x =
                other_entity->heading_x * ego_entity->heading_x +
                other_entity->heading_y * ego_entity->heading_y; // cos(a-b) = cos(a)cos(b) + sin(a)sin(b)
            float rel_heading_y =
                other_entity->heading_y * ego_entity->heading_x -
                other_entity->heading_x * ego_entity->heading_y; // sin(a-b) = sin(a)cos(b) - cos(a)sin(b)

            obs[obs_idx + 4] = rel_heading_x;
            obs[obs_idx + 5] = rel_heading_y;

            // relative speed
            float other_speed_magnitude =
                sqrtf(other_entity->vx * other_entity->vx + other_entity->vy * other_entity->vy);
            float other_v_dot_heading =
                other_entity->vx * other_entity->heading_x + other_entity->vy * other_entity->heading_y;
            float other_signed_speed = copysignf(other_speed_magnitude, other_v_dot_heading);
            obs[obs_idx + 6] = other_signed_speed / MAX_SPEED;
            cars_seen++;
            obs_idx += 7; // Move to next observation slot
        }
        int remaining_partner_obs = (MAX_PARTNER_OBS - cars_seen) * 7;
        memset(&obs[obs_idx], 0, remaining_partner_obs * sizeof(float));
        obs_idx += remaining_partner_obs;
        // map observations
        GridMapEntity entity_list[MAX_ENTITIES_PER_CELL * 25];
        int grid_idx = getGridIndex(env, ego_entity->x, ego_entity->y);

        int list_size = get_neighbor_cache_entities(env, grid_idx, entity_list, MAX_ROAD_SEGMENT_OBSERVATIONS);

        for (int k = 0; k < list_size; k++) {
            int entity_idx = entity_list[k].entity_idx;
            int geometry_idx = entity_list[k].geometry_idx;

            // Validate entity_idx before accessing
            if (entity_idx < 0 || entity_idx >= env->num_entities) {
                printf("ERROR: Invalid entity_idx %d (max: %d)\n", entity_idx, env->num_entities - 1);
                continue;
            }

            Entity *entity = &env->entities[entity_idx];

            // Validate geometry_idx before accessing
            if (geometry_idx < 0 || geometry_idx >= entity->array_size) {
                printf("ERROR: Invalid geometry_idx %d for entity %d (max: %d)\n", geometry_idx, entity_idx,
                       entity->array_size - 1);
                continue;
            }
            float start_x = entity->traj_x[geometry_idx];
            float start_y = entity->traj_y[geometry_idx];
            float end_x = entity->traj_x[geometry_idx + 1];
            float end_y = entity->traj_y[geometry_idx + 1];
            float mid_x = (start_x + end_x) / 2.0f;
            float mid_y = (start_y + end_y) / 2.0f;
            float rel_x = mid_x - ego_entity->x;
            float rel_y = mid_y - ego_entity->y;
            float x_obs = rel_x * cos_heading + rel_y * sin_heading;
            float y_obs = -rel_x * sin_heading + rel_y * cos_heading;
            float length = relative_distance_2d(mid_x, mid_y, end_x, end_y);
            float width = 0.1;
            // Calculate angle from ego to midpoint (vector from ego to midpoint)
            float dx = end_x - mid_x;
            float dy = end_y - mid_y;
            float dx_norm = dx;
            float dy_norm = dy;
            float hypot = sqrtf(dx * dx + dy * dy);
            if (hypot > 0) {
                dx_norm /= hypot;
                dy_norm /= hypot;
            }
            // Compute sin and cos of relative angle directly without atan2f
            float cos_angle = dx_norm * cos_heading + dy_norm * sin_heading;
            float sin_angle = -dx_norm * sin_heading + dy_norm * cos_heading;
            obs[obs_idx] = x_obs * 0.02f;
            obs[obs_idx + 1] = y_obs * 0.02f;
            obs[obs_idx + 2] = length / MAX_ROAD_SEGMENT_LENGTH;
            obs[obs_idx + 3] = width / MAX_ROAD_SCALE;
            obs[obs_idx + 4] = cos_angle;
            obs[obs_idx + 5] = sin_angle;
            obs[obs_idx + 6] = entity->type - 4.0f;
            obs_idx += 7;
        }
        int remaining_obs = (MAX_ROAD_SEGMENT_OBSERVATIONS - list_size) * 7;
        // Set the entire block to 0 at once
        memset(&obs[obs_idx], 0, remaining_obs * sizeof(float));
    }

    if (env->obs_mode == OBS_MODE_RENDER_STATE)
        fill_render_state(env);
}

void sample_new_goal(Drive *env, int agent_idx) {
    // Samples a new goal position based on the existing road lane points
    Entity *agent = &env->entities[agent_idx];
    float best_x = agent->x;
    float best_y = agent->y;
    float best_distance_error = 1e30f;

    // Sample points from all road lanes
    for (int i = env->num_objects; i < env->num_entities; i++) {
        if (env->entities[i].type != ROAD_LANE)
            continue;

        Entity *lane = &env->entities[i];

        // Check every point in the lane
        for (int j = 0; j < lane->array_size; j++) {
            float point_x = lane->traj_x[j];
            float point_y = lane->traj_y[j];

            // Calculate vector from agent to point
            float to_point_x = point_x - agent->x;
            float to_point_y = point_y - agent->y;

            // Check if point is ahead of agent
            float dot = to_point_x * agent->heading_x + to_point_y * agent->heading_y;
            if (dot <= 0.0f)
                continue;

            // Calculate distance to point
            float distance = sqrtf(to_point_x * to_point_x + to_point_y * to_point_y);

            // Find point closest to target distance
            float distance_error = fabsf(distance - env->goal_target_distance);
            if (distance_error < best_distance_error) {
                best_distance_error = distance_error;
                best_x = point_x;
                best_y = point_y;
            }
        }
    }

    // If no valid goal found, use another agent's initial goal
    if (best_distance_error >= 1e30f && env->active_agent_count > 1) {
        int other_idx = env->active_agent_indices[(agent_idx + 1) % env->active_agent_count];
        best_x = env->entities[other_idx].init_goal_x;
        best_y = env->entities[other_idx].init_goal_y;
    }

    agent->goal_position_x = best_x;
    agent->goal_position_y = best_y;
    agent->goals_sampled_this_episode += 1;
}

void c_reset(Drive *env) {
    env->timestep = env->init_steps;
    teddy_spawn_all(env);
    for (int x = 0; x < env->active_agent_count; x++) {
        env->logs[x] = (Log){0};
        int agent_idx = env->active_agent_indices[x];
        env->entities[agent_idx].respawn_timestep = -1;
        env->entities[agent_idx].respawn_count = 0;
        env->entities[agent_idx].collided_before_goal = 0;
        env->entities[agent_idx].goals_reached_this_episode = 0.0f;
        // teddy_spawn_all -> teddy_sample_route already counted this episode's first
        // goal, so unlike ocean there is no data-file goal to seed here.
        env->entities[agent_idx].goals_sampled_this_episode = 1.0f;
        env->entities[agent_idx].current_goal_reached = 0;
        env->entities[agent_idx].metrics_array[COLLISION_IDX] = 0.0f;
        env->entities[agent_idx].metrics_array[OFFROAD_IDX] = 0.0f;
        env->entities[agent_idx].metrics_array[REACHED_GOAL_IDX] = 0.0f;
        env->entities[agent_idx].metrics_array[LANE_ALIGNED_IDX] = 0.0f;
        env->entities[agent_idx].stopped = 0;
        env->entities[agent_idx].removed = 0;

        if (env->goal_behavior == GOAL_GENERATE_NEW) {
            env->entities[agent_idx].goal_position_x = env->entities[agent_idx].init_goal_x;
            env->entities[agent_idx].goal_position_y = env->entities[agent_idx].init_goal_y;
        }

        compute_agent_metrics(env, agent_idx);
    }
    compute_observations(env);
}

void respawn_agent(Drive *env, int agent_idx) {
    // Re-place from the lane graph with a fresh embodiment and route. Avoiding every
    // other agent (num_objects, not just the ones placed so far) matters here: the
    // scene is already populated and moving.
    teddy_place_agent(env, agent_idx, env->num_objects);

    // Stamped, not latched: `teddy_place_agent` already cleared this to -1, and every
    // consumer other than the ego flag treats the recycled agent as an ordinary one.
    // Gigaflow keeps agent density constant by recycling, so an agent that is invisible
    // to the cameras and inert in collision after its first goal would drain the very
    // traffic the self-play is meant to produce.
    env->entities[agent_idx].respawn_timestep = env->timestep;
    env->entities[agent_idx].collided_before_goal = 0;
    env->entities[agent_idx].stopped = 0;
    env->entities[agent_idx].removed = 0;
    env->entities[agent_idx].a_long = 0.0f;
    env->entities[agent_idx].a_lat = 0.0f;
    env->entities[agent_idx].jerk_long = 0.0f;
    env->entities[agent_idx].jerk_lat = 0.0f;
    env->entities[agent_idx].steering_angle = 0.0f;
}

void c_step(Drive *env) {
    memset(env->rewards, 0, env->active_agent_count * sizeof(float));
    memset(env->terminals, 0, env->active_agent_count * sizeof(unsigned char));
    memset(env->truncations, 0, env->active_agent_count * sizeof(unsigned char));
    if (env->debug_terms) {
        int rows = (env->active_agent_count < env->debug_max_rows) ? env->active_agent_count : env->debug_max_rows;
        memset(env->debug_terms, 0, (size_t)rows * TEDDY_DEBUG_FEATURES * sizeof(float));
    }
    env->timestep++;

    // Move static experts
    for (int i = 0; i < env->expert_static_agent_count; i++) {
        int expert_idx = env->expert_static_agent_indices[i];
        if (env->entities[expert_idx].x == INVALID_POSITION)
            continue;
        move_expert(env, env->actions, expert_idx);
    }
    // Process actions for all active agents
    for (int i = 0; i < env->active_agent_count; i++) {
        env->logs[i].score = 0.0f;
        env->logs[i].episode_length += 1;
        int agent_idx = env->active_agent_indices[i];
        env->entities[agent_idx].collision_state = 0;
        float prev_vx = env->entities[agent_idx].vx;
        float prev_vy = env->entities[agent_idx].vy;

        move_dynamics(env, i, agent_idx);

        // Tiny jerk penalty for smoothness
        if (env->dynamics_model == CLASSIC) {
            float delta_vx = env->entities[agent_idx].vx - prev_vx;
            float delta_vy = env->entities[agent_idx].vy - prev_vy;
            float jerk_penalty = -0.0002f * sqrtf(delta_vx * delta_vx + delta_vy * delta_vy) / env->dt;
            env->rewards[i] += jerk_penalty;
            env->logs[i].episode_return += jerk_penalty;
            if (env->debug_terms && i < env->debug_max_rows)
                env->debug_terms[(size_t)i * TEDDY_DEBUG_FEATURES + TEDDY_DBG_R_JERK] = jerk_penalty;
        }
    }

    // Compute rewards
    for (int i = 0; i < env->active_agent_count; i++) {
        int agent_idx = env->active_agent_indices[i];
        env->entities[agent_idx].collision_state = 0;

        compute_agent_metrics(env, agent_idx);
        int collision_state = env->entities[agent_idx].collision_state;

        // ocean's reward, unchanged in form: a fixed penalty for hitting a car, a
        // fixed penalty for leaving the road, and a fixed payment for reaching a
        // goal. Every weight is a config value shared by the whole fleet -- there is
        // no per-agent conditioning here, which is the whole point of `teddy`. The
        // only adaptation is the goal test, which walks the waypoint route `teddy`
        // inherits from the Gigaflow-style initialization instead of testing a single
        // dataset goal.
        Entity *a = &env->entities[agent_idx];
        float speed = sqrtf(a->vx * a->vx + a->vy * a->vy);
        float signed_v = copysignf(speed, a->vx * a->heading_x + a->vy * a->heading_y);
        // One named variable per term rather than a single running sum, so the trace
        // below can attribute the step's reward.
        float r_collision = 0.0f, r_offroad = 0.0f, r_goal = 0.0f;

        if (collision_state == VEHICLE_COLLISION) {
            r_collision = env->reward_vehicle_collision;
            env->logs[i].collision_rate = 1.0f;
            env->logs[i].collisions_per_agent += 1.0f;
            a->collided_before_goal = 1;
        } else if (collision_state == OFFROAD) {
            r_offroad = env->reward_offroad_collision;
            env->logs[i].offroad_rate = 1.0f;
            env->logs[i].offroad_per_agent += 1.0f;
            a->collided_before_goal = 1;
        }

        float distance_to_goal = relative_distance_2d(a->x, a->y, a->goal_position_x, a->goal_position_y);
        bool within_distance = distance_to_goal < env->goal_radius;
        bool within_speed = speed <= env->goal_speed;
        // Only the final goal has to be reached at low speed. Intermediate waypoints
        // are drive-through: they mark a route, and stopping on each one would make a
        // three-waypoint route a different task from a one-waypoint route.
        int is_final = (a->current_waypoint >= a->num_waypoints - 1);

        if (within_distance && !a->current_goal_reached && (!is_final || within_speed)) {
            if (!is_final) {
                r_goal = env->reward_goal;
                a->current_waypoint++;
                a->goal_position_x = a->waypoints[a->current_waypoint][0];
                a->goal_position_y = a->waypoints[a->current_waypoint][1];
                a->goals_sampled_this_episode += 1.0f;
            } else if (env->goal_behavior == GOAL_RESPAWN && a->respawn_timestep != -1) {
                // Not the agent's first goal of the episode: `respawn_timestep` is -1
                // until teddy_place_agent recycles it, so this pays the discounted
                // rate for every goal after the first.
                r_goal = env->reward_goal_post_respawn;
                a->current_goal_reached = 1;
                a->metrics_array[REACHED_GOAL_IDX] = 1.0f;
                a->goals_reached_this_episode += 1.0f;
            } else if (env->goal_behavior == GOAL_GENERATE_NEW) {
                r_goal = env->reward_goal;
                sample_new_goal(env, agent_idx);
                a->current_goal_reached = 0;
                a->metrics_array[REACHED_GOAL_IDX] = 1.0f;
                a->goals_reached_this_episode += 1.0f;
            } else if (env->goal_behavior == GOAL_REMOVE) {
                // One shot: reward_goal_post_respawn only means anything once
                // GOAL_RESPAWN gives an agent a second life to pay a lower rate on,
                // and this agent never gets one. Actual removal happens in the
                // respawn/stop pass below, once every agent's reward this step is in.
                r_goal = env->reward_goal;
                a->metrics_array[REACHED_GOAL_IDX] = 1.0f;
                a->goals_reached_this_episode += 1.0f;
            } else {
                // GOAL_STOP, and also the first goal of an agent's life under
                // GOAL_RESPAWN (respawn_timestep is still -1 then).
                //
                // ONE DELIBERATE DEVIATION FROM OCEAN. ocean writes
                //     env->rewards[i]            = env->reward_goal;
                //     env->logs[i].episode_return = env->reward_goal;
                // -- assignment, not accumulation. Under ocean's 91-step episodes,
                // where an agent reaches its single dataset goal at most once and the
                // episode ends there, that is nearly invisible. Here it is not: with
                // GOAL_RESPAWN every agent passes through this branch on its first
                // goal and then respawns, many times over a 1280-step episode, so
                // assigning would (a) erase a collision charged on the very same step,
                // paying full price for a goal reached by driving through someone, and
                // (b) reset episode_return -- the headline training metric -- to 1.0
                // each time. Accumulating instead.
                r_goal = env->reward_goal;
                a->stopped = 1;
                a->vx = a->vy = 0.0f;
                a->metrics_array[REACHED_GOAL_IDX] = 1.0f;
                a->goals_reached_this_episode += 1.0f;
            }
            env->logs[i].speed_at_goal = speed;
        }

        float r = 0.0f;
        r += r_collision;
        r += r_offroad;
        r += r_goal;

        env->rewards[i] += r;
        env->logs[i].episode_return += r;

        if (env->debug_terms && i < env->debug_max_rows) {
            float *row = &env->debug_terms[(size_t)i * TEDDY_DEBUG_FEATURES];
            // rewards[i], not r: under classic dynamics the jerk penalty was already
            // added in the movement loop, and the trace has to add up to what the
            // trainer sees.
            row[TEDDY_DBG_REWARD] = env->rewards[i];
            row[TEDDY_DBG_R_COLLISION] = r_collision;
            row[TEDDY_DBG_R_OFFROAD] = r_offroad;
            row[TEDDY_DBG_R_GOAL] = r_goal;
            row[TEDDY_DBG_X] = a->x;
            row[TEDDY_DBG_Y] = a->y;
            row[TEDDY_DBG_HEADING] = a->heading;
            row[TEDDY_DBG_SPEED] = speed;
            row[TEDDY_DBG_SIGNED_V] = signed_v;
            row[TEDDY_DBG_A_LONG] = a->a_long;
            row[TEDDY_DBG_A_LAT] = a->a_lat;
            row[TEDDY_DBG_JERK_LONG] = a->jerk_long;
            row[TEDDY_DBG_JERK_LAT] = a->jerk_lat;
            row[TEDDY_DBG_STEERING] = a->steering_angle;
            row[TEDDY_DBG_COLLISION_STATE] = (float)collision_state;
            row[TEDDY_DBG_LANE_VALID] = (float)a->lane_valid;
            row[TEDDY_DBG_LANE_HEADING_ERR] = a->lane_heading_error;
            row[TEDDY_DBG_LANE_LATERAL_OFFSET] = a->lane_lateral_offset;
            row[TEDDY_DBG_LANE_ALIGNED] = a->metrics_array[LANE_ALIGNED_IDX];
            row[TEDDY_DBG_DIST_TO_GOAL] = distance_to_goal;
            row[TEDDY_DBG_CURRENT_WAYPOINT] = (float)a->current_waypoint;
            row[TEDDY_DBG_NUM_WAYPOINTS] = (float)a->num_waypoints;
            row[TEDDY_DBG_GOALS_REACHED] = a->goals_reached_this_episode;
            row[TEDDY_DBG_REACHED_FINAL] = a->metrics_array[REACHED_GOAL_IDX];
            row[TEDDY_DBG_RESPAWN_COUNT] = (float)a->respawn_count;
            // Discrete: the joint action index. Continuous: the longitudinal channel
            // only, since the row has one slot and steering is recoverable from
            // steering_angle above.
            row[TEDDY_DBG_ACTION] =
                (env->action_type == 1) ? ((float *)env->actions)[2 * i] : (float)((int *)env->actions)[i];
            row[TEDDY_DBG_TIMESTEP] = (float)env->timestep;
        }

        // Accumulated, not assigned: this is the fraction of the episode spent
        // aligned, normalized by episode_length in add_log. Assigning here made it a
        // snapshot of the final step, which over a 1280-step episode says almost
        // nothing about how the agent drove.
        env->logs[i].lane_alignment_rate += env->entities[agent_idx].metrics_array[LANE_ALIGNED_IDX];
    }

    if (env->goal_behavior == GOAL_RESPAWN) {
        for (int i = 0; i < env->active_agent_count; i++) {
            int agent_idx = env->active_agent_indices[i];
            int reached_goal = env->entities[agent_idx].metrics_array[REACHED_GOAL_IDX];
            if (reached_goal) {
                env->terminals[i] = 1;
                respawn_agent(env, agent_idx);
                env->entities[agent_idx].respawn_count++;
            }
        }
    } else if (env->goal_behavior == GOAL_STOP) {
        for (int i = 0; i < env->active_agent_count; i++) {
            int agent_idx = env->active_agent_indices[i];
            int reached_goal = env->entities[agent_idx].metrics_array[REACHED_GOAL_IDX];
            if (reached_goal) {
                env->entities[agent_idx].stopped = 1;
                env->entities[agent_idx].vx = env->entities[agent_idx].vy = 0.0f;
            }
        }
    } else if (env->goal_behavior == GOAL_REMOVE) {
        // Unlike GOAL_STOP, this ends the agent's life: terminal, same as GOAL_RESPAWN,
        // so the value function does not bootstrap across the vanish.
        for (int i = 0; i < env->active_agent_count; i++) {
            int agent_idx = env->active_agent_indices[i];
            int reached_goal = env->entities[agent_idx].metrics_array[REACHED_GOAL_IDX];
            if (reached_goal) {
                env->terminals[i] = 1;
                env->entities[agent_idx].removed = 1;
                env->entities[agent_idx].x = env->entities[agent_idx].y = -10000.0f;
            }
        }
    }

    // Episode boundary after this step: treat time-limit and early-termination as truncation.
    // `timestep` is incremented at the top of c_step, so after the k-th step it
    // equals init_steps + k. Truncating at `timestep + 1 >= episode_length` ran one
    // step short (1279 of a configured 1280); compare `timestep` itself.
    int originals_remaining = 0;
    for (int i = 0; i < env->active_agent_count; i++) {
        int agent_idx = env->active_agent_indices[i];
        if (env->entities[agent_idx].respawn_count == 0) {
            originals_remaining = 1;
            break;
        }
    }
    int reached_time_limit = env->timestep >= env->episode_length;
    int reached_early_termination = (!originals_remaining && env->termination_mode == 1);
    if (reached_time_limit || reached_early_termination) {
        for (int i = 0; i < env->active_agent_count; i++) {
            env->truncations[i] = 1;
        }
        add_log(env);
        c_reset(env);
        return;
    }

    compute_observations(env);
}

typedef struct Client Client;

// Geometry of the camera strip drawn under the simulator view: panels side by
// side, in the order Python packed them, which is left to right across the rig.
//
// A pure function of the window width and the rig because make_client has to
// know the strip's height before InitWindow, in order to make the window that
// much taller and keep the simulator view its full size.
typedef struct {
    int panel_w;
    int panel_h;
    int label_h;
    int margin;
    int gap;
    int strip_h; // total height reserved along the bottom edge
    int x0;      // left edge of the first panel
} CameraStrip;

static CameraStrip camera_strip_layout(int window_w, int n, int cam_w, int cam_h) {
    CameraStrip s = {0};
    if (n <= 0 || cam_w <= 0 || cam_h <= 0)
        return s;

    s.margin = 12;
    s.gap = 10;
    s.label_h = 18;

    int avail = window_w - 2 * s.margin - (n - 1) * s.gap;
    s.panel_w = avail / n;
    // Cap the panel width so a one- or two-camera rig does not swallow the
    // window. The views are 96 px wide, so beyond this it is mostly upscaling.
    if (s.panel_w > 360)
        s.panel_w = 360;
    if (s.panel_w < 16)
        s.panel_w = 16;
    s.panel_h = (int)((float)s.panel_w * (float)cam_h / (float)cam_w + 0.5f);

    s.strip_h = 2 * s.margin + s.label_h + s.panel_h;
    // The strip is added to the recorded frame height and h264 rejects odd
    // dimensions, so keep it even.
    s.strip_h += s.strip_h & 1;

    int total_w = n * s.panel_w + (n - 1) * s.gap;
    s.x0 = (window_w - total_w) / 2;
    if (s.x0 < s.margin)
        s.x0 = s.margin;
    return s;
}

struct Client {
    float width;
    float height;
    Texture2D puffers;
    Vector3 camera_target;
    float camera_zoom;
    Camera3D camera;
    Model cars[6];
    Model cyclist;
    Model pedestrian;
    ModelAnimation *cycle_anim;
    int car_assignments[MAX_AGENTS];
    Vector3 default_camera_position;
    Vector3 default_camera_target;
    int recorder_pipefd[2];
    pid_t recorder_pid;
    // Lazily created texture holding the stacked camera views.
    Texture2D camera_tex;
    int camera_tex_w;
    int camera_tex_h;
    // Band reserved along the bottom edge for the camera panels, and the height
    // left over above it for the simulator view. `cam_strip_h` is 0 when no rig
    // is bound, and then `view_h` is the full window height.
    int cam_strip_h;
    int view_h;
    // Off-screen target the simulator view is drawn into when the strip is
    // present, so the panels sit beside the view rather than on top of it.
    RenderTexture2D sim_view;
    pid_t xvfb_pid;
    int xvfb_display_num;
};

Client *make_client(Drive *env) {

    Client *client = (Client *)calloc(1, sizeof(Client));

    if (env->render_mode == RENDER_HEADLESS && getenv("DISPLAY") == NULL) {

        // Kill any existing Xvfb first
        system("pkill -9 Xvfb");
        usleep(200000);
        unlink("/tmp/.X99-lock");
        unlink("/tmp/.X11-unix/X99");

        // Hardcode to single display because we only run this in one process at once
        client->xvfb_display_num = 99;

        // Clean up stale lock if process is dead
        FILE *f = fopen("/tmp/.X99-lock", "r");
        if (f) {
            pid_t pid = -1;
            fscanf(f, "%d", &pid);
            fclose(f);
            if (pid > 0 && kill(pid, 0) != 0)
                unlink("/tmp/.X99-lock");
        }

        client->xvfb_pid = fork();
        if (client->xvfb_pid == 0) {
            close(STDOUT_FILENO);
            close(STDERR_FILENO);
            execlp("Xvfb", "Xvfb", ":99", "-screen", "0", "1280x720x24", "+extension", "GLX", "-ac", "-noreset", NULL);
            _exit(1);
        }

        setenv("DISPLAY", ":99", 1);
        // Xvfb starts asynchronously after fork(), so we poll until it creates its
        // lock file (max 2s) then wait an extra 200ms for GLX to finish initializing.
        // Without this, raylib's InitWindow() would try to connect before Xvfb is ready.
        for (int i = 0; i < 20 && access("/tmp/.X99-lock", F_OK) != 0; i++)
            usleep(100000);
        usleep(200000);
    }

    if (env->render_mode == RENDER_WINDOW) {
        client->width = 1280;
        client->height = 704;
        SetConfigFlags(FLAG_MSAA_4X_HINT);
        SetTargetFPS(30);

        // Set up camera for interactive window
        Vector3 target_pos = {0, 0, 1}; // Y is up, Z is depth

        client->default_camera_position = (Vector3){
            0,      // Same X as target
            120.0f, // 20 units above target
            175.0f  // 20 units behind target
        };
        client->default_camera_target = target_pos;
        client->camera.position = client->default_camera_position;
        client->camera.target = client->default_camera_target;
        client->camera.up = (Vector3){0.0f, -1.0f, 0.0f}; // Y is up
        client->camera.fovy = 45.0f;
        client->camera.projection = CAMERA_PERSPECTIVE;

    } else { // Headless rendering
        SetConfigFlags(FLAG_WINDOW_HIDDEN);
        SetTargetFPS(6000);

        float map_width = env->grid_map->bottom_right_x - env->grid_map->top_left_x;
        float map_height = env->grid_map->top_left_y - env->grid_map->bottom_right_y;
        float scale = 6.0f; // Controls the resolution of the output video
        int img_width = (int)roundf(map_width * scale / 2.0f) * 2;
        int img_height = (int)roundf(map_height * scale / 2.0f) * 2;

        client->width = img_width;
        client->height = img_height;
    }

    // Reserve a band along the bottom for the policy's camera views. The rig is
    // bound before the first render call, so its size is known here, and growing
    // the window rather than carving into it leaves the simulator view exactly the
    // size it had without cameras.
    CameraStrip strip = camera_strip_layout((int)client->width, env->render_camera_count,
                                            env->render_camera_width, env->render_camera_height);
    client->cam_strip_h = strip.strip_h;
    client->height += (float)client->cam_strip_h;
    client->view_h = (int)client->height - client->cam_strip_h;

    SetTraceLogLevel(LOG_WARNING); // Only show warnings and errors
    InitWindow(client->width, client->height, "PufferDrive");

    // Needs the GL context, so it cannot be folded into the sizing above.
    if (client->cam_strip_h > 0)
        client->sim_view = LoadRenderTexture((int)client->width, client->view_h);

    // Load assets
    client->cars[0] = LoadModel("resources/drive/RedCar.glb");
    client->cars[1] = LoadModel("resources/drive/WhiteCar.glb");
    client->cars[2] = LoadModel("resources/drive/BlueCar.glb");
    client->cars[3] = LoadModel("resources/drive/YellowCar.glb");
    client->cars[4] = LoadModel("resources/drive/GreenCar.glb");
    client->cars[5] = LoadModel("resources/drive/GreyCar.glb");
    client->cyclist = LoadModel("resources/drive/cyclist.glb");
    client->pedestrian = LoadModel("resources/drive/pedestrian.glb");
    int animCountCyc = 0;
    client->cycle_anim = LoadModelAnimations("resources/drive/cyclist.glb", &animCountCyc);
    for (int i = 0; i < MAX_AGENTS; i++) {
        client->car_assignments[i] = (rand() % 4) + 1;
    }

    // Set up ffmpeg process for recording
    if (env->render_mode == RENDER_HEADLESS) {
        if (pipe(client->recorder_pipefd) == -1) {
            fprintf(stderr, "Failed to create pipe\n");
            free(client);
            return NULL;
        }

        char size_str[64];
        snprintf(size_str, sizeof(size_str), "%dx%d", (int)client->width, (int)client->height);

        char filename[256];
        snprintf(filename, sizeof(filename), "%s.mp4", env->scenario_id);

        client->recorder_pid = fork();
        if (client->recorder_pid == -1) {
            fprintf(stderr, "Failed to fork\n");
            free(client);
            return NULL;
        }

        if (client->recorder_pid == 0) { // Child process
            close(client->recorder_pipefd[1]);
            dup2(client->recorder_pipefd[0], STDIN_FILENO);
            close(client->recorder_pipefd[0]);
            for (int fd = 3; fd < 256; fd++)
                close(fd);
            execlp("ffmpeg", "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", size_str, "-r", "30", "-i",
                   "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast", "-crf", "23", "-loglevel",
                   "error", filename, NULL);
            fprintf(stderr, "execlp ffmpeg failed\n");
            _exit(1);
        }
        close(client->recorder_pipefd[0]);
    }

    return client;
}

// Camera control functions
void handle_camera_controls(Client *client) {
    static Vector2 prev_mouse_pos = {0};
    static bool is_dragging = false;
    float camera_move_speed = 0.5f;

    // Handle mouse drag for camera movement
    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        prev_mouse_pos = GetMousePosition();
        is_dragging = true;
    }

    if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
        is_dragging = false;
    }

    if (is_dragging) {
        Vector2 current_mouse_pos = GetMousePosition();
        Vector2 delta = {(current_mouse_pos.x - prev_mouse_pos.x) * camera_move_speed,
                         -(current_mouse_pos.y - prev_mouse_pos.y) * camera_move_speed};

        // Update camera position (only X and Y)
        client->camera.position.x += delta.x;
        client->camera.position.y += delta.y;

        // Update camera target (only X and Y)
        client->camera.target.x += delta.x;
        client->camera.target.y += delta.y;

        prev_mouse_pos = current_mouse_pos;
    }

    // Handle mouse wheel for zoom
    float wheel = GetMouseWheelMove();
    if (wheel != 0) {
        float zoom_factor = 1.0f - (wheel * 0.1f);
        // Calculate the current direction vector from target to position
        Vector3 direction = {client->camera.position.x - client->camera.target.x,
                             client->camera.position.y - client->camera.target.y,
                             client->camera.position.z - client->camera.target.z};

        // Scale the direction vector by the zoom factor
        direction.x *= zoom_factor;
        direction.y *= zoom_factor;
        direction.z *= zoom_factor;

        // Update the camera position based on the scaled direction
        client->camera.position.x = client->camera.target.x + direction.x;
        client->camera.position.y = client->camera.target.y + direction.y;
        client->camera.position.z = client->camera.target.z + direction.z;
    }
}

void draw_agent_obs(Drive *env, int agent_index, int mode, int obs_only, int lasers) {
    // Diamond dimensions
    float diamond_height = 3.0f; // Total height of diamond
    float diamond_width = 1.5f;  // Width of diamond
    float diamond_z = 8.0f;      // Base Z position

    // Define diamond points
    Vector3 top_point = (Vector3){0.0f, 0.0f, diamond_z + diamond_height / 2};    // Top point
    Vector3 bottom_point = (Vector3){0.0f, 0.0f, diamond_z - diamond_height / 2}; // Bottom point
    Vector3 front_point = (Vector3){0.0f, diamond_width / 2, diamond_z};          // Front point
    Vector3 back_point = (Vector3){0.0f, -diamond_width / 2, diamond_z};          // Back point
    Vector3 left_point = (Vector3){-diamond_width / 2, 0.0f, diamond_z};          // Left point
    Vector3 right_point = (Vector3){diamond_width / 2, 0.0f, diamond_z};          // Right point

    // Draw the diamond faces
    // Top pyramid
    if (mode == 0) {
        DrawTriangle3D(top_point, front_point, right_point, PUFF_CYAN); // Front-right face
        DrawTriangle3D(top_point, right_point, back_point, PUFF_CYAN);  // Back-right face
        DrawTriangle3D(top_point, back_point, left_point, PUFF_CYAN);   // Back-left face
        DrawTriangle3D(top_point, left_point, front_point, PUFF_CYAN);  // Front-left face

        // Bottom pyramid
        DrawTriangle3D(bottom_point, right_point, front_point, PUFF_CYAN); // Front-right face
        DrawTriangle3D(bottom_point, back_point, right_point, PUFF_CYAN);  // Back-right face
        DrawTriangle3D(bottom_point, left_point, back_point, PUFF_CYAN);   // Back-left face
        DrawTriangle3D(bottom_point, front_point, left_point, PUFF_CYAN);  // Front-left face
    }
    if (!IsKeyDown(KEY_LEFT_CONTROL) && obs_only == 0) {
        return;
    }

    // This overlay visualizes the vectorized entity sets, which do not exist in
    // perspective mode -- the observation there is the ego vector alone.
    if (env->obs_mode == OBS_MODE_RENDER_STATE)
        return;

    int ego_dim = drive_ego_dim(env);
    int max_obs = drive_obs_size(env);
    float (*observations)[max_obs] = (float (*)[max_obs])env->observations;
    float *agent_obs = &observations[agent_index][0];
    // self
    int active_idx = env->active_agent_indices[agent_index];
    float heading_self_x = env->entities[active_idx].heading_x;
    float heading_self_y = env->entities[active_idx].heading_y;
    float px = env->entities[active_idx].x;
    float py = env->entities[active_idx].y;
    // draw goal
    float goal_x = agent_obs[0] * 200;
    float goal_y = agent_obs[1] * 200;

    int agent_type = env->entities[active_idx].type;
    Color goal_color = LIGHTBLUE;
    if (agent_type == PEDESTRIAN)
        goal_color = LIGHT_ORANGE;
    else if (agent_type == CYCLIST)
        goal_color = LIGHT_PURPLE;

    if (mode == 0) { // agent-relative coordinates
        DrawSphere((Vector3){goal_x, goal_y, Z_AGENT_DETAILS}, 0.5f, goal_color);
        DrawCircle3D((Vector3){goal_x, goal_y, Z_AGENT_DETAILS}, env->goal_radius, (Vector3){0, 0, 1}, 90.0f,
                     Fade(goal_color, 0.3f));
    }

    if (mode == 1) { // world coordinates

        float goal_x_world = px + (goal_x * heading_self_x - goal_y * heading_self_y);
        float goal_y_world = py + (goal_x * heading_self_y + goal_y * heading_self_x);
        DrawSphere((Vector3){goal_x_world, goal_y_world, Z_AGENT_DETAILS}, 0.5f, goal_color);
        DrawCircle3D((Vector3){goal_x_world, goal_y_world, Z_AGENT_DETAILS}, env->goal_radius, (Vector3){0, 0, 1},
                     90.0f, Fade(goal_color, 0.3f));
    }
    // First draw other agent observations
    int obs_idx = ego_dim; // Start after ego obs
    for (int j = 0; j < MAX_PARTNER_OBS; j++) {
        if (agent_obs[obs_idx] == 0 || agent_obs[obs_idx + 1] == 0) {
            obs_idx += 7; // Move to next agent observation
            continue;
        }
        // Draw position of other agents
        float x = agent_obs[obs_idx] * 50;
        float y = agent_obs[obs_idx + 1] * 50;
        if (lasers && mode == 0) {
            DrawLine3D((Vector3){0, 0, 0}, (Vector3){x, y, Z_AGENT_DETAILS}, ORANGE);
        }

        float partner_x = px + (x * heading_self_x - y * heading_self_y);
        float partner_y = py + (x * heading_self_y + y * heading_self_x);
        if (lasers && mode == 1) {
            DrawLine3D((Vector3){px, py, Z_AGENT_DETAILS}, (Vector3){partner_x, partner_y, Z_AGENT_DETAILS}, ORANGE);
        }

        float half_width = 0.5 * agent_obs[obs_idx + 2] * MAX_VEH_WIDTH;
        float half_len = 0.5 * agent_obs[obs_idx + 3] * MAX_VEH_LEN;
        float theta_x = agent_obs[obs_idx + 4];
        float theta_y = agent_obs[obs_idx + 5];
        float partner_angle = atan2f(theta_y, theta_x);
        float cos_heading = cosf(partner_angle);
        float sin_heading = sinf(partner_angle);
        Vector3 corners[4] = {
            (Vector3){x + (half_len * cos_heading - half_width * sin_heading),
                      y + (half_len * sin_heading + half_width * cos_heading), Z_AGENT_DETAILS},
            (Vector3){x + (half_len * cos_heading + half_width * sin_heading),
                      y + (half_len * sin_heading - half_width * cos_heading), Z_AGENT_DETAILS},
            (Vector3){x + (-half_len * cos_heading + half_width * sin_heading),
                      y + (-half_len * sin_heading - half_width * cos_heading), Z_AGENT_DETAILS},
            (Vector3){x + (-half_len * cos_heading - half_width * sin_heading),
                      y + (-half_len * sin_heading + half_width * cos_heading), Z_AGENT_DETAILS},
        };

        if (mode == 0) {
            for (int j = 0; j < 4; j++) {
                DrawLine3D(corners[j], corners[(j + 1) % 4], ORANGE);
            }
        }

        if (mode == 1) {
            Vector3 world_corners[4];
            for (int j = 0; j < 4; j++) {
                float lx = corners[j].x;
                float ly = corners[j].y;

                world_corners[j].x = px + (lx * heading_self_x - ly * heading_self_y);
                world_corners[j].y = py + (lx * heading_self_y + ly * heading_self_x);
                world_corners[j].z = 1;
            }
            for (int j = 0; j < 4; j++) {
                DrawLine3D(world_corners[j], world_corners[(j + 1) % 4], ORANGE);
            }
        }

        // draw an arrow above the car pointing in the direction that the partner is going
        float arrow_length = 2.5f;
        float arrow_x = x + arrow_length * cosf(partner_angle);
        float arrow_y = y + arrow_length * sinf(partner_angle);
        float arrow_x_world;
        float arrow_y_world;
        if (mode == 0) {
            DrawLine3D((Vector3){x, y, Z_AGENT_DETAILS}, (Vector3){arrow_x, arrow_y, Z_AGENT_DETAILS}, PUFF_WHITE);
        }
        if (mode == 1) {
            arrow_x_world = px + (arrow_x * heading_self_x - arrow_y * heading_self_y);
            arrow_y_world = py + (arrow_x * heading_self_y + arrow_y * heading_self_x);
            DrawLine3D((Vector3){partner_x, partner_y, Z_AGENT_DETAILS},
                       (Vector3){arrow_x_world, arrow_y_world, Z_AGENT_DETAILS}, PUFF_WHITE);
        }
        // Calculate perpendicular offsets for arrow head
        float arrow_size = 0.3f; // Size of the arrow head
        float dx = arrow_x - x;
        float dy = arrow_y - y;
        float length = sqrtf(dx * dx + dy * dy);
        if (length > 0) {
            // Normalize direction vector
            dx /= length;
            dy /= length;

            // Calculate perpendicular vector
            float perp_x = -dy * arrow_size;
            float perp_y = dx * arrow_size;

            float arrow_x_end1 = arrow_x - dx * arrow_size + perp_x;
            float arrow_y_end1 = arrow_y - dy * arrow_size + perp_y;
            float arrow_x_end2 = arrow_x - dx * arrow_size - perp_x;
            float arrow_y_end2 = arrow_y - dy * arrow_size - perp_y;

            // Draw the two lines forming the arrow head
            if (mode == 0) {
                DrawLine3D((Vector3){arrow_x, arrow_y, 0.0}, (Vector3){arrow_x_end1, arrow_y_end1, 0.0}, PUFF_WHITE);
                DrawLine3D((Vector3){arrow_x, arrow_y, 0.0}, (Vector3){arrow_x_end2, arrow_y_end2, 0.0}, PUFF_WHITE);
            }

            if (mode == 1) {
                float arrow_x_end1_world = px + (arrow_x_end1 * heading_self_x - arrow_y_end1 * heading_self_y);
                float arrow_y_end1_world = py + (arrow_x_end1 * heading_self_y + arrow_y_end1 * heading_self_x);
                float arrow_x_end2_world = px + (arrow_x_end2 * heading_self_x - arrow_y_end2 * heading_self_y);
                float arrow_y_end2_world = py + (arrow_x_end2 * heading_self_y + arrow_y_end2 * heading_self_x);
                DrawLine3D((Vector3){arrow_x_world, arrow_y_world, 0.0},
                           (Vector3){arrow_x_end1_world, arrow_y_end1_world, 0.0}, PUFF_WHITE);
                DrawLine3D((Vector3){arrow_x_world, arrow_y_world, 0.0},
                           (Vector3){arrow_x_end2_world, arrow_y_end2_world, 0.0}, PUFF_WHITE);
            }
        }

        obs_idx += PARTNER_FEATURES; // Move to next agent observation (7 values per agent)
    }
    // Then draw map observations
    int map_start_idx = ego_dim + PARTNER_FEATURES * MAX_PARTNER_OBS; // Start after agent observations
    for (int k = 0; k < MAX_ROAD_SEGMENT_OBSERVATIONS; k++) {          // Loop through potential map entities
        int entity_idx = map_start_idx + k * 7;
        if (agent_obs[entity_idx] == 0 && agent_obs[entity_idx + 1] == 0) {
            continue;
        }
        Color lineColor = BLUE; // Default color
        int entity_type = (int)agent_obs[entity_idx + 6];
        // Choose color based on entity type
        if (entity_type + 4 != ROAD_EDGE) {
            continue;
        }
        lineColor = PUFF_CYAN;
        // For road segments, draw line between start and end points
        float x_middle = agent_obs[entity_idx] * 50;
        float y_middle = agent_obs[entity_idx + 1] * 50;
        float rel_angle_x = (agent_obs[entity_idx + 4]);
        float rel_angle_y = (agent_obs[entity_idx + 5]);
        float rel_angle = atan2f(rel_angle_y, rel_angle_x);
        float segment_length = agent_obs[entity_idx + 2] * MAX_ROAD_SEGMENT_LENGTH;
        // Calculate endpoint using the relative angle directly
        // Calculate endpoint directly
        float x_start = x_middle - segment_length * cosf(rel_angle);
        float y_start = y_middle - segment_length * sinf(rel_angle);
        float x_end = x_middle + segment_length * cosf(rel_angle);
        float y_end = y_middle + segment_length * sinf(rel_angle);

        if (lasers && mode == 0) {
            DrawLine3D((Vector3){0, 0, 0}, (Vector3){x_middle, y_middle, 1}, lineColor);
        }

        if (mode == 1) {
            float x_middle_world = px + (x_middle * heading_self_x - y_middle * heading_self_y);
            float y_middle_world = py + (x_middle * heading_self_y + y_middle * heading_self_x);
            float x_start_world = px + (x_start * heading_self_x - y_start * heading_self_y);
            float y_start_world = py + (x_start * heading_self_y + y_start * heading_self_x);
            float x_end_world = px + (x_end * heading_self_x - y_end * heading_self_y);
            float y_end_world = py + (x_end * heading_self_y + y_end * heading_self_x);
            DrawCube((Vector3){x_middle_world, y_middle_world, 1}, 0.5f, 0.5f, 0.5f, lineColor);
            DrawLine3D((Vector3){x_start_world, y_start_world, 1}, (Vector3){x_end_world, y_end_world, 1}, BLUE);
            if (lasers)
                DrawLine3D((Vector3){px, py, 1}, (Vector3){x_middle_world, y_middle_world, 1}, lineColor);
        }
        if (mode == 0) {
            DrawCube((Vector3){x_middle, y_middle, 1}, 0.5f, 0.5f, 0.5f, lineColor);
            DrawLine3D((Vector3){x_start, y_start, 1}, (Vector3){x_end, y_end, 1}, BLUE);
        }
    }
}

void draw_road_edge(Drive *env, float start_x, float start_y, float end_x, float end_y) {
    Color CURB_TOP = (Color){220, 220, 220, 255};  // Top surface - lightest
    Color CURB_SIDE = (Color){180, 180, 180, 255}; // Side faces - medium
    Color CURB_BOTTOM = (Color){160, 160, 160, 255};
    // Calculate curb dimensions
    float curb_height = 0.5f; // Height of the curb
    float curb_width = 0.3f;  // Width/thickness of the curb
    float road_z = 0.0f;      // Ensure z-level for roads is below agents

    // Calculate direction vector between start and end
    Vector3 direction = {end_x - start_x, end_y - start_y, 0.0f};

    // Calculate length of the segment
    float length = sqrtf(direction.x * direction.x + direction.y * direction.y);

    // Normalize direction vector
    Vector3 normalized_dir = {direction.x / length, direction.y / length, 0.0f};

    // Calculate perpendicular vector for width
    Vector3 perpendicular = {-normalized_dir.y, normalized_dir.x, 0.0f};

    // Calculate the four bottom corners of the curb
    Vector3 b1 = {start_x - perpendicular.x * curb_width / 2, start_y - perpendicular.y * curb_width / 2, road_z};
    Vector3 b2 = {start_x + perpendicular.x * curb_width / 2, start_y + perpendicular.y * curb_width / 2, road_z};
    Vector3 b3 = {end_x + perpendicular.x * curb_width / 2, end_y + perpendicular.y * curb_width / 2, road_z};
    Vector3 b4 = {end_x - perpendicular.x * curb_width / 2, end_y - perpendicular.y * curb_width / 2, road_z};

    // Draw the curb faces
    // Bottom face
    DrawTriangle3D(b1, b2, b3, CURB_BOTTOM);
    DrawTriangle3D(b1, b3, b4, CURB_BOTTOM);

    // Top face (raised by curb_height)
    Vector3 t1 = {b1.x, b1.y, b1.z + curb_height};
    Vector3 t2 = {b2.x, b2.y, b2.z + curb_height};
    Vector3 t3 = {b3.x, b3.y, b3.z + curb_height};
    Vector3 t4 = {b4.x, b4.y, b4.z + curb_height};
    DrawTriangle3D(t1, t3, t2, CURB_TOP);
    DrawTriangle3D(t1, t4, t3, CURB_TOP);

    // Side faces
    DrawTriangle3D(b1, t1, b2, CURB_SIDE);
    DrawTriangle3D(t1, t2, b2, CURB_SIDE);
    DrawTriangle3D(b2, t2, b3, CURB_SIDE);
    DrawTriangle3D(t2, t3, b3, CURB_SIDE);
    DrawTriangle3D(b3, t3, b4, CURB_SIDE);
    DrawTriangle3D(t3, t4, b4, CURB_SIDE);
    DrawTriangle3D(b4, t4, b1, CURB_SIDE);
    DrawTriangle3D(t4, t1, b1, CURB_SIDE);
}

void draw_scene(Drive *env, Client *client, int mode, int obs_only, int lasers, int show_grid) {

    if (show_grid) {
        float grid_start_x = env->grid_map->top_left_x;
        float grid_start_y = env->grid_map->bottom_right_y;
        for (int i = 0; i < env->grid_map->grid_cols; i++) {
            for (int j = 0; j < env->grid_map->grid_rows; j++) {
                float x = grid_start_x + i * GRID_CELL_SIZE;
                float y = grid_start_y + j * GRID_CELL_SIZE;
                DrawCubeWires((Vector3){x + GRID_CELL_SIZE / 2, y + GRID_CELL_SIZE / 2, 0.0f}, GRID_CELL_SIZE,
                              GRID_CELL_SIZE, 0.1f, Fade(PUFF_BACKGROUND2, 0.3f));
            }
        }
    }

    // Draw a grid to help with orientation
    for (int i = 0; i < env->num_entities; i++) {
        // Draw objects
        if (env->entities[i].type == VEHICLE || env->entities[i].type == PEDESTRIAN ||
            env->entities[i].type == CYCLIST) {
            // Check if this vehicle is an active agent
            bool is_active_agent = false;
            bool is_static_agent = false;
            int agent_index = -1;
            for (int j = 0; j < env->active_agent_count; j++) {
                if (env->active_agent_indices[j] == i) {
                    is_active_agent = true;
                    agent_index = j;
                    break;
                }
            }

            for (int j = 0; j < env->expert_static_agent_count; j++) {
                if (env->expert_static_agent_indices[j] == i) {
                    is_static_agent = true;
                    break;
                }
            }

            for (int j = 0; j < env->static_agent_count; j++) {
                if (env->static_agent_indices[j] == i) {
                    is_static_agent = true;
                    break;
                }
            }

            if (!is_active_agent && !is_static_agent) {
                continue;
            }
            Vector3 position;
            float heading;
            position = (Vector3){env->entities[i].x, env->entities[i].y, Z_AGENTS};
            heading = env->entities[i].heading;

            // Create size vector
            Vector3 size = {env->entities[i].length, env->entities[i].width, env->entities[i].height};

            bool is_expert = (!is_active_agent) && (env->entities[i].mark_as_expert == 1);

            // Save current transform
            if (mode == 1) {
                float cos_heading = env->entities[i].heading_x;
                float sin_heading = env->entities[i].heading_y;

                // Calculate half dimensions
                float half_len = env->entities[i].length * 0.5f;
                float half_width = env->entities[i].width * 0.5f;

                // Calculate the four corners of the collision box
                Vector3 corners[4] = {
                    (Vector3){position.x + (half_len * cos_heading - half_width * sin_heading),
                              position.y + (half_len * sin_heading + half_width * cos_heading), position.z},
                    (Vector3){position.x + (half_len * cos_heading + half_width * sin_heading),
                              position.y + (half_len * sin_heading - half_width * cos_heading), position.z},
                    (Vector3){position.x + (-half_len * cos_heading + half_width * sin_heading),
                              position.y + (-half_len * sin_heading - half_width * cos_heading), position.z},
                    (Vector3){position.x + (-half_len * cos_heading - half_width * sin_heading),
                              position.y + (-half_len * sin_heading + half_width * cos_heading), position.z},

                };

                if (agent_index == env->human_agent_idx &&
                    !env->entities[agent_index].metrics_array[REACHED_GOAL_IDX]) {
                    draw_agent_obs(env, agent_index, mode, obs_only, lasers);
                }

                if ((obs_only || IsKeyDown(KEY_LEFT_CONTROL)) && agent_index != env->human_agent_idx) {
                    continue;
                }

                // Draw the agent bounding boxes
                Color agent_color = GRAY;
                if (is_expert) {
                    if (env->entities[i].type == PEDESTRIAN || env->entities[i].type == CYCLIST)
                        agent_color = EXPERT_REPLAY_SMALL;
                    else
                        agent_color = EXPERT_REPLAY;
                }
                if (is_active_agent) {
                    if (env->entities[i].type == PEDESTRIAN)
                        agent_color = LIGHT_ORANGE;
                    else if (env->entities[i].type == CYCLIST)
                        agent_color = LIGHT_PURPLE;
                    else
                        agent_color = BLUE;
                }
                if (is_active_agent && env->entities[i].collision_state > 0)
                    agent_color = RED;

                rlPushMatrix();
                rlTranslatef(position.x, position.y, position.z);
                rlRotatef(heading * RAD2DEG, 0.0f, 0.0f, 1.0f);
                DrawCube((Vector3){0.0f, 0.0f, 0.0f}, size.x, size.y, 1.0f, Fade(agent_color, 0.5f));
                DrawCubeWires((Vector3){0.0f, 0.0f, 0.0f}, size.x, size.y, 1.0f, agent_color);
                rlPopMatrix();

                // Draw a heading arrow pointing forward
                Vector3 arrowStart = position;
                Vector3 arrowEnd = {position.x + cos_heading * half_len * 1.5f, // extend arrow beyond car
                                    position.y + sin_heading * half_len * 1.5f, position.z};

                DrawLine3D(arrowStart, arrowEnd, agent_color);
                DrawSphere(arrowEnd, 0.2f, agent_color); // arrow tip

            } else { // Agent view
                rlPushMatrix();
                // Translate to position, rotate around Y axis, then draw
                rlTranslatef(position.x, position.y, position.z);
                rlRotatef(heading * RAD2DEG, 0.0f, 0.0f, 1.0f); // Convert radians to degrees

                // Select car model (skip index 0)
                Model car_model = client->cars[(i % 5) + 1]; // Cycles through indices 1-5

                if (agent_index == env->human_agent_idx) {
                    car_model = client->cars[0]; // Ego agent always uses red car
                } else if (is_active_agent) {

                    car_model = client->cars[(i % 5) + 1];

                    if (env->entities[i].collision_state > 0) {
                        car_model = client->cars[0]; // Collided agents use red
                    }
                }
                // Draw obs for selected agent index
                if (agent_index == env->human_agent_idx &&
                    (!env->entities[agent_index].metrics_array[REACHED_GOAL_IDX] ||
                     env->goal_behavior == GOAL_GENERATE_NEW || env->goal_behavior == GOAL_STOP)) {
                    draw_agent_obs(env, agent_index, mode, obs_only, lasers);
                }

                // Draw cube for cars static and active
                // Calculate scale factors based on desired size and model dimensions
                BoundingBox bounds = GetModelBoundingBox(car_model);
                Vector3 model_size = {bounds.max.x - bounds.min.x, bounds.max.y - bounds.min.y,
                                      bounds.max.z - bounds.min.z};
                Vector3 scale = {size.x / model_size.x, size.y / model_size.y, size.z / model_size.z};

                if (env->entities[i].type == CYCLIST) {
                    scale = (Vector3){0.01, 0.01, 0.01};
                    car_model = client->cyclist;
                }
                if (env->entities[i].type == PEDESTRIAN) {
                    scale = (Vector3){2, 2, 2};
                    car_model = client->pedestrian;
                }
                DrawModelEx(car_model, (Vector3){0, 0, 0}, (Vector3){1, 0, 0}, 90.0f, scale, WHITE);
                {
                    float half_len = env->entities[i].length * 0.5f;
                    float half_width = env->entities[i].width * 0.5f;
                    Vector3 corners[4] = {
                        (Vector3){half_len, -half_width, 0},  // Front-left
                        (Vector3){half_len, half_width, 0},   // Front-right
                        (Vector3){-half_len, half_width, 0},  // Back-right
                        (Vector3){-half_len, -half_width, 0}, // Back-left
                    };
                    Color wire_color = GRAY;
                    if (!is_active_agent && env->entities[i].mark_as_expert == 1)
                        wire_color = EXPERT_REPLAY;
                    if (is_active_agent)
                        wire_color = BLUE; // Policy-controlled
                    if (is_active_agent && env->entities[i].collision_state > 0)
                        wire_color = RED;
                    rlSetLineWidth(2.0f);
                    for (int j = 0; j < 4; j++) {
                        DrawLine3D(corners[j], corners[(j + 1) % 4], wire_color);
                    }
                }
                rlPopMatrix();
            }

            // FPV Camera Control
            if (IsKeyDown(KEY_SPACE) && env->human_agent_idx == agent_index) {
                Vector3 camera_position = (Vector3){position.x - (25.0f * cosf(heading)),
                                                    position.y - (25.0f * sinf(heading)), position.z + 15};

                Vector3 camera_target = (Vector3){position.x + 40.0f * cosf(heading),
                                                  position.y + 40.0f * sinf(heading), position.z - 5.0f};
                client->camera.position = camera_position;
                client->camera.target = camera_target;
                client->camera.up = (Vector3){0, 0, 1};
            }
            if (IsKeyReleased(KEY_SPACE)) {
                client->camera.position = client->default_camera_position;
                client->camera.target = client->default_camera_target;
                client->camera.up = (Vector3){0, 0, 1};
            }
            // Draw goal position for active agents
            if (!is_active_agent || env->entities[i].valid == 0) {
                continue;
            }
            if (!IsKeyDown(KEY_LEFT_CONTROL) && obs_only == 0) {
                Color goal_color = DEEPBLUE;
                if (env->entities[i].type == PEDESTRIAN)
                    goal_color = LIGHT_ORANGE;
                else if (env->entities[i].type == CYCLIST)
                    goal_color = LIGHT_PURPLE;

                DrawSphere(
                    (Vector3){env->entities[i].goal_position_x, env->entities[i].goal_position_y, Z_AGENT_DETAILS},
                    0.5f, goal_color);
                DrawCircle3D(
                    (Vector3){env->entities[i].goal_position_x, env->entities[i].goal_position_y, Z_AGENT_DETAILS},
                    env->goal_radius, (Vector3){0, 0, Z_AGENT_DETAILS}, 90.0f, Fade(goal_color, 0.9f));
            }
        }
        // Draw road elements
        if (env->entities[i].type <= 3 && env->entities[i].type >= 7) {
            continue;
        }
        for (int j = 0; j < env->entities[i].array_size - 1; j++) {
            Vector3 start = {env->entities[i].traj_x[j], env->entities[i].traj_y[j], Z_ROAD_MARKINGS};
            Vector3 end = {env->entities[i].traj_x[j + 1], env->entities[i].traj_y[j + 1], Z_ROAD_MARKINGS};
            Color lineColor = GRAY;
            if (env->entities[i].type == ROAD_LANE)
                lineColor = Fade(SOFT_YELLOW, 0.25f);
            else if (env->entities[i].type == ROAD_LINE)
                lineColor = WHITE;
            else if (env->entities[i].type == ROAD_EDGE)
                lineColor = Fade(WHITE, 0.7f);
            else if (env->entities[i].type == DRIVEWAY)
                lineColor = RED;

            if (!IsKeyDown(KEY_LEFT_CONTROL) && obs_only == 0) {
                if (env->entities[i].type == ROAD_EDGE) {
                    draw_road_edge(env, start.x, start.y, end.x, end.y);
                } else if (env->entities[i].type == ROAD_LANE || env->entities[i].type == ROAD_LINE) {
                    // Draw road lanes and lines as purple lines
                    rlSetLineWidth(2.0f);
                    DrawLine3D(start, end, lineColor);
                }
            }
        }
    }

    EndMode3D();

    // Draw track indices for the tracks to predict
    if (mode == 1 && env->control_mode == CONTROL_WOSAC) {
        float map_height = env->grid_map->top_left_y - env->grid_map->bottom_right_y;
        float pixels_per_world_unit = client->view_h / map_height;

        for (int i = 0; i < env->active_agent_count; i++) {
            // Ignore respawned agents
            if (env->entities[i].respawn_timestep != -1) {
                continue;
            }
            int agent_idx = env->active_agent_indices[i];
            int womd_track_idx = env->tracks_to_predict_indices[i];

            float raw_x = -env->entities[agent_idx].x * pixels_per_world_unit;
            float raw_y = env->entities[agent_idx].y * pixels_per_world_unit;

            int screen_x = (int)raw_x + client->width / 2 + 20;
            int screen_y = (int)raw_y + client->view_h / 2 - 25;

            if (screen_x >= 0 && screen_x <= client->width && screen_y >= 0 && screen_y <= client->view_h) {
                char text[32];
                snprintf(text, sizeof(text), "%d", womd_track_idx);
                int text_width = MeasureText(text, 20);
                DrawText(text, screen_x - text_width / 2, screen_y, 20, PUFF_WHITE);
            }
        }
    }
}

// The simulator view is drawn off screen whenever a camera strip is reserved,
// then blitted to the top of the window. Rendering into a target of exactly the
// view's size is what keeps the projection's aspect ratio correct; clipping a
// full-window render would crop the map instead of fitting it.
static void begin_sim_view(Client *client) {
    if (client->cam_strip_h <= 0)
        return;
    BeginTextureMode(client->sim_view);
    ClearBackground(ROAD_COLOR);
}

static void end_sim_view(Client *client) {
    if (client->cam_strip_h <= 0)
        return;
    EndTextureMode();
    // Render textures come out y-flipped, hence the negative source height.
    Rectangle src = {0.0f, 0.0f, (float)client->sim_view.texture.width,
                     -(float)client->sim_view.texture.height};
    DrawTextureRec(client->sim_view.texture, src, (Vector2){0.0f, 0.0f}, WHITE);
}

// Draw the selected agent's camera views as a horizontal strip under the
// simulator view, in the rig's left-to-right order.
//
// These are the exact pixels handed to the policy, blitted from the rasterizer's
// output rather than redrawn with raylib, so what you see is the observation and
// not an approximation of it. Must be called in 2D space, after end_sim_view.
static void draw_camera_panels(Drive *env, Client *client) {
    if (env->render_camera_rgb == NULL || env->render_camera_count <= 0 || client->cam_strip_h <= 0)
        return;

    int cam_w = env->render_camera_width;
    int cam_h = env->render_camera_height;
    int n = env->render_camera_count;
    int tex_h = cam_h * n;

    if (client->camera_tex.id == 0 || client->camera_tex_w != cam_w || client->camera_tex_h != tex_h) {
        if (client->camera_tex.id != 0)
            UnloadTexture(client->camera_tex);
        Image img = {.data = env->render_camera_rgb,
                     .width = cam_w,
                     .height = tex_h,
                     .mipmaps = 1,
                     .format = PIXELFORMAT_UNCOMPRESSED_R8G8B8};
        client->camera_tex = LoadTextureFromImage(img);
        client->camera_tex_w = cam_w;
        client->camera_tex_h = tex_h;
    } else {
        UpdateTexture(client->camera_tex, env->render_camera_rgb);
    }
    // Nearest-neighbour keeps the low-resolution pixels legible when scaled up.
    SetTextureFilter(client->camera_tex, TEXTURE_FILTER_POINT);

    // Same layout make_client sized the window with, so the strip fits the band
    // exactly.
    CameraStrip s = camera_strip_layout((int)client->width, n, cam_w, cam_h);
    int strip_top = client->view_h;

    DrawRectangle(0, strip_top, (int)client->width, client->cam_strip_h, PUFF_BACKGROUND);
    // A hairline against the view above, so the strip reads as its own area.
    DrawRectangle(0, strip_top, (int)client->width, 1, PUFF_BACKGROUND2);

    int y = strip_top + s.margin;
    for (int i = 0; i < n; i++) {
        int x = s.x0 + i * (s.panel_w + s.gap);
        Rectangle src = {0.0f, (float)(i * cam_h), (float)cam_w, (float)cam_h};
        Rectangle dst = {(float)x, (float)(y + s.label_h), (float)s.panel_w, (float)s.panel_h};
        DrawTexturePro(client->camera_tex, src, dst, (Vector2){0, 0}, 0.0f, WHITE);
        DrawRectangleLines(x, y + s.label_h, s.panel_w, s.panel_h, PUFF_CYAN);

        char label[64];
        if (env->render_camera_names != NULL) {
            const char *name = env->render_camera_names + (size_t)i * env->render_camera_name_stride;
            snprintf(label, sizeof(label), "%s  %dx%d", name, cam_w, cam_h);
        } else {
            snprintf(label, sizeof(label), "cam %d  %dx%d", i, cam_w, cam_h);
        }
        DrawText(label, x, y + 2, 14, PUFF_WHITE);
    }
}

void c_render(Drive *env, int view_mode, int draw_traces) {
    // Kept in the signature so the binding stays call-compatible with ocean's, but a
    // synthetic scene has nothing to trace. See the sim-state branch below.
    (void)draw_traces;

    // Create client on first render call
    if (env->client == NULL) {
        env->client = make_client(env);
    }

    Client *client = env->client;

    if (env->render_mode == RENDER_HEADLESS) { // Headless rendering via ffmpeg
        float map_width = env->grid_map->bottom_right_x - env->grid_map->top_left_x;
        float map_height = env->grid_map->top_left_y - env->grid_map->bottom_right_y;

        Camera3D camera = {0};

        if (view_mode == VIEW_MODE_SIM_STATE) {
            // Orthographic bird's-eye view over the entire map (fully observable)
            camera.position = (Vector3){0.0, 0.0, 400.0f}; // Above the scene
            camera.target = (Vector3){0.0, 0.0, 0.0};      // Look at origin
            camera.up = (Vector3){0.0f, -1.0f, 0.0f};
            camera.projection = CAMERA_ORTHOGRAPHIC;
            camera.fovy = map_height;

            BeginDrawing();
            ClearBackground(ROAD_COLOR);
            begin_sim_view(client);
            BeginMode3D(camera);

            // ocean draws the logged trajectory of every agent here as one sphere per
            // timestep. A Gigaflow scene has no logged trajectory to draw: the agents
            // are synthesised, teddy_build_agents gives each a length-1 traj_* array
            // holding only the spawn pose, and there are no expert-replay agents at
            // all. Running the loop anyway read `episode_length` (1280) floats out of
            // a 1-element allocation and drew ~77k spheres of heap garbage per frame,
            // which is what made eval rendering ~27x slower than ocean's.

            draw_scene(env, client, 1, 0, 0, 0);

        } else if (view_mode == VIEW_MODE_BEV_AGENT_OBS) {
            // Orthographic bird's-eye view centered on the selected agent,
            // showing only that agent's observations
            int agent_idx = env->active_agent_indices[env->human_agent_idx];
            Entity *agent = &env->entities[agent_idx];

            Camera3D camera = {0};
            camera.position = (Vector3){agent->x, agent->y, 400.0f};
            camera.target = (Vector3){agent->x, agent->y, 0.0f};
            camera.up = (Vector3){0.0f, -1.0f, 0.0f};
            camera.projection = CAMERA_ORTHOGRAPHIC;
            camera.fovy = env->grid_map->vision_range * GRID_CELL_SIZE * 2.0f;

            BeginDrawing();
            ClearBackground(ROAD_COLOR);
            begin_sim_view(client);
            BeginMode3D(camera);
            draw_scene(env, client, 1, 1, 0, 0);

        } else { // First-person perspective from a selected agent
            int agent_idx = env->active_agent_indices[env->human_agent_idx];
            Entity *agent = &env->entities[agent_idx];

            Camera3D camera = {0};
            // Position camera behind and above the agent
            camera.position =
                (Vector3){agent->x - (25.0f * cosf(agent->heading)), agent->y - (25.0f * sinf(agent->heading)), 15.0f};
            camera.target =
                (Vector3){agent->x + 40.0f * cosf(agent->heading), agent->y + 40.0f * sinf(agent->heading), 1.0f};
            camera.up = (Vector3){0.0f, 0.0f, 1.0f};
            camera.fovy = 60.0f;
            camera.projection = CAMERA_PERSPECTIVE;

            BeginDrawing();
            ClearBackground(ROAD_COLOR);
            begin_sim_view(client);
            BeginMode3D(camera);
            draw_scene(env, client, 0, 0, 0, 1);
        }

        end_sim_view(client);
        draw_camera_panels(env, client);
        EndDrawing();

        unsigned char *screen_data = rlReadScreenPixels((int)client->width, (int)client->height);
        if (screen_data) {
            write(client->recorder_pipefd[1], screen_data, (int)client->width * (int)client->height * 4);
            RL_FREE(screen_data);
        }
    } else { // Pop-up window
        BeginDrawing();
        ClearBackground(ROAD_COLOR);
        begin_sim_view(client);
        BeginMode3D(client->camera);
        handle_camera_controls(env->client);
        draw_scene(env, client, 0, 0, 0, 0);
        end_sim_view(client);

        if (IsKeyPressed(KEY_TAB) && env->active_agent_count > 0) {
            env->human_agent_idx = (env->human_agent_idx + 1) % env->active_agent_count;
        }

        DrawText(TextFormat("Timestep: %d", env->timestep), 10, 50, 20, PUFF_WHITE);
        DrawText(TextFormat("Controlling agent: %d", env->human_agent_idx), 10, 70, 20, PUFF_WHITE);
        int human_idx = env->active_agent_indices[env->human_agent_idx];

        Color action_color = IsKeyDown(KEY_LEFT_SHIFT) ? YELLOW : PUFF_WHITE;

        if (env->action_type == 0) { // discrete
            int *action_array = (int *)env->actions;
            int action_val = action_array[env->human_agent_idx];

            if (env->dynamics_model == CLASSIC) {
                int num_steer = 13;
                int accel_idx = action_val / num_steer;
                int steer_idx = action_val % num_steer;
                float accel_value = ACCELERATION_VALUES[accel_idx];
                float steer_value = STEERING_VALUES[steer_idx];

                DrawText(TextFormat("Acceleration: %.2f m/s^2", accel_value), 10, 110, 20, action_color);
                DrawText(TextFormat("Steering: %.3f", steer_value), 10, 130, 20, action_color);
            } else if (env->dynamics_model == JERK) {
                int num_lat = 3;
                int jerk_long_idx = action_val / num_lat;
                int jerk_lat_idx = action_val % num_lat;
                float jerk_long_value = JERK_LONG[jerk_long_idx];
                float jerk_lat_value = JERK_LAT[jerk_lat_idx];

                DrawText(TextFormat("Longitudinal Jerk: %.2f m/s^3", jerk_long_value), 10, 110, 20, action_color);
                DrawText(TextFormat("Lateral Jerk: %.2f m/s^3", jerk_lat_value), 10, 130, 20, action_color);
            }
        } else { // continuous
            float (*action_array_f)[2] = (float (*)[2])env->actions;
            DrawText(TextFormat("Acceleration: %.2f", action_array_f[env->human_agent_idx][0]), 10, 110, 20,
                     action_color);
            DrawText(TextFormat("Steering: %.2f", action_array_f[env->human_agent_idx][1]), 10, 130, 20, action_color);
        }

        int status_y = 150;
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            DrawText("[shift pressed]", 10, status_y, 20, YELLOW);
            status_y += 20;
        }
        if (IsKeyDown(KEY_SPACE)) {
            DrawText("[space pressed]", 10, status_y, 20, YELLOW);
            status_y += 20;
        }
        if (IsKeyDown(KEY_LEFT_CONTROL)) {
            DrawText("[ctrl pressed]", 10, status_y, 20, YELLOW);
            status_y += 20;
        }

        DrawText("Controls: SHIFT + W/S - Accelerate/Brake, SHIFT + A/D - Steer, TAB - Switch Agent", 10,
                 client->view_h - 30, 20, PUFF_WHITE);
        DrawText(TextFormat("Grid Rows: %d", env->grid_map->grid_rows), 10, status_y, 20, PUFF_WHITE);
        DrawText(TextFormat("Grid Cols: %d", env->grid_map->grid_cols), 10, status_y + 20, 20, PUFF_WHITE);
        draw_camera_panels(env, client);
        EndDrawing();
    }
}

void close_client(Client *client) {
    if (client->sim_view.id != 0) {
        UnloadRenderTexture(client->sim_view);
        client->sim_view.id = 0;
    }
    if (client->camera_tex.id != 0) {
        UnloadTexture(client->camera_tex);
        client->camera_tex.id = 0;
    }
    if (client->recorder_pid > 0) {
        close(client->recorder_pipefd[1]);
        waitpid(client->recorder_pid, NULL, 0);
    }
    for (int i = 0; i < 6; i++)
        UnloadModel(client->cars[i]);
    UnloadModel(client->cyclist);
    UnloadModel(client->pedestrian);
    CloseWindow();
    if (client->xvfb_pid > 0) {
        kill(client->xvfb_pid, SIGTERM);
        waitpid(client->xvfb_pid, NULL, 0);
        unlink("/tmp/.X99-lock");
        unsetenv("DISPLAY");
    }

    free(client);
}
