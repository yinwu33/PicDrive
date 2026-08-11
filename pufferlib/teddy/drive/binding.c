#include "drive.h"
#define Env Drive
#define MY_SHARED
#define MY_PUT
#include "../env_binding.h"

// Fetch an optional contiguous float32 array from kwargs and hand back its data
// pointer and element count. Returns 0 when the key is absent, 1 on success,
// -1 with a Python error set on a malformed argument.
static int unpack_optional_f32(PyObject *kwargs, const char *key, float **data_out, int *count_out) {
    PyObject *obj = PyDict_GetItemString(kwargs, key);
    if (obj == NULL) {
        return 0;
    }
    if (!PyObject_TypeCheck(obj, &PyArray_Type)) {
        PyErr_Format(PyExc_TypeError, "%s must be a NumPy array", key);
        return -1;
    }
    PyArrayObject *arr = (PyArrayObject *)obj;
    if (!PyArray_ISCONTIGUOUS(arr)) {
        PyErr_Format(PyExc_ValueError, "%s must be contiguous", key);
        return -1;
    }
    if (PyArray_TYPE(arr) != NPY_FLOAT32) {
        PyErr_Format(PyExc_ValueError, "%s must be float32", key);
        return -1;
    }
    *data_out = (float *)PyArray_DATA(arr);
    *count_out = (int)PyArray_SIZE(arr);
    return 1;
}

static int my_put(Env *env, PyObject *args, PyObject *kwargs) {
    // Perspective-rendering buffers (Pictura). Optional: absent in vectorized mode.
    // These are written by the env and read by the CUDA rasterizer; they never
    // reach the policy.
    // Bind the counts array first so the road fill below can record its size.
    PyObject *counts = PyDict_GetItemString(kwargs, "render_counts");
    if (counts != NULL) {
        if (!PyObject_TypeCheck(counts, &PyArray_Type) || PyArray_TYPE((PyArrayObject *)counts) != NPY_INT32 ||
            !PyArray_ISCONTIGUOUS((PyArrayObject *)counts) || PyArray_SIZE((PyArrayObject *)counts) < 2) {
            PyErr_SetString(PyExc_ValueError, "render_counts must be a contiguous int32 array of size >= 2");
            return 1;
        }
        env->render_counts = (int *)PyArray_DATA((PyArrayObject *)counts);
    }

    float *data = NULL;
    int count = 0;
    int found = unpack_optional_f32(kwargs, "render_agents", &data, &count);
    if (found < 0)
        return 1;
    if (found) {
        env->render_agents = data;
        env->render_max_agents = count / RENDER_AGENT_FEATURES;
    }
    found = unpack_optional_f32(kwargs, "render_egos", &data, &count);
    if (found < 0)
        return 1;
    if (found) {
        env->render_egos = data;
    }
    found = unpack_optional_f32(kwargs, "render_roads", &data, &count);
    if (found < 0)
        return 1;
    if (found) {
        env->render_roads = data;
        env->render_max_roads = count / RENDER_ROAD_FEATURES;
        // Road geometry is static for the lifetime of the map, so fill it once here
        // rather than on every step.
        fill_render_roads(env);
    }

    // Per-step reward trace. Optional and off in training; see the TEDDY_DBG_* block
    // in drive.h for the row layout.
    found = unpack_optional_f32(kwargs, "debug_terms", &data, &count);
    if (found < 0)
        return 1;
    if (found) {
        if (count % TEDDY_DEBUG_FEATURES != 0) {
            PyErr_Format(PyExc_ValueError, "debug_terms must be a multiple of TEDDY_DEBUG_FEATURES (%d) long",
                         TEDDY_DEBUG_FEATURES);
            return 1;
        }
        env->debug_terms = data;
        env->debug_max_rows = count / TEDDY_DEBUG_FEATURES;
    }

    PyObject *cam = PyDict_GetItemString(kwargs, "render_camera_rgb");
    if (cam != NULL) {
        if (!PyObject_TypeCheck(cam, &PyArray_Type) || PyArray_TYPE((PyArrayObject *)cam) != NPY_UINT8 ||
            !PyArray_ISCONTIGUOUS((PyArrayObject *)cam) || PyArray_NDIM((PyArrayObject *)cam) != 4) {
            PyErr_SetString(PyExc_ValueError,
                            "render_camera_rgb must be a contiguous uint8 array [num_cameras, h, w, 3]");
            return 1;
        }
        PyArrayObject *arr = (PyArrayObject *)cam;
        if (PyArray_DIM(arr, 3) != 3) {
            PyErr_SetString(PyExc_ValueError, "render_camera_rgb must have 3 channels");
            return 1;
        }
        env->render_camera_rgb = (unsigned char *)PyArray_DATA(arr);
        env->render_camera_count = (int)PyArray_DIM(arr, 0);
        env->render_camera_height = (int)PyArray_DIM(arr, 1);
        env->render_camera_width = (int)PyArray_DIM(arr, 2);
    }

    // Panel labels, as fixed-stride NUL-padded ASCII rather than a list of
    // Python strings, so the viewer reads them straight out of the array Python
    // keeps alive alongside render_camera_rgb.
    PyObject *cam_names = PyDict_GetItemString(kwargs, "render_camera_names");
    if (cam_names != NULL) {
        if (!PyObject_TypeCheck(cam_names, &PyArray_Type) ||
            PyArray_TYPE((PyArrayObject *)cam_names) != NPY_UINT8 ||
            !PyArray_ISCONTIGUOUS((PyArrayObject *)cam_names) ||
            PyArray_NDIM((PyArrayObject *)cam_names) != 2) {
            PyErr_SetString(PyExc_ValueError,
                            "render_camera_names must be a contiguous uint8 array [num_cameras, stride]");
            return 1;
        }
        PyArrayObject *names = (PyArrayObject *)cam_names;
        if ((int)PyArray_DIM(names, 0) != env->render_camera_count) {
            PyErr_SetString(PyExc_ValueError, "render_camera_names must have one row per camera");
            return 1;
        }
        env->render_camera_names = (char *)PyArray_DATA(names);
        env->render_camera_name_stride = (int)PyArray_DIM(names, 1);
    }

    PyObject *obs = PyDict_GetItemString(kwargs, "observations");
    if (obs == NULL) {
        // Nothing else to bind. Used when only the render buffers are handed over.
        return 0;
    }
    if (!PyObject_TypeCheck(obs, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Observations must be a NumPy array");
        return 1;
    }
    PyArrayObject *observations = (PyArrayObject *)obs;
    if (!PyArray_ISCONTIGUOUS(observations)) {
        PyErr_SetString(PyExc_ValueError, "Observations must be contiguous");
        return 1;
    }
    env->observations = PyArray_DATA(observations);

    PyObject *act = PyDict_GetItemString(kwargs, "actions");
    if (!PyObject_TypeCheck(act, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Actions must be a NumPy array");
        return 1;
    }
    PyArrayObject *actions = (PyArrayObject *)act;
    if (!PyArray_ISCONTIGUOUS(actions)) {
        PyErr_SetString(PyExc_ValueError, "Actions must be contiguous");
        return 1;
    }
    env->actions = PyArray_DATA(actions);
    if (PyArray_ITEMSIZE(actions) == sizeof(double)) {
        PyErr_SetString(PyExc_ValueError, "Action tensor passed as float64 (pass np.float32 buffer)");
        return 1;
    }

    PyObject *rew = PyDict_GetItemString(kwargs, "rewards");
    if (!PyObject_TypeCheck(rew, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Rewards must be a NumPy array");
        return 1;
    }
    PyArrayObject *rewards = (PyArrayObject *)rew;
    if (!PyArray_ISCONTIGUOUS(rewards)) {
        PyErr_SetString(PyExc_ValueError, "Rewards must be contiguous");
        return 1;
    }
    if (PyArray_NDIM(rewards) != 1) {
        PyErr_SetString(PyExc_ValueError, "Rewards must be 1D");
        return 1;
    }
    env->rewards = PyArray_DATA(rewards);

    PyObject *term = PyDict_GetItemString(kwargs, "terminals");
    if (!PyObject_TypeCheck(term, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Terminals must be a NumPy array");
        return 1;
    }
    PyArrayObject *terminals = (PyArrayObject *)term;
    if (!PyArray_ISCONTIGUOUS(terminals)) {
        PyErr_SetString(PyExc_ValueError, "Terminals must be contiguous");
        return 1;
    }
    if (PyArray_NDIM(terminals) != 1) {
        PyErr_SetString(PyExc_ValueError, "Terminals must be 1D");
        return 1;
    }
    env->terminals = PyArray_DATA(terminals);
    return 0;
}

static PyObject *my_shared(PyObject *self, PyObject *args, PyObject *kwargs) {
    int num_agents = unpack(kwargs, "num_agents");
    int num_maps = unpack(kwargs, "num_maps");
    int agents_per_map_min = unpack(kwargs, "agents_per_map_min");
    int agents_per_map_max = unpack(kwargs, "agents_per_map_max");
    int seed = unpack(kwargs, "seed");

    // ocean loaded every candidate map here just to count how many logged tracks
    // were controllable. In a Gigaflow scene the agent count is a free parameter, so
    // the packing is pure arithmetic: no map is touched, which also makes
    // resample_maps cheap. The seed is honoured (ocean reseeded from the wall clock
    // and ignored the argument), so a scene layout is reproducible -- which is what
    // makes the viz.py acceptance render meaningful.
    if (agents_per_map_min < 1)
        agents_per_map_min = 1;
    if (agents_per_map_max > MAX_AGENTS)
        agents_per_map_max = MAX_AGENTS;
    if (agents_per_map_max < agents_per_map_min)
        agents_per_map_max = agents_per_map_min;

    TeddyRng rng;
    teddy_rand_seed(&rng, (uint64_t)seed, 0x9E3779B97F4A7C15ULL);

    int max_envs = num_agents;
    PyObject *agent_offsets = PyList_New(max_envs + 1);
    PyObject *map_ids = PyList_New(max_envs);

    int total_agent_count = 0;
    int env_count = 0;
    while (total_agent_count < num_agents && env_count < max_envs) {
        int map_id = teddy_rand_int(&rng, 0, num_maps - 1);
        int n = teddy_rand_int(&rng, agents_per_map_min, agents_per_map_max);
        if (total_agent_count + n > num_agents)
            n = num_agents - total_agent_count; // clamp the last scene to fit the budget
        if (n < 1)
            break;

        PyList_SetItem(map_ids, env_count, PyLong_FromLong(map_id));
        PyList_SetItem(agent_offsets, env_count, PyLong_FromLong(total_agent_count));
        total_agent_count += n;
        env_count++;
    }

    PyList_SetItem(agent_offsets, env_count, PyLong_FromLong(total_agent_count));
    PyObject *final_env_count = PyLong_FromLong(env_count);

    PyObject *resized_agent_offsets = PyList_GetSlice(agent_offsets, 0, env_count + 1);
    PyObject *resized_map_ids = PyList_GetSlice(map_ids, 0, env_count);
    Py_DECREF(agent_offsets);
    Py_DECREF(map_ids);
    PyObject *tuple = PyTuple_New(3);
    PyTuple_SetItem(tuple, 0, resized_agent_offsets);
    PyTuple_SetItem(tuple, 1, resized_map_ids);
    PyTuple_SetItem(tuple, 2, final_env_count);
    return tuple;
}

static int my_init(Env *env, PyObject *args, PyObject *kwargs) {
    env->human_agent_idx = unpack(kwargs, "human_agent_idx");
    env->ini_file = unpack_str(kwargs, "ini_file");
    env_init_config conf = {0};
    env_config_set_teddy_defaults(&conf);
    if (ini_parse(env->ini_file, handler, &conf) < 0) {
        printf("Error while loading %s", env->ini_file);
    }
    if (kwargs && PyDict_GetItemString(kwargs, "episode_length")) {
        conf.episode_length = (int)unpack(kwargs, "episode_length");
    }
    if (conf.episode_length <= 0) {
        PyErr_SetString(PyExc_ValueError, "episode_length must be > 0 (set in INI or kwargs)");
        return -1;
    }

// Allow all settings to be overridden via kwargs (ini provides defaults)
#define OVERRIDE_INT(field)                                                                                            \
    if (kwargs && PyDict_GetItemString(kwargs, #field)) {                                                              \
        conf.field = (int)unpack(kwargs, #field);                                                                      \
    }
#define OVERRIDE_FLOAT(field)                                                                                          \
    if (kwargs && PyDict_GetItemString(kwargs, #field)) {                                                              \
        conf.field = (float)unpack(kwargs, #field);                                                                    \
    }

    OVERRIDE_INT(render_mode);
    OVERRIDE_INT(action_type);
    OVERRIDE_INT(dynamics_model);
    OVERRIDE_FLOAT(reward_vehicle_collision);
    OVERRIDE_FLOAT(reward_offroad_collision);
    OVERRIDE_FLOAT(reward_goal);
    OVERRIDE_FLOAT(reward_goal_post_respawn);
    OVERRIDE_INT(collision_behavior);
    OVERRIDE_INT(offroad_behavior);
    OVERRIDE_FLOAT(dt);
    OVERRIDE_INT(termination_mode);
    OVERRIDE_INT(init_mode);
    OVERRIDE_INT(control_mode);
    OVERRIDE_INT(goal_behavior);
    OVERRIDE_FLOAT(goal_target_distance);
    OVERRIDE_FLOAT(goal_radius);
    OVERRIDE_FLOAT(goal_speed);
    OVERRIDE_INT(max_controlled_agents);
    OVERRIDE_FLOAT(partner_obs_radius);
    OVERRIDE_INT(obs_mode);
    OVERRIDE_INT(render_road_types);
    OVERRIDE_INT(agents_per_map_min);
    OVERRIDE_INT(agents_per_map_max);
    OVERRIDE_FLOAT(spawn_speed_max);
    OVERRIDE_FLOAT(spawn_heading_jitter_deg);
    OVERRIDE_FLOAT(wrong_way_frac);
    OVERRIDE_INT(num_waypoints_max);
    OVERRIDE_FLOAT(waypoint_min_dist);
    OVERRIDE_FLOAT(waypoint_max_dist);

#undef OVERRIDE_INT
#undef OVERRIDE_FLOAT

    env->action_type = conf.action_type;
    env->dynamics_model = conf.dynamics_model;
    env->reward_vehicle_collision = conf.reward_vehicle_collision;
    env->reward_offroad_collision = conf.reward_offroad_collision;
    env->reward_goal = conf.reward_goal;
    env->reward_goal_post_respawn = conf.reward_goal_post_respawn;
    env->episode_length = conf.episode_length;
    env->termination_mode = conf.termination_mode;
    env->collision_behavior = conf.collision_behavior;
    env->offroad_behavior = conf.offroad_behavior;
    env->max_controlled_agents = unpack(kwargs, "max_controlled_agents");
    // Non-positive (including a config that omits the key, since env_init_config is
    // zero-initialized) falls back to the radius both envs used before it was a knob.
    env->partner_obs_radius =
        (conf.partner_obs_radius > 0.0f) ? conf.partner_obs_radius : PARTNER_OBS_RADIUS_DEFAULT;
    env->dt = conf.dt;
    env->init_mode = (int)unpack(kwargs, "init_mode");
    env->control_mode = (int)unpack(kwargs, "control_mode");
    env->goal_behavior = (int)unpack(kwargs, "goal_behavior");
    env->goal_target_distance = (float)unpack(kwargs, "goal_target_distance");
    env->goal_radius = (float)unpack(kwargs, "goal_radius");
    env->goal_speed = (float)unpack(kwargs, "goal_speed");
    env->render_mode = (int)unpack(kwargs, "render_mode");
    // Must be set before init(): it decides the observation stride and seeds the
    // default road-type mask.
    env->obs_mode = conf.obs_mode;
    env->render_road_types = conf.render_road_types;
    env->agents_per_map_min = conf.agents_per_map_min;
    env->agents_per_map_max = conf.agents_per_map_max;
    env->spawn_speed_max = conf.spawn_speed_max;
    env->spawn_heading_jitter_deg = conf.spawn_heading_jitter_deg;
    env->wrong_way_frac = conf.wrong_way_frac;
    env->num_waypoints_max = conf.num_waypoints_max;
    env->waypoint_min_dist = conf.waypoint_min_dist;
    env->waypoint_max_dist = conf.waypoint_max_dist;

    char *agent_dist_path = unpack_str(kwargs, "agent_dist_path");
    if (!agent_dist_load(agent_dist_path)) {
        PyErr_Format(PyExc_ValueError,
                     "could not load the agent state distribution from '%s'. Build it with: "
                     "python data_utils/womd/build_agent_dist.py",
                     agent_dist_path);
        free(agent_dist_path);
        return -1;
    }
    free(agent_dist_path);

    char *map_dir = unpack_str(kwargs, "map_dir");
    int map_id = unpack(kwargs, "map_id");
    int max_agents = unpack(kwargs, "max_agents");
    int init_steps = unpack(kwargs, "init_steps");
    char map_file[512];
    snprintf(map_file, sizeof(map_file), "%s/map_%03d.bin", map_dir, map_id);
    env->num_agents = max_agents;
    // Seed per env, not per process. Every scene draws its own poses, sizes and
    // routes, so a shared global stream would make scenes correlated and would make
    // the viz.py acceptance render irreproducible. `map_id` enters the stream id so
    // two envs handed the same seed but different maps still diverge.
    int env_seed = (int)unpack(kwargs, "seed");
    teddy_rand_seed(&env->rng, (uint64_t)env_seed, (uint64_t)(map_id * 2654435761u + 1u));
    env->map_name = strdup(map_file);
    env->init_steps = init_steps;
    env->timestep = init_steps;
    init(env);
    return 0;
}

static int my_log(PyObject *dict, Log *log) {
    assign_to_dict(dict, "n", log->n);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "offroad_rate", log->offroad_rate);
    assign_to_dict(dict, "collision_rate", log->collision_rate);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "dnf_rate", log->dnf_rate);
    assign_to_dict(dict, "completion_rate", log->completion_rate);
    assign_to_dict(dict, "lane_alignment_rate", log->lane_alignment_rate);
    assign_to_dict(dict, "perc_controlled", log->perc_controlled);
    assign_to_dict(dict, "perc_other", log->perc_other);
    assign_to_dict(dict, "offroad_per_agent", log->offroad_per_agent);
    assign_to_dict(dict, "collisions_per_agent", log->collisions_per_agent);
    assign_to_dict(dict, "goals_sampled_this_episode", log->goals_sampled_this_episode);
    assign_to_dict(dict, "goals_reached_this_episode", log->goals_reached_this_episode);
    assign_to_dict(dict, "speed_at_goal", log->speed_at_goal);
    // assign_to_dict(dict, "avg_displacement_error", log->avg_displacement_error);
    return 0;
}
