// CUDA perspective rasterizer for PufferDrive (Pictura reproduction).
//
// Must match pufferlib/ocean/drive/raster_ref.py within one uint8 level; that
// reference is the semantic definition and this is the fast path. See
// tests/test_raster.py::test_cuda_matches_reference.
//
// Following the paper, the scene is drawn as flat-shaded geometric primitives on
// the GPU's general-purpose compute cores rather than through the fixed-function
// graphics pipeline, so it runs inside the training process with no host round
// trip and needs no graphics driver.
//
// Structure, per rendered image:
//   1. count_kernel projects every primitive and counts the triangles that land
//      on screen, without writing them.
//   2. The host prefix-sums those counts, so each image owns exactly as much
//      scratch as it needs.
//   3. transform_kernel repeats the projection and writes the surviving
//      triangles compacted into that slice, alongside the index each triangle
//      would have had in the old fixed-stride layout.
//   4. raster_kernel takes one tile of one image per block, one pixel per thread.
//      Threads cooperatively compact the triangles whose bounding box meets the
//      tile into shared memory, a chunk at a time, and each thread shades its own
//      pixel against each chunk in turn.
//
// Compaction is what makes this affordable at self-play scale. A scene holds a
// couple of thousand road segments but one 96x64 camera sees about a sixth of
// them, and the rest used to occupy scratch and be rescanned by every tile: at
// 2048 egos x 3 cameras that was 5.4 GB written and 17.7 GB reread per step,
// more than half of it padding to the largest scene in the batch. Counting first
// costs one extra projection pass, which is compute the transform was not bound
// by, and removes both.
//
// The scratch order now depends on the order threads happen to reserve slots in,
// so nothing downstream may depend on it. Nothing does: `tri_index` carries the
// old fixed-stride index, road fragments merge by colour and sort on it, and
// agent fragments sort on (quantised depth, index). Both orderings are recovered
// from the key rather than from the buffer, which is what makes the compaction
// safe.
//
// Visibility follows the reference's three layers: an analytic ray/ground-plane
// background, road markings drawn over it (coplanar, so no depth test), and agent
// boxes depth-tested against the ground intercept. Within a layer, fragments are
// combined front to back with analytic edge coverage as alpha, except that road
// fragments of one colour are folded together first: they tile a surface rather
// than stack on it, so their coverages add.

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace pufferlib {

#define RASTER_AGENT_FEATURES 8
#define RASTER_ROAD_FEATURES 6
#define RASTER_RIG_STRIDE 20

// Floats per transformed triangle in the scratch buffer:
// u0,v0,z0, u1,v1,z1, u2,v2,z2, r,g,b, umin,umax,vmin,vmax
#define TRI_STRIDE 16

// Index slots reserved per source triangle. Clipping one triangle against the
// near plane yields at most two, and the pair shares a base index so the
// tie-break key stays exactly what the fixed-stride layout used to produce.
#define TRIS_PER_PRIM 2

// Screen-space tile handled by one block.
#define TILE_W 32
#define TILE_H 16

// Threads per block. The raster pass runs one thread per tile pixel, which is
// what lets a pixel's fragment state live in that thread across the chunk loop
// below, and what lifts occupancy: shared memory is charged per block, so at 128
// threads a 32 KB block bought only 384 resident threads on an A6000 (25% of the
// 1536 it can hold), while at 512 it buys all of them from the same allocation.
#define TRANSFORM_THREADS 128
#define RASTER_THREADS (TILE_W * TILE_H)

// Triangles staged into shared memory at a time. The tile's triangles are walked
// in index-range chunks of this size, so the staging list cannot overflow however
// dense the scene is -- at most TILE_CHUNK candidates are ever offered to it. The
// fixed 4096-entry list this replaces silently dropped whatever a tile held past
// it; measured on a 2048-ego batch the mean tile holds 58 road triangles and the
// densest 4074, so the cap had to be sized for a case 70x the mean and still had
// no margin. Road and agent triangles reuse the one buffer, since their passes
// are sequential.
#define TILE_CHUNK 1024

// Fragments retained per pixel per layer, nearest first. Front-to-back
// compositing terminates once alpha saturates, so this only truncates when many
// partially covering fragments stack up at one pixel.
#define MAX_FRAGS 16

#define PALETTE_DEPTH_FALLOFF 60.0f
#define PALETTE_DEPTH_MIN_SCALE 0.30f
#define NEAR_EPS 1e-9f

// Relative slack on the agent layer's ground-intercept bound. A box's bottom face
// lies exactly on the ground plane, so its depth equals the intercept to within
// rounding; comparing exactly decides that face by float noise, and the CPU
// reference rounds it the other way. See raster_ref.GROUND_DEPTH_SLACK.
#define GROUND_DEPTH_SLACK 1e-4f

// Depth-ordering quantum, in metres; see frag_after and
// raster_ref.DEPTH_ORDER_QUANTUM.
#define DEPTH_ORDER_QUANTUM 1e-3f

__device__ __forceinline__ float depth_scale(float depth) {
    float s = 1.0f - fmaxf(depth, 0.0f) / PALETTE_DEPTH_FALLOFF;
    return fminf(fmaxf(s, PALETTE_DEPTH_MIN_SCALE), 1.0f);
}

// Horizon colour. Both the background and every fragment fade toward it with
// distance; see raster_ref._haze for why the two cannot fade differently.
__constant__ float PALETTE_SKY[3] = {0.62f, 0.70f, 0.80f};

// Flat-shaded palette, mirroring raster_ref.Palette.
__device__ __forceinline__ void agent_color(int type_id, float *out) {
    if (type_id == 2) {  // PEDESTRIAN
        out[0] = 0.30f; out[1] = 0.85f; out[2] = 0.40f;
    } else if (type_id == 3) {  // CYCLIST
        out[0] = 0.98f; out[1] = 0.78f; out[2] = 0.20f;
    } else {  // VEHICLE and anything unexpected
        out[0] = 0.90f; out[1] = 0.26f; out[2] = 0.24f;
    }
}

__device__ __forceinline__ void road_color(int type_id, float *out) {
    switch (type_id) {
    case 4:  out[0] = 0.55f; out[1] = 0.55f; out[2] = 0.55f; break;  // ROAD_LANE
    case 6:  out[0] = 0.60f; out[1] = 0.60f; out[2] = 0.64f; break;  // ROAD_EDGE
    case 8:  out[0] = 0.92f; out[1] = 0.92f; out[2] = 0.92f; break;  // CROSSWALK
    case 9:  out[0] = 0.95f; out[1] = 0.80f; out[2] = 0.25f; break;  // SPEED_BUMP
    case 10: out[0] = 0.45f; out[1] = 0.45f; out[2] = 0.48f; break;  // DRIVEWAY
    case 11: out[0] = 0.32f; out[1] = 0.32f; out[2] = 0.34f; break;  // shared lane area
    case 12: out[0] = 1.00f; out[1] = 0.82f; out[2] = 0.00f; break;  // shared road edge
    default: out[0] = 0.95f; out[1] = 0.95f; out[2] = 0.95f; break;  // ROAD_LINE
    }
}

// Per-face brightness of an agent box: front, back, left, right, top, bottom.
// A heading code rather than a light model, so the six only have to stay mutually
// distinguishable; `back` is raised off the floor because car-following looks at
// it constantly. See raster_ref.Palette.face_shade.
__constant__ float FACE_SHADE[6] = {1.00f, 0.72f, 0.86f, 0.62f, 0.95f, 0.32f};

// Cuboid faces as indices into the 8 corners, ordered (front/back, left/right,
// bottom/top) with bit weights 4/2/1. Two triangles per face.
__constant__ int BOX_TRIS[12][3] = {
    {4, 6, 7}, {4, 7, 5},  // front  (+x local)
    {2, 0, 1}, {2, 1, 3},  // back   (-x local)
    {6, 2, 3}, {6, 3, 7},  // left   (+y local)
    {0, 4, 5}, {0, 5, 1},  // right  (-y local)
    {1, 5, 7}, {1, 7, 3},  // top    (+z)
    {0, 2, 6}, {0, 6, 4},  // bottom (z = 0)
};

struct Cam {
    float rot[9];  // ego -> camera, rows are right/down/forward
    float pos[3];
    float fx, fy, cx, cy;
    int width, height;
    float near_z, far_z;
};

__device__ __forceinline__ Cam load_cam(const float *rig, int cam_idx) {
    const float *r = rig + cam_idx * RASTER_RIG_STRIDE;
    Cam c;
#pragma unroll
    for (int i = 0; i < 9; i++) c.rot[i] = r[i];
#pragma unroll
    for (int i = 0; i < 3; i++) c.pos[i] = r[9 + i];
    c.fx = r[12]; c.fy = r[13]; c.cx = r[14]; c.cy = r[15];
    c.width = (int)r[16]; c.height = (int)r[17];
    c.near_z = r[18]; c.far_z = r[19];
    return c;
}

// World -> ego (2D rotation about the ego pose) -> camera.
__device__ __forceinline__ void to_camera(const Cam &c, const float *ego, float wx, float wy, float wz,
                                          float *out) {
    float dx = wx - ego[0];
    float dy = wy - ego[1];
    float ex = dx * ego[2] + dy * ego[3] - c.pos[0];
    float ey = -dx * ego[3] + dy * ego[2] - c.pos[1];
    float ez = wz - c.pos[2];
    out[0] = c.rot[0] * ex + c.rot[1] * ey + c.rot[2] * ez;
    out[1] = c.rot[3] * ex + c.rot[4] * ey + c.rot[5] * ez;
    out[2] = c.rot[6] * ex + c.rot[7] * ey + c.rot[8] * ez;
}

// Project one camera-frame triangle to screen space. Returns false when its
// bounding box misses the frame, which is the test that used to run after the
// triangle had already been written to scratch. Every vertex must already be at
// or beyond the near plane, which is what clip_prim below guarantees.
__device__ __forceinline__ bool project_tri(const Cam &c, const float *p0, const float *p1,
                                            const float *p2, const float *color, float *t) {
    float u0 = c.fx * p0[0] / p0[2] + c.cx, v0 = c.fy * p0[1] / p0[2] + c.cy;
    float u1 = c.fx * p1[0] / p1[2] + c.cx, v1 = c.fy * p1[1] / p1[2] + c.cy;
    float u2 = c.fx * p2[0] / p2[2] + c.cx, v2 = c.fy * p2[1] / p2[2] + c.cy;

    float umin = fminf(u0, fminf(u1, u2)), umax = fmaxf(u0, fmaxf(u1, u2));
    float vmin = fminf(v0, fminf(v1, v2)), vmax = fmaxf(v0, fmaxf(v1, v2));
    if (umax < -0.5f || umin > c.width + 0.5f || vmax < -0.5f || vmin > c.height + 0.5f) return false;

    if (t == nullptr) return true;
    t[0] = u0; t[1] = v0; t[2] = p0[2];
    t[3] = u1; t[4] = v1; t[5] = p1[2];
    t[6] = u2; t[7] = v2; t[8] = p2[2];
    t[9] = color[0]; t[10] = color[1]; t[11] = color[2];
    t[12] = umin; t[13] = umax; t[14] = vmin; t[15] = vmax;
    return true;
}

// Where the segment p_in -> p_out crosses the near plane. p_out is behind the
// plane and p_in in front of it, so the denominator is negative by construction
// and clamping its magnitude has to keep that sign.
__device__ __forceinline__ void near_point(const float *p_in, const float *p_out, float near_z,
                                           float *out) {
    float den = p_out[2] - p_in[2];
    if (den > -NEAR_EPS) den = -NEAR_EPS;
    float t = (near_z - p_in[2]) / den;
#pragma unroll
    for (int k = 0; k < 3; k++) out[k] = p_in[k] + t * (p_out[k] - p_in[k]);
}

// Reserve a compacted slot and store the triangle, or just count it when
// `data` is null. `base_idx` is the index this triangle had in the old
// fixed-stride layout and travels with it as the tie-break key.
__device__ __forceinline__ void put_tri(const Cam &c, const float *p0, const float *p1, const float *p2,
                                        const float *color, int base_idx, float *data, int *index,
                                        int *cursor, int capacity, int &counted) {
    if (data == nullptr) {
        if (project_tri(c, p0, p1, p2, color, nullptr)) counted++;
        return;
    }
    float rec[TRI_STRIDE];
    if (!project_tri(c, p0, p1, p2, color, rec)) return;
    int slot = atomicAdd(cursor, 1);
    // Exact by construction: the counting pass runs this same projection. The
    // bound is insurance against a write past the slice rather than an expected
    // path -- dropping a triangle is far cheaper than corrupting a neighbour.
    if (slot >= capacity) return;
    float *dst = data + (size_t)slot * TRI_STRIDE;
#pragma unroll
    for (int k = 0; k < TRI_STRIDE; k++) dst[k] = rec[k];
    index[slot] = base_idx;
    counted++;
}

// Clip a camera-frame triangle against the near plane and emit the pieces.
//
// Dropping straddling triangles instead, which is what this used to do, blanks
// whatever the camera is standing on. With the drivable area drawn as ground
// quads that is the road right in front of the vehicle: the quad the ego sat in
// vanished for as long as it was inside that segment -- up to 16 m of road on
// WOMD's decimated centrelines -- and returned at the next one, so the surface
// flickered off block by block as the car drove. The cut and the order the pieces
// are emitted in mirror raster_ref._clip_near, so both implementations rasterize
// the same geometry.
__device__ __forceinline__ void clip_prim(const Cam &c, const float *p0, const float *p1,
                                          const float *p2, const float *color, int base_idx,
                                          float *data, int *index, int *cursor, int capacity,
                                          int &counted) {
    const float *p[3] = {p0, p1, p2};
    int inside[3];
    int n_inside = 0;
#pragma unroll
    for (int i = 0; i < 3; i++) {
        inside[i] = p[i][2] >= c.near_z;
        n_inside += inside[i];
    }

    if (n_inside == 0) return;

    if (n_inside == 3) {
        put_tri(c, p0, p1, p2, color, base_idx, data, index, cursor, capacity, counted);
        return;
    }

    if (n_inside == 1) {
        // One vertex inside: the triangle shrinks to a smaller triangle.
        int i = inside[0] ? 0 : (inside[1] ? 1 : 2);
        const float *a = p[i], *b = p[(i + 1) % 3], *d = p[(i + 2) % 3];
        float ab[3], ad[3];
        near_point(a, b, c.near_z, ab);
        near_point(a, d, c.near_z, ad);
        put_tri(c, a, ab, ad, color, base_idx, data, index, cursor, capacity, counted);
        return;
    }

    // Two vertices inside: the triangle becomes a quad, emitted as two triangles.
    int i = !inside[0] ? 0 : (!inside[1] ? 1 : 2);
    const float *a = p[i], *b = p[(i + 1) % 3], *d = p[(i + 2) % 3];
    float ab[3], ad[3];
    near_point(b, a, c.near_z, ab);
    near_point(d, a, c.near_z, ad);
    put_tri(c, b, d, ad, color, base_idx, data, index, cursor, capacity, counted);
    put_tri(c, b, ad, ab, color, base_idx + 1, data, index, cursor, capacity, counted);
}

// Project one image's primitives, either counting the survivors or writing them.
// Both passes call this so the count is exactly what the write produces.
//
// Egos are grouped into scenes; each scene owns a contiguous slice of the agent
// and road arrays. One launch therefore covers every environment in the batch,
// which matters because a training step holds on the order of a hundred of them
// and per-scene launches would be dominated by their own overhead.
__device__ __forceinline__ void project_image(int image, const float *agents, const float *roads,
                                              const float *egos, int ego_stride, const float *rig,
                                              int num_cams, const int *ego_scene,
                                              const int *agent_ranges, const int *road_ranges,
                                              float *road_data, int *road_index, int *road_cursor,
                                              int road_capacity, float *agent_data, int *agent_index,
                                              int *agent_cursor, int agent_capacity, int &n_road,
                                              int &n_agent) {
    int ego_i = image / num_cams;
    int cam_i = image % num_cams;
    Cam c = load_cam(rig, cam_i);
    const float *ego = egos + (size_t)ego_i * ego_stride;
    int self_index = ego_stride > 4 ? (int)ego[4] : -1;

    int scene = ego_scene[ego_i];
    int agent_lo = agent_ranges[scene], agent_hi = agent_ranges[scene + 1];
    int road_lo = road_ranges[scene], road_hi = road_ranges[scene + 1];
    int num_agents = agent_hi - agent_lo;
    int num_roads = road_hi - road_lo;

    for (int i = threadIdx.x; i < num_roads; i += blockDim.x) {
        const float *r = roads + (size_t)(road_lo + i) * RASTER_ROAD_FEATURES;
        float half_w = r[4] * 0.5f;
        float dx = r[2] - r[0], dy = r[3] - r[1];
        float len = fmaxf(sqrtf(dx * dx + dy * dy), 1e-6f);
        // Left normal of the segment direction, giving the painted strip width.
        float nx = -dy / len * half_w, ny = dx / len * half_w;

        float col[3];
        road_color((int)r[5], col);

        // The opaque lane area sits just below the painted features it
        // carries. Which of the two is drawn on top comes from buffer order, not
        // from this gap; see finalize_road_fragments.
        float road_z = ((int)r[5] == 11) ? -0.01f : 0.0f;
        float a[3], b[3], d[3], e[3];
        to_camera(c, ego, r[0] + nx, r[1] + ny, road_z, a);
        to_camera(c, ego, r[0] - nx, r[1] - ny, road_z, b);
        to_camera(c, ego, r[2] - nx, r[3] - ny, road_z, d);
        to_camera(c, ego, r[2] + nx, r[3] + ny, road_z, e);
        // Reference order is every first triangle, then every second.
        clip_prim(c, a, b, d, col, TRIS_PER_PRIM * i, road_data, road_index, road_cursor,
                  road_capacity, n_road);
        clip_prim(c, a, d, e, col, TRIS_PER_PRIM * (num_roads + i), road_data, road_index,
                  road_cursor, road_capacity, n_road);
    }

    for (int i = threadIdx.x; i < num_agents; i += blockDim.x) {
        // A camera does not see the vehicle it is mounted on.
        if (i == self_index) continue;

        const float *g = agents + (size_t)(agent_lo + i) * RASTER_AGENT_FEATURES;
        float cos_h = g[2], sin_h = g[3];
        float half_l = g[4] * 0.5f, half_w = g[5] * 0.5f, height = g[6];

        float base[3];
        agent_color((int)g[7], base);

        float corners[8][3];
        int k = 0;
        for (int sx = -1; sx <= 1; sx += 2) {
            for (int sy = -1; sy <= 1; sy += 2) {
                for (int sz = 0; sz <= 1; sz++) {
                    float lx = sx * half_l, ly = sy * half_w;
                    to_camera(c, ego, g[0] + lx * cos_h - ly * sin_h, g[1] + lx * sin_h + ly * cos_h,
                              height * sz, corners[k]);
                    k++;
                }
            }
        }
        for (int t = 0; t < 12; t++) {
            float shade = FACE_SHADE[t / 2];
            float col[3] = {base[0] * shade, base[1] * shade, base[2] * shade};
            clip_prim(c, corners[BOX_TRIS[t][0]], corners[BOX_TRIS[t][1]], corners[BOX_TRIS[t][2]],
                      col, TRIS_PER_PRIM * (t * num_agents + i), agent_data, agent_index,
                      agent_cursor, agent_capacity, n_agent);
        }
    }
}

// Pass 1: how many triangles of each layer land on screen, per image.
__global__ void count_kernel(const float *agents, const float *roads, const float *egos,
                             int ego_stride, const float *rig, int num_cams, const int *ego_scene,
                             const int *agent_ranges, const int *road_ranges, int *road_counts,
                             int *agent_counts) {
    int image = blockIdx.x;
    __shared__ int block_road, block_agent;
    if (threadIdx.x == 0) { block_road = 0; block_agent = 0; }
    __syncthreads();

    int n_road = 0, n_agent = 0;
    project_image(image, agents, roads, egos, ego_stride, rig, num_cams, ego_scene, agent_ranges,
                  road_ranges, nullptr, nullptr, nullptr, 0, nullptr, nullptr, nullptr, 0, n_road,
                  n_agent);

    atomicAdd(&block_road, n_road);
    atomicAdd(&block_agent, n_agent);
    __syncthreads();
    if (threadIdx.x == 0) {
        road_counts[image] = block_road;
        agent_counts[image] = block_agent;
    }
}

// Pass 2: write those triangles compacted into the slice the prefix sum assigned.
__global__ void transform_kernel(const float *agents, const float *roads, const float *egos,
                                 int ego_stride, const float *rig, int num_cams, const int *ego_scene,
                                 const int *agent_ranges, const int *road_ranges,
                                 const int *road_offsets, const int *agent_offsets, float *road_data,
                                 int *road_index, float *agent_data, int *agent_index) {
    int image = blockIdx.x;
    __shared__ int road_cursor, agent_cursor;
    if (threadIdx.x == 0) { road_cursor = 0; agent_cursor = 0; }
    __syncthreads();

    int road_base = road_offsets[image];
    int agent_base = agent_offsets[image];
    int road_cap = road_offsets[image + 1] - road_base;
    int agent_cap = agent_offsets[image + 1] - agent_base;

    int n_road = 0, n_agent = 0;
    project_image(image, agents, roads, egos, ego_stride, rig, num_cams, ego_scene, agent_ranges,
                  road_ranges, road_data + (size_t)road_base * TRI_STRIDE, road_index + road_base,
                  &road_cursor, road_cap, agent_data + (size_t)agent_base * TRI_STRIDE,
                  agent_index + agent_base, &agent_cursor, agent_cap, n_road, n_agent);
}

// Analytic edge coverage and interpolated depth for one triangle at one pixel.
// Returns 0 coverage when the pixel is outside the triangle's dilated bounding
// box: for a sliver, the two long edges are nearly collinear, and dilating both
// by the half pixel the coverage ramp uses makes their half-planes overlap in a
// narrow wedge reaching far past the shared vertex. Bounding here also matches
// what a bounding-box rasterizer naturally does.
__device__ __forceinline__ float coverage_depth(const float *t, float px, float py, float *depth_out) {
    if (px < t[12] - 0.5f || px > t[13] + 0.5f || py < t[14] - 0.5f || py > t[15] + 0.5f) return 0.0f;

    float ax = t[0], ay = t[1], bx = t[3], by = t[4], cx = t[6], cy = t[7];
    float area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
    if (fabsf(area) < NEAR_EPS) return 0.0f;
    float sign = area >= 0.0f ? 1.0f : -1.0f;
    float area_abs = fabsf(area);

    float cov = 1.0f;
    float w[3];
    const float px0[3] = {bx, cx, ax};
    const float py0[3] = {by, cy, ay};
    const float px1[3] = {cx, ax, bx};
    const float py1[3] = {cy, ay, by};
#pragma unroll
    for (int i = 0; i < 3; i++) {
        // Edge function (q - p) x (X - p), normalised so the interior is positive
        // whatever the winding.
        float ex = px1[i] - px0[i];
        float ey = py1[i] - py0[i];
        float e = (ex * (py - py0[i]) - ey * (px - px0[i])) * sign;
        w[i] = e / area_abs;
        float elen = fmaxf(sqrtf(ex * ex + ey * ey), NEAR_EPS);
        float ramp = e / elen + 0.5f;
        cov *= fminf(fmaxf(ramp, 0.0f), 1.0f);
        if (cov <= 0.0f) return 0.0f;
    }
    // Perspective-correct depth: screen-space barycentrics interpolate 1/z
    // linearly, not z. Interpolating z directly is only accurate while a triangle
    // covers little depth on screen, and the ground quad the camera stands on runs
    // from the near plane to tens of metres -- shading that from an affine depth
    // painted the haze of its far end across its near end. See
    // raster_ref._coverage_and_depth.
    float inv = w[0] / t[2] + w[1] / t[5] + w[2] / t[8];
    *depth_out = inv > NEAR_EPS ? 1.0f / inv : INFINITY;
    return cov;
}

// Fragment order for the agent layer: depth rounded to DEPTH_ORDER_QUANTUM, ties
// to the lower primitive index. Two faces of a box meet along an edge where their
// depths are equal in exact arithmetic, and ordering those on the raw float
// decides which face shades the edge by rounding -- the CPU reference rounds it
// the other way. See raster_ref.DEPTH_ORDER_QUANTUM.
__device__ __forceinline__ bool frag_after(float da, int ia, float db, int ib) {
    float qa = rintf(da / DEPTH_ORDER_QUANTUM);
    float qb = rintf(db / DEPTH_ORDER_QUANTUM);
    return qa > qb || (qa == qb && ia > ib);
}

__device__ __forceinline__ void insert_fragment(float *fd, float *fc, int *fi, int &count, float depth,
                                                int tri, const float *color, float cov) {
    if (count == MAX_FRAGS && !frag_after(fd[MAX_FRAGS - 1], fi[MAX_FRAGS - 1], depth, tri)) return;
    int pos = count < MAX_FRAGS ? count : MAX_FRAGS - 1;
    while (pos > 0 && frag_after(fd[pos - 1], fi[pos - 1], depth, tri)) {
        fd[pos] = fd[pos - 1];
        fi[pos] = fi[pos - 1];
#pragma unroll
        for (int k = 0; k < 4; k++) fc[pos * 4 + k] = fc[(pos - 1) * 4 + k];
        pos--;
    }
    fd[pos] = depth;
    fi[pos] = tri;
    fc[pos * 4 + 0] = color[0];
    fc[pos * 4 + 1] = color[1];
    fc[pos * 4 + 2] = color[2];
    fc[pos * 4 + 3] = cov;
    if (count < MAX_FRAGS) count++;
}

// Accumulate one road fragment into the surface of its own colour at this pixel.
//
// Coplanar road primitives tile a surface rather than stack on it: two abutting
// lane-area quads split the pixels along their shared joint between them, and so
// do the two triangles each quad is made of. Compositing those with `over` leaves
// 1 - (1 - a)(1 - b) of the pixel to the ground below, drawing a dark seam along
// every joint and every quad diagonal -- which is what made the drivable area read
// as a mosaic of tiles rather than one surface. Adding the coverages is exact
// where the primitives tile and saturates where they overlap, which is what an
// opaque surface wants either way.
//
// The accumulator holds the coverage sum, the coverage-weighted depth sum and the
// lowest primitive index of the colour, in arrival order; finalize_road_fragments
// turns it into fragments. Mirrors raster_ref._merge_coplanar.
__device__ __forceinline__ void accumulate_road_fragment(float *fd, float *fc, int *fi, int &count,
                                                         float depth, int tri, const float *color,
                                                         float cov) {
    for (int p = 0; p < count; p++) {
        if (fc[p * 4 + 0] != color[0] || fc[p * 4 + 1] != color[1] || fc[p * 4 + 2] != color[2])
            continue;
        fc[p * 4 + 3] += cov;
        fd[p] += cov * depth;
        fi[p] = min(fi[p], tri);
        return;
    }
    // The road palette holds fewer colours than there are fragment slots, so this
    // only guards against a palette that outgrows them.
    if (count == MAX_FRAGS) return;
    fd[count] = cov * depth;
    fc[count * 4 + 0] = color[0];
    fc[count * 4 + 1] = color[1];
    fc[count * 4 + 2] = color[2];
    fc[count * 4 + 3] = cov;
    fi[count] = tri;
    count++;
}

// Coverage-weighted mean depth per surface, coverage saturated at one, ordered
// the way composite() expects. The mean depth is exact for coplanar fragments and,
// unlike picking the nearest of them, moves continuously with a coverage that is
// itself only accurate to float rounding.
//
// The order is the primitive order, not the depth order: road fragments are decals
// on one plane, so which of them is on top is a painter's-order decision that
// fill_render_roads already encodes by emitting markings before the lane area they
// are painted on. Their interpolated depths agree to within float noise, and
// sorting on that would settle the order differently here than in the CPU
// reference. See the order_key argument of raster_ref._composite_shaded.
__device__ __forceinline__ void finalize_road_fragments(float *fd, float *fc, int *fi, int count) {
    for (int p = 0; p < count; p++) {
        fd[p] /= fmaxf(fc[p * 4 + 3], NEAR_EPS);
        fc[p * 4 + 3] = fminf(fc[p * 4 + 3], 1.0f);
    }
    // Insertion sort; at most one fragment per road colour reaches here.
    for (int p = 1; p < count; p++) {
        float depth = fd[p], col[4];
#pragma unroll
        for (int k = 0; k < 4; k++) col[k] = fc[p * 4 + k];
        int tri = fi[p];
        int q = p - 1;
        while (q >= 0 && fi[q] > tri) {
            fd[q + 1] = fd[q];
            fi[q + 1] = fi[q];
#pragma unroll
            for (int k = 0; k < 4; k++) fc[(q + 1) * 4 + k] = fc[q * 4 + k];
            q--;
        }
        fd[q + 1] = depth;
        fi[q + 1] = tri;
#pragma unroll
        for (int k = 0; k < 4; k++) fc[(q + 1) * 4 + k] = col[k];
    }
}

// Front-to-back "over" compositing, with aerial perspective per fragment.
// Writes premultiplied colour into acc[0..2] and the surviving transmittance into
// acc[3], so the caller finishes with `colour = acc.rgb + background * acc[3]`.
__device__ __forceinline__ void composite(const float *fd, const float *fc, int count, float *acc) {
    float trans = 1.0f;
    acc[0] = acc[1] = acc[2] = 0.0f;
    for (int i = 0; i < count; i++) {
        float cov = fc[i * 4 + 3];
        float w = cov * trans;
        float scale = depth_scale(fd[i]);
#pragma unroll
        for (int k = 0; k < 3; k++)
            acc[k] += w * (fc[i * 4 + k] * scale + PALETTE_SKY[k] * (1.0f - scale));
        trans *= (1.0f - cov);
    }
    acc[3] = trans;
}

__global__ void raster_kernel(const float *roads, const float *egos, int ego_stride, const float *rig,
                              int num_cams, const int *ego_scene, const int *road_ranges,
                              const int *road_offsets, const int *agent_offsets,
                              const float *road_data, const int *road_index, const float *agent_data,
                              const int *agent_index, unsigned char *out, int tiles_x, int tiles_y) {
    int image = blockIdx.x;
    int tile = blockIdx.y;
    int tile_x = (tile % tiles_x) * TILE_W;
    int tile_y = (tile / tiles_x) * TILE_H;

    int cam_i = image % num_cams;
    Cam c = load_cam(rig, cam_i);

    int scene = ego_scene[image / num_cams];
    int road_lo = road_ranges[scene], road_hi = road_ranges[scene + 1];
    // fill_render_roads emits lane areas last. Their renderer-only tag opts the
    // shared ocean/teddy/giga camera path into black non-road ground.
    bool black_ground = road_hi > road_lo && (int)roads[(size_t)(road_hi - 1) * RASTER_ROAD_FEATURES + 5] == 11;

    int road_base = road_offsets[image];
    int agent_base = agent_offsets[image];
    int road_total = road_offsets[image + 1] - road_base;
    int agent_total = agent_offsets[image + 1] - agent_base;
    const float *r_scratch = road_data + (size_t)road_base * TRI_STRIDE;
    const float *a_scratch = agent_data + (size_t)agent_base * TRI_STRIDE;
    const int *r_index = road_index + road_base;
    const int *a_index = agent_index + agent_base;

    __shared__ int tri_list[TILE_CHUNK];
    __shared__ int n_tri;

    float tu0 = tile_x - 0.5f, tu1 = tile_x + TILE_W + 0.5f;
    float tv0 = tile_y - 0.5f, tv1 = tile_y + TILE_H + 0.5f;

    // One pixel per thread, fixed for the life of the block.
    int lx = threadIdx.x % TILE_W, ly = threadIdx.x / TILE_W;
    int x = tile_x + lx, y = tile_y + ly;
    bool in_frame = (x < c.width && y < c.height);
    float px = x + 0.5f, py = y + 0.5f;

    // Background: sky above the horizon, asphalt below with exact depth from
    // the ray/ground-plane intersection.
    float dcx = (px - c.cx) / c.fx, dcy = (py - c.cy) / c.fy;
    // Camera -> ego is the transpose of the ego -> camera rotation.
    float dez = c.rot[2] * dcx + c.rot[5] * dcy + c.rot[8];
    float ground_depth = INFINITY;
    float rgb[3];
    if (dez < -1e-6f) {
        ground_depth = -c.pos[2] / dez;  // dcz is 1, so the ray parameter is the depth
        float s = depth_scale(ground_depth);
        const float ground[3] = {
            black_ground ? 0.0f : 0.16f,
            black_ground ? 0.0f : 0.16f,
            black_ground ? 0.0f : 0.17f,
        };
#pragma unroll
        for (int k = 0; k < 3; k++) rgb[k] = ground[k] * s + PALETTE_SKY[k] * (1.0f - s);
    } else {
#pragma unroll
        for (int k = 0; k < 3; k++) rgb[k] = PALETTE_SKY[k];
    }

    float fd[MAX_FRAGS];
    float fc[MAX_FRAGS * 4];
    int fi[MAX_FRAGS];

    // Layer 1: road markings. Coplanar with the ground, so drawn over it rather
    // than depth-tested against it. The fragment set survives the chunk loop, so
    // splitting the triangle list changes nothing about what it ends up holding.
    int count = 0;
    for (int base = 0; base < road_total; base += TILE_CHUNK) {
        int hi = min(base + TILE_CHUNK, road_total);
        if (threadIdx.x == 0) n_tri = 0;
        __syncthreads();
        for (int i = base + threadIdx.x; i < hi; i += blockDim.x) {
            const float *t = r_scratch + (size_t)i * TRI_STRIDE;
            if (t[13] >= tu0 && t[12] <= tu1 && t[15] >= tv0 && t[14] <= tv1)
                tri_list[atomicAdd(&n_tri, 1)] = i;
        }
        __syncthreads();
        int nr = n_tri;
        if (in_frame) {
            for (int i = 0; i < nr; i++) {
                int local = tri_list[i];
                const float *t = r_scratch + (size_t)local * TRI_STRIDE;
                float depth;
                float cov = coverage_depth(t, px, py, &depth);
                if (cov <= 0.0f || depth <= 0.0f || depth > c.far_z) continue;
                accumulate_road_fragment(fd, fc, fi, count, depth, r_index[local], t + 9, cov);
            }
        }
        __syncthreads();
    }
    if (count) {
        finalize_road_fragments(fd, fc, fi, count);
        float acc[4] = {0.0f, 0.0f, 0.0f, 1.0f};
        composite(fd, fc, count, acc);
#pragma unroll
        for (int k = 0; k < 3; k++) rgb[k] = acc[k] + rgb[k] * acc[3];
    }

    // Layer 2: agent boxes, bounded by the ground intercept. A box farther than
    // the ground point seen through this pixel cannot be visible here.
    count = 0;
    for (int base = 0; base < agent_total; base += TILE_CHUNK) {
        int hi = min(base + TILE_CHUNK, agent_total);
        if (threadIdx.x == 0) n_tri = 0;
        __syncthreads();
        for (int i = base + threadIdx.x; i < hi; i += blockDim.x) {
            const float *t = a_scratch + (size_t)i * TRI_STRIDE;
            if (t[13] >= tu0 && t[12] <= tu1 && t[15] >= tv0 && t[14] <= tv1)
                tri_list[atomicAdd(&n_tri, 1)] = i;
        }
        __syncthreads();
        int na = n_tri;
        if (in_frame) {
            for (int i = 0; i < na; i++) {
                int local = tri_list[i];
                const float *t = a_scratch + (size_t)local * TRI_STRIDE;
                float depth;
                float cov = coverage_depth(t, px, py, &depth);
                if (cov <= 0.0f || depth <= 0.0f || depth > c.far_z ||
                    depth > ground_depth * (1.0f + GROUND_DEPTH_SLACK))
                    continue;
                insert_fragment(fd, fc, fi, count, depth, a_index[local], t + 9, cov);
            }
        }
        __syncthreads();
    }
    if (count) {
        float acc[4] = {0.0f, 0.0f, 0.0f, 1.0f};
        composite(fd, fc, count, acc);
#pragma unroll
        for (int k = 0; k < 3; k++) rgb[k] = acc[k] + rgb[k] * acc[3];
    }

    if (!in_frame) return;
    size_t plane = (size_t)c.width * c.height;
    size_t base = (size_t)image * 3 * plane + (size_t)y * c.width + x;
#pragma unroll
    for (int k = 0; k < 3; k++) {
        float v = fminf(fmaxf(rgb[k], 0.0f), 1.0f) * 255.0f + 0.5f;
        out[base + k * plane] = (unsigned char)v;
    }
}

void drive_raster_cuda(torch::Tensor agents, torch::Tensor roads, torch::Tensor egos, torch::Tensor rig,
                       torch::Tensor ego_scene, torch::Tensor agent_ranges, torch::Tensor road_ranges,
                       torch::Tensor out) {
    TORCH_CHECK(agents.is_cuda() && roads.is_cuda() && egos.is_cuda() && rig.is_cuda() && out.is_cuda(),
                "All tensors must be on the GPU");
    TORCH_CHECK(ego_scene.is_cuda() && agent_ranges.is_cuda() && road_ranges.is_cuda(),
                "Scene index tensors must be on the GPU");
    TORCH_CHECK(ego_scene.scalar_type() == torch::kInt32 && agent_ranges.scalar_type() == torch::kInt32 &&
                    road_ranges.scalar_type() == torch::kInt32,
                "Scene index tensors must be int32");
    TORCH_CHECK(ego_scene.is_contiguous() && agent_ranges.is_contiguous() && road_ranges.is_contiguous(),
                "Scene index tensors must be contiguous");
    TORCH_CHECK(agent_ranges.size(0) == road_ranges.size(0), "Range tensors must agree on scene count");
    TORCH_CHECK(agents.scalar_type() == torch::kFloat32 && roads.scalar_type() == torch::kFloat32 &&
                    egos.scalar_type() == torch::kFloat32 && rig.scalar_type() == torch::kFloat32,
                "Scene tensors must be float32");
    TORCH_CHECK(out.scalar_type() == torch::kUInt8, "Output must be uint8");
    TORCH_CHECK(agents.is_contiguous() && roads.is_contiguous() && egos.is_contiguous() &&
                    rig.is_contiguous() && out.is_contiguous(),
                "All tensors must be contiguous");
    TORCH_CHECK(agents.dim() == 2 && agents.size(1) == RASTER_AGENT_FEATURES, "agents must be [A, 8]");
    TORCH_CHECK(roads.dim() == 2 && roads.size(1) == RASTER_ROAD_FEATURES, "roads must be [R, 6]");
    TORCH_CHECK(egos.dim() == 2 && egos.size(1) >= 4, "egos must be [E, 4] or [E, 5]");
    TORCH_CHECK(rig.dim() == 2 && rig.size(1) == RASTER_RIG_STRIDE, "rig must be [C, 20]");

    int num_egos = egos.size(0);
    int ego_stride = egos.size(1);
    int num_cams = rig.size(0);
    int num_images = num_egos * num_cams;
    TORCH_CHECK(out.dim() == 5 && out.size(0) == num_egos && out.size(1) == num_cams && out.size(2) == 3,
                "out must be [E, C, 3, H, W]");
    int height = out.size(3), width = out.size(4);
    TORCH_CHECK(ego_scene.size(0) == num_egos, "ego_scene must have one entry per ego");

    auto opts_f = torch::TensorOptions().dtype(torch::kFloat32).device(agents.device());
    auto opts_i = torch::TensorOptions().dtype(torch::kInt32).device(agents.device());

    const float *agents_p = agents.data_ptr<float>();
    const float *roads_p = roads.data_ptr<float>();
    const float *egos_p = egos.data_ptr<float>();
    const float *rig_p = rig.data_ptr<float>();
    const int *scene_p = ego_scene.data_ptr<int>();
    const int *ar_p = agent_ranges.data_ptr<int>();
    const int *rr_p = road_ranges.data_ptr<int>();

    // Pass 1: count the on-screen triangles each image will produce, so its
    // scratch slice can be sized exactly rather than padded to the largest scene
    // in the batch.
    auto counts = torch::empty({2, num_images}, opts_i);
    count_kernel<<<num_images, TRANSFORM_THREADS>>>(agents_p, roads_p, egos_p, ego_stride, rig_p,
                                                    num_cams, scene_p, ar_p, rr_p,
                                                    counts[0].data_ptr<int>(),
                                                    counts[1].data_ptr<int>());

    // Exclusive prefix sum per layer, with the total in the last slot.
    auto offsets = torch::zeros({2, num_images + 1}, opts_i);
    offsets.slice(1, 1, num_images + 1).copy_(counts.cumsum(1).to(torch::kInt32));
    // The only host round trip left, and it moves eight bytes: the allocator needs
    // the totals. The scene ranges used to be copied back here as well, purely to
    // recover a maximum the caller already knew.
    auto totals = offsets.select(1, num_images).to(torch::kCPU);
    int64_t total_road = std::max<int64_t>(totals[0].item<int>(), 1);
    int64_t total_agent = std::max<int64_t>(totals[1].item<int>(), 1);

    auto road_data = torch::empty({total_road * TRI_STRIDE}, opts_f);
    auto road_index = torch::empty({total_road}, opts_i);
    auto agent_data = torch::empty({total_agent * TRI_STRIDE}, opts_f);
    auto agent_index = torch::empty({total_agent}, opts_i);

    const int *road_off = offsets[0].data_ptr<int>();
    const int *agent_off = offsets[1].data_ptr<int>();

    transform_kernel<<<num_images, TRANSFORM_THREADS>>>(
        agents_p, roads_p, egos_p, ego_stride, rig_p, num_cams, scene_p, ar_p, rr_p, road_off,
        agent_off, road_data.data_ptr<float>(), road_index.data_ptr<int>(),
        agent_data.data_ptr<float>(), agent_index.data_ptr<int>());

    int tiles_x = (width + TILE_W - 1) / TILE_W;
    int tiles_y = (height + TILE_H - 1) / TILE_H;
    dim3 grid(num_images, tiles_x * tiles_y);

    raster_kernel<<<grid, RASTER_THREADS>>>(
        roads_p, egos_p, ego_stride, rig_p, num_cams, scene_p, rr_p, road_off, agent_off,
        road_data.data_ptr<float>(), road_index.data_ptr<int>(), agent_data.data_ptr<float>(),
        agent_index.data_ptr<int>(), out.data_ptr<unsigned char>(), tiles_x, tiles_y);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

TORCH_LIBRARY_IMPL(pufferlib, CUDA, m) {
    m.impl("drive_raster", &drive_raster_cuda);
}

}  // namespace pufferlib
