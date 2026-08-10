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
//   1. transform_kernel bins nothing and simply projects every primitive into
//      screen space once, writing triangles to a scratch buffer. Scenes hold on
//      the order of a thousand segments, so this is cheap and avoids
//      re-transforming per pixel.
//   2. raster_kernel takes one tile of one image per block. Threads cooperatively
//      compact the triangles whose bounding box meets the tile into shared
//      memory, then each thread shades its own pixels against that short list.
//
// Visibility follows the reference's three layers: an analytic ray/ground-plane
// background, road markings drawn over it (coplanar, so no depth test), and agent
// boxes depth-tested against the ground intercept. Within a layer, fragments are
// combined front to back with analytic edge coverage as alpha.

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

// Screen-space tile handled by one block.
#define TILE_W 32
#define TILE_H 16
#define MAX_TILE_TRIS 4096

// Fragments retained per pixel per layer, nearest first. Front-to-back
// compositing terminates once alpha saturates, so this only truncates when many
// partially covering fragments stack up at one pixel.
#define MAX_FRAGS 16

#define PALETTE_DEPTH_FALLOFF 60.0f
#define PALETTE_DEPTH_MIN_SCALE 0.30f
#define NEAR_EPS 1e-9f

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

// Emit one screen-space triangle, near-clipping by rejection. Triangles that
// straddle the near plane are dropped rather than re-cut; at the 0.15 m near
// plane used here that only affects geometry essentially inside the camera.
// Slots are assigned by primitive index rather than by an atomic counter, so the
// triangle ordering is deterministic and identical to the reference's. Ties in
// fragment depth are then broken by that index, which makes the composited result
// independent of the order threads happen to visit primitives in.
__device__ __forceinline__ void emit_triangle(float *scratch, int slot, const Cam &c, const float *p0,
                                              const float *p1, const float *p2, const float *color) {
    float *t = scratch + (size_t)slot * TRI_STRIDE;
    // An inverted bounding box marks the slot empty; the tile test rejects it.
    if (p0[2] < c.near_z || p1[2] < c.near_z || p2[2] < c.near_z) {
        t[12] = 1.0f; t[13] = -1.0f; t[14] = 1.0f; t[15] = -1.0f;
        return;
    }

    float u0 = c.fx * p0[0] / p0[2] + c.cx, v0 = c.fy * p0[1] / p0[2] + c.cy;
    float u1 = c.fx * p1[0] / p1[2] + c.cx, v1 = c.fy * p1[1] / p1[2] + c.cy;
    float u2 = c.fx * p2[0] / p2[2] + c.cx, v2 = c.fy * p2[1] / p2[2] + c.cy;

    float umin = fminf(u0, fminf(u1, u2)), umax = fmaxf(u0, fmaxf(u1, u2));
    float vmin = fminf(v0, fminf(v1, v2)), vmax = fmaxf(v0, fmaxf(v1, v2));
    if (umax < -0.5f || umin > c.width + 0.5f || vmax < -0.5f || vmin > c.height + 0.5f) {
        t[12] = 1.0f; t[13] = -1.0f; t[14] = 1.0f; t[15] = -1.0f;
        return;
    }

    t[0] = u0; t[1] = v0; t[2] = p0[2];
    t[3] = u1; t[4] = v1; t[5] = p1[2];
    t[6] = u2; t[7] = v2; t[8] = p2[2];
    t[9] = color[0]; t[10] = color[1]; t[11] = color[2];
    t[12] = umin; t[13] = umax; t[14] = vmin; t[15] = vmax;
}

// One block per (image, layer). Projects that layer's primitives into scratch.
// Egos are grouped into scenes; each scene owns a contiguous slice of the agent
// and road arrays. One launch therefore covers every environment in the batch,
// which matters because a training step holds on the order of a hundred of them
// and per-scene launches would be dominated by their own overhead.
__global__ void transform_kernel(const float *agents, const float *roads, const float *egos,
                                 int ego_stride, const float *rig, int num_cams, const int *ego_scene,
                                 const int *agent_ranges, const int *road_ranges, float *road_scratch,
                                 int max_road_tris, float *agent_scratch, int max_agent_tris) {
    int image = blockIdx.x;
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

    float *r_scratch = road_scratch + (size_t)image * max_road_tris * TRI_STRIDE;
    float *a_scratch = agent_scratch + (size_t)image * max_agent_tris * TRI_STRIDE;

    // Slots past this scene's primitive count are marked empty so the tile scan,
    // which walks a fixed stride, skips them.
    for (int i = num_roads * 2 + threadIdx.x; i < max_road_tris; i += blockDim.x) {
        float *t = r_scratch + (size_t)i * TRI_STRIDE;
        t[12] = 1.0f; t[13] = -1.0f; t[14] = 1.0f; t[15] = -1.0f;
    }
    for (int i = num_agents * 12 + threadIdx.x; i < max_agent_tris; i += blockDim.x) {
        float *t = a_scratch + (size_t)i * TRI_STRIDE;
        t[12] = 1.0f; t[13] = -1.0f; t[14] = 1.0f; t[15] = -1.0f;
    }

    for (int i = threadIdx.x; i < num_roads; i += blockDim.x) {
        const float *r = roads + (size_t)(road_lo + i) * RASTER_ROAD_FEATURES;
        float half_w = r[4] * 0.5f;
        float dx = r[2] - r[0], dy = r[3] - r[1];
        float len = fmaxf(sqrtf(dx * dx + dy * dy), 1e-6f);
        // Left normal of the segment direction, giving the painted strip width.
        float nx = -dy / len * half_w, ny = dx / len * half_w;

        float col[3];
        road_color((int)r[5], col);

        float a[3], b[3], d[3], e[3];
        to_camera(c, ego, r[0] + nx, r[1] + ny, 0.0f, a);
        to_camera(c, ego, r[0] - nx, r[1] - ny, 0.0f, b);
        to_camera(c, ego, r[2] - nx, r[3] - ny, 0.0f, d);
        to_camera(c, ego, r[2] + nx, r[3] + ny, 0.0f, e);
        // Reference order is every first triangle, then every second.
        emit_triangle(r_scratch, i, c, a, b, d, col);
        emit_triangle(r_scratch, num_roads + i, c, a, d, e, col);
    }

    for (int i = threadIdx.x; i < num_agents; i += blockDim.x) {
        // A camera does not see the vehicle it is mounted on. The slots are still
        // written, marked empty, so indices stay aligned with the reference.
        if (i == self_index) {
            for (int t = 0; t < 12; t++) {
                float *slot = a_scratch + (size_t)(t * num_agents + i) * TRI_STRIDE;
                slot[12] = 1.0f; slot[13] = -1.0f; slot[14] = 1.0f; slot[15] = -1.0f;
            }
            continue;
        }
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
            emit_triangle(a_scratch, t * num_agents + i, c, corners[BOX_TRIS[t][0]],
                          corners[BOX_TRIS[t][1]], corners[BOX_TRIS[t][2]], col);
        }
    }
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
    *depth_out = w[0] * t[2] + w[1] * t[5] + w[2] * t[8];
    return cov;
}

// Insert a fragment into a per-pixel list kept sorted nearest-first.
__device__ __forceinline__ bool frag_after(float da, int ia, float db, int ib) {
    return da > db || (da == db && ia > ib);
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

__global__ void raster_kernel(const float *egos, int ego_stride, const float *rig, int num_cams,
                              const int *ego_scene, const int *agent_ranges, const int *road_ranges,
                              const float *road_scratch, const float *agent_scratch, int max_road_tris,
                              int max_agent_tris, unsigned char *out, int tiles_x, int tiles_y) {
    int image = blockIdx.x;
    int tile = blockIdx.y;
    int tile_x = (tile % tiles_x) * TILE_W;
    int tile_y = (tile / tiles_x) * TILE_H;

    int cam_i = image % num_cams;
    Cam c = load_cam(rig, cam_i);

    int scene = ego_scene[image / num_cams];
    int road_total = (road_ranges[scene + 1] - road_ranges[scene]) * 2;
    int agent_total = (agent_ranges[scene + 1] - agent_ranges[scene]) * 12;

    extern __shared__ int shared_idx[];
    int *road_list = shared_idx;
    int *agent_list = shared_idx + MAX_TILE_TRIS;
    __shared__ int n_road, n_agent;
    if (threadIdx.x == 0) { n_road = 0; n_agent = 0; }
    __syncthreads();

    float tu0 = tile_x - 0.5f, tu1 = tile_x + TILE_W + 0.5f;
    float tv0 = tile_y - 0.5f, tv1 = tile_y + TILE_H + 0.5f;

    const float *r_scratch = road_scratch + (size_t)image * max_road_tris * TRI_STRIDE;
    const float *a_scratch = agent_scratch + (size_t)image * max_agent_tris * TRI_STRIDE;

    // Compact the triangles that touch this tile, preserving scratch order so the
    // result does not depend on scheduling.
    for (int i = threadIdx.x; i < road_total; i += blockDim.x) {
        const float *t = r_scratch + (size_t)i * TRI_STRIDE;
        if (t[13] >= tu0 && t[12] <= tu1 && t[15] >= tv0 && t[14] <= tv1) {
            int s = atomicAdd(&n_road, 1);
            if (s < MAX_TILE_TRIS) road_list[s] = i;
        }
    }
    for (int i = threadIdx.x; i < agent_total; i += blockDim.x) {
        const float *t = a_scratch + (size_t)i * TRI_STRIDE;
        if (t[13] >= tu0 && t[12] <= tu1 && t[15] >= tv0 && t[14] <= tv1) {
            int s = atomicAdd(&n_agent, 1);
            if (s < MAX_TILE_TRIS) agent_list[s] = i;
        }
    }
    __syncthreads();
    int nr = min(n_road, MAX_TILE_TRIS);
    int na = min(n_agent, MAX_TILE_TRIS);

    int pixels = TILE_W * TILE_H;
    for (int p = threadIdx.x; p < pixels; p += blockDim.x) {
        int lx = p % TILE_W, ly = p / TILE_W;
        int x = tile_x + lx, y = tile_y + ly;
        if (x >= c.width || y >= c.height) continue;
        float px = x + 0.5f, py = y + 0.5f;

        // Background: sky above the horizon, asphalt below with exact depth from
        // the ray/ground-plane intersection.
        float dcx = (px - c.cx) / c.fx, dcy = (py - c.cy) / c.fy;
        // Camera -> ego is the transpose of the ego -> camera rotation.
        float dez = c.rot[2] * dcx + c.rot[5] * dcy + c.rot[8];
        float ground_depth = INFINITY;
        float rgb[4];
        if (dez < -1e-6f) {
            ground_depth = -c.pos[2] / dez;  // dcz is 1, so the ray parameter is the depth
            float s = depth_scale(ground_depth);
            const float ground[3] = {0.16f, 0.16f, 0.17f};
#pragma unroll
            for (int k = 0; k < 3; k++) rgb[k] = ground[k] * s + PALETTE_SKY[k] * (1.0f - s);
        } else {
#pragma unroll
            for (int k = 0; k < 3; k++) rgb[k] = PALETTE_SKY[k];
        }

        float fd[MAX_FRAGS];
        float fc[MAX_FRAGS * 4];
        int fi[MAX_FRAGS];

        // Layer 1: road markings. Coplanar with the ground, so drawn over it
        // rather than depth-tested against it.
        int count = 0;
        for (int i = 0; i < nr; i++) {
            const float *t = r_scratch + (size_t)road_list[i] * TRI_STRIDE;
            float depth;
            float cov = coverage_depth(t, px, py, &depth);
            if (cov <= 0.0f || depth <= 0.0f || depth > c.far_z) continue;
            insert_fragment(fd, fc, fi, count, depth, road_list[i], t + 9, cov);
        }
        if (count) {
            float acc[4] = {0.0f, 0.0f, 0.0f, 1.0f};
            composite(fd, fc, count, acc);
#pragma unroll
            for (int k = 0; k < 3; k++) rgb[k] = acc[k] + rgb[k] * acc[3];
        }

        // Layer 2: agent boxes, bounded by the ground intercept. A box farther
        // than the ground point seen through this pixel cannot be visible here.
        count = 0;
        for (int i = 0; i < na; i++) {
            const float *t = a_scratch + (size_t)agent_list[i] * TRI_STRIDE;
            float depth;
            float cov = coverage_depth(t, px, py, &depth);
            if (cov <= 0.0f || depth <= 0.0f || depth > c.far_z || depth > ground_depth) continue;
            insert_fragment(fd, fc, fi, count, depth, agent_list[i], t + 9, cov);
        }
        if (count) {
            float acc[4] = {0.0f, 0.0f, 0.0f, 1.0f};
            composite(fd, fc, count, acc);
#pragma unroll
            for (int k = 0; k < 3; k++) rgb[k] = acc[k] + rgb[k] * acc[3];
        }

        size_t plane = (size_t)c.width * c.height;
        size_t base = (size_t)image * 3 * plane + (size_t)y * c.width + x;
#pragma unroll
        for (int k = 0; k < 3; k++) {
            float v = fminf(fmaxf(rgb[k], 0.0f), 1.0f) * 255.0f + 0.5f;
            out[base + k * plane] = (unsigned char)v;
        }
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

    int num_agents = agents.size(0);
    int num_roads = roads.size(0);
    int num_egos = egos.size(0);
    int ego_stride = egos.size(1);
    int num_cams = rig.size(0);
    int num_images = num_egos * num_cams;
    TORCH_CHECK(out.dim() == 5 && out.size(0) == num_egos && out.size(1) == num_cams && out.size(2) == 3,
                "out must be [E, C, 3, H, W]");
    int height = out.size(3), width = out.size(4);

    auto opts_f = torch::TensorOptions().dtype(torch::kFloat32).device(agents.device());
    auto opts_i = torch::TensorOptions().dtype(torch::kInt32).device(agents.device());

    TORCH_CHECK(ego_scene.size(0) == num_egos, "ego_scene must have one entry per ego");

    // Scratch is sized by the largest scene in the batch, since every image walks
    // a fixed stride.
    auto ar = agent_ranges.to(torch::kCPU);
    auto rr = road_ranges.to(torch::kCPU);
    const int *arp = ar.data_ptr<int>();
    const int *rrp = rr.data_ptr<int>();
    int num_scenes = ar.size(0) - 1;
    int max_scene_agents = 0, max_scene_roads = 0;
    for (int i = 0; i < num_scenes; i++) {
        max_scene_agents = std::max(max_scene_agents, arp[i + 1] - arp[i]);
        max_scene_roads = std::max(max_scene_roads, rrp[i + 1] - rrp[i]);
    }
    int max_road_tris = std::max(max_scene_roads * 2, 1);
    int max_agent_tris = std::max(max_scene_agents * 12, 1);
    (void)opts_i;
    (void)num_roads;
    (void)num_agents;

    auto road_scratch = torch::empty({(int64_t)num_images * max_road_tris * TRI_STRIDE}, opts_f);
    auto agent_scratch = torch::empty({(int64_t)num_images * max_agent_tris * TRI_STRIDE}, opts_f);
    transform_kernel<<<num_images, 128>>>(
        agents.data_ptr<float>(), roads.data_ptr<float>(), egos.data_ptr<float>(), ego_stride,
        rig.data_ptr<float>(), num_cams, ego_scene.data_ptr<int>(), agent_ranges.data_ptr<int>(),
        road_ranges.data_ptr<int>(), road_scratch.data_ptr<float>(), max_road_tris,
        agent_scratch.data_ptr<float>(), max_agent_tris);

    int tiles_x = (width + TILE_W - 1) / TILE_W;
    int tiles_y = (height + TILE_H - 1) / TILE_H;
    dim3 grid(num_images, tiles_x * tiles_y);
    size_t shmem = 2 * MAX_TILE_TRIS * sizeof(int);

    raster_kernel<<<grid, 128, shmem>>>(
        egos.data_ptr<float>(), ego_stride, rig.data_ptr<float>(), num_cams, ego_scene.data_ptr<int>(),
        agent_ranges.data_ptr<int>(), road_ranges.data_ptr<int>(), road_scratch.data_ptr<float>(),
        agent_scratch.data_ptr<float>(), max_road_tris, max_agent_tris, out.data_ptr<unsigned char>(),
        tiles_x, tiles_y);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

TORCH_LIBRARY_IMPL(pufferlib, CUDA, m) {
    m.impl("drive_raster", &drive_raster_cuda);
}

}  // namespace pufferlib
