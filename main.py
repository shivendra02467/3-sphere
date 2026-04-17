import pyray as pr
import numpy as np
import time

fragment_shader = """
#version 120
uniform vec4 cameraPos;
uniform vec4 forward;
uniform vec4 right;
uniform vec4 up;
uniform vec2 iResolution;
uniform float iTime;

const int NUM_SPHERES = 3;
const float EPS = 1e-4;
const float MAX_T = 20.0;
const float TWO_PI = 6.283185307179586;

vec4 retract(vec4 x, vec4 v) {
    float lv = length(v);
    if(lv > 0.0) return cos(lv)*x + (sin(lv)/lv)*v;
    return x;
}

float norm2pi(float a) {
    float r = mod(a, TWO_PI);
    return (r < 0.0) ? r + TWO_PI : r;
}

float intersect_sphere_exact(vec4 cam, vec4 u, vec4 c, float R) {
    float A = dot(cam, c);
    float B = dot(u, c);
    float C = cos(R);
    float D = sqrt(A*A + B*B);
    if(D < 1e-6) return 1e6;
    float ratio = C / D;
    if(ratio < -1.0 || ratio > 1.0) return 1e6;
    float phi = atan(B, A);
    float alpha = acos(clamp(ratio, -1.0, 1.0));
    float t1 = norm2pi(phi + alpha);
    float t2 = norm2pi(phi - alpha);

    float best = 1e6;
    int K = int(ceil(MAX_T / TWO_PI)) + 1;
    for(int k = 0; k <= K; ++k) {
        float cand1 = t1 + float(k) * TWO_PI;
        if(cand1 > EPS && cand1 <= MAX_T) best = min(best, cand1);
        float cand2 = t2 + float(k) * TWO_PI;
        if(cand2 > EPS && cand2 <= MAX_T) best = min(best, cand2);
    }
    return best;
}

void main() {
    vec2 frag = gl_FragCoord.xy;
    vec2 res = iResolution.xy;
    vec2 uv = (frag - 0.5*res) / min(res.x, res.y);

    vec4 cam = normalize(cameraPos);
    vec4 fwd = forward;
    vec4 rgt = right;
    vec4 upv = up;

    vec4 dir = normalize(fwd + uv.x * rgt + uv.y * upv);

    vec4 tpos[NUM_SPHERES];
    tpos[0] = vec4(0.0, 0.7, 0.7, 0.0);
    tpos[1] = vec4(0.0, 1.0, 0.0, 0.0);
    tpos[2] = vec4(0.0, 0.0, 1.0, 0.0);

    float R[NUM_SPHERES];
    R[0] = 0.4; R[1] = 0.4; R[2] = 0.4;

    float bestT = 1e6;
    int bestId = -1;
    for(int i=0; i<NUM_SPHERES; ++i) {
        float t = intersect_sphere_exact(cam, dir, tpos[i], R[i]);
        if(t < bestT) {
            bestT = t;
            bestId = i;
        }
    }

    vec3 color = vec3(0.0);
    if(bestId >= 0 && bestT < 1e5) {
        vec4 x = retract(cam, dir * bestT);
        float cosR = cos(R[bestId]);
        vec4 c = tpos[bestId];
        vec4 n = c - x * dot(x, c);
        float ln = length(n);
        if(ln < 1e-6) n = normalize(c - x * cosR);
        else n = normalize(n);

        vec4 light_world = vec4(0.3, 0.7, 0.2, 0.1);
        vec4 light_t = light_world - x * dot(x, light_world);
        light_t = normalize(light_t);
        float lam = max(0.0, dot(n, light_t));

        if(bestId == 0) color = vec3(0.9, 0.3, 0.2) * (0.18 + 0.82 * lam);
        else if(bestId == 1) color = vec3(0.2, 0.9, 0.3) * (0.18 + 0.82 * lam);
        else color = vec3(0.2, 0.3, 0.9) * (0.18 + 0.82 * lam);

        float rim = 1.0 - max(0.0, dot(n, fwd));
        color += 0.12 * vec3(1.0) * pow(rim, 2.5);
    } else {
        color = vec3(0.02, 0.02, 0.04) + 0.35 * vec3(abs(dir.x), abs(dir.y), abs(dir.z));
    }

    gl_FragColor = vec4(color, 1.0);
}
"""

def retract(x, v):
    lv = np.linalg.norm(v)
    if lv > 0.0:
        return np.cos(lv) * x + (np.sin(lv) / lv) * v
    return x.copy()

def parallel_transport(x, v, u, t=1.0):
    lv = np.linalg.norm(v)
    if lv == 0.0:
        return u.copy()
    vdotv = np.dot(v, v)
    if vdotv == 0.0:
        return u.copy()
    vdotu = np.dot(v, u)
    term1 = u + v * (vdotu / vdotv) * (np.cos(t*lv) - 1.0)
    term2 = x * (np.sin(t*lv) * vdotu / lv)
    return term1 - term2

def project_to_tangent(x, u):
    return u - x * np.dot(x, u)

def normalize_safe(u):
    n = np.linalg.norm(u)
    if n == 0.0:
        return u
    return u / n

def orthonormalize_at(x, vectors):
    outs = []
    for v in vectors:
        vt = project_to_tangent(x, v)
        for prev in outs:
            vt -= prev * np.dot(prev, vt)
        vt = normalize_safe(vt)
        outs.append(vt)
    return outs

def main():
    width, height = 900, 600
    pr.init_window(width, height, "S^3")
    pr.set_target_fps(60)

    shader = pr.load_shader_from_memory(pr.ffi.NULL, fragment_shader)

    loc_res = pr.get_shader_location(shader, "iResolution")
    loc_time = pr.get_shader_location(shader, "iTime")
    loc_cam = pr.get_shader_location(shader, "cameraPos")
    loc_fwd = pr.get_shader_location(shader, "forward")
    loc_right = pr.get_shader_location(shader, "right")
    loc_up = pr.get_shader_location(shader, "up")

    cameraPos = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    f = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
    r = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)
    u = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    f = normalize_safe(project_to_tangent(cameraPos, f))
    r = project_to_tangent(cameraPos, r)
    r = r - f * np.dot(f, r)
    r = normalize_safe(r)
    u = project_to_tangent(cameraPos, u) - f*np.dot(f, u) - r*np.dot(r, u)
    u = normalize_safe(u)

    cameraDirs = [f, r, u]

    moveSpeed = 1.2
    mouse_sensitivity = 0.004
    t0 = time.perf_counter()

    while not pr.window_should_close():
        dt = pr.get_frame_time()
        dt = min(dt, 0.05)
        current_time = time.perf_counter() - t0

        if pr.is_mouse_button_down(pr.MOUSE_BUTTON_LEFT):
            delta = pr.get_mouse_delta()
            dx, dy = delta.x, delta.y

            yaw = -dx * mouse_sensitivity
            pitch = dy * mouse_sensitivity

            f, r, u = cameraDirs[0], cameraDirs[1], cameraDirs[2]

            ca, sa = np.cos(yaw), np.sin(yaw)
            f1 = ca*f + sa*r
            r1 = -sa*f + ca*r

            cb, sb = np.cos(pitch), np.sin(pitch)
            f2 = cb*f1 + sb*u
            u2 = -sb*f1 + cb*u

            f2 = normalize_safe(project_to_tangent(cameraPos, f2))
            r2 = normalize_safe(project_to_tangent(cameraPos, r1))
            r2 = r2 - f2 * np.dot(f2, r2)
            r2 = normalize_safe(r2)
            u2 = project_to_tangent(cameraPos, u2)
            u2 = u2 - f2*np.dot(f2, u2) - r2*np.dot(r2, u2)
            u2 = normalize_safe(u2)

            cameraDirs = [f2, r2, u2]

        local_disp = np.zeros(4, dtype=np.float64)
        speed = moveSpeed * dt

        if pr.is_key_down(pr.KEY_W): local_disp += cameraDirs[0] * speed
        if pr.is_key_down(pr.KEY_S): local_disp -= cameraDirs[0] * speed
        if pr.is_key_down(pr.KEY_D): local_disp += cameraDirs[1] * speed
        if pr.is_key_down(pr.KEY_A): local_disp -= cameraDirs[1] * speed
        if pr.is_key_down(pr.KEY_E): local_disp += cameraDirs[2] * speed
        if pr.is_key_down(pr.KEY_Q): local_disp -= cameraDirs[2] * speed

        if not np.allclose(local_disp, 0.0):
            old_pos = cameraPos.copy()
            new_dirs = [parallel_transport(old_pos, local_disp, d, 1.0) for d in cameraDirs]
            new_pos = retract(old_pos, local_disp)

            new_dirs = orthonormalize_at(new_pos, new_dirs)
            cameraPos = normalize_safe(new_pos)
            cameraDirs = new_dirs

        time_ptr = pr.ffi.new("float *", current_time)
        pr.set_shader_value(shader, loc_time, time_ptr, pr.SHADER_UNIFORM_FLOAT)

        res_ptr = pr.ffi.new("float[2]", [float(width), float(height)])
        pr.set_shader_value(shader, loc_res, res_ptr, pr.SHADER_UNIFORM_VEC2)

        cam_ptr = pr.ffi.new("float[4]", cameraPos.astype(np.float32).tolist())
        pr.set_shader_value(shader, loc_cam, cam_ptr, pr.SHADER_UNIFORM_VEC4)

        fwd_ptr = pr.ffi.new("float[4]", cameraDirs[0].astype(np.float32).tolist())
        pr.set_shader_value(shader, loc_fwd, fwd_ptr, pr.SHADER_UNIFORM_VEC4)

        rgt_ptr = pr.ffi.new("float[4]", cameraDirs[1].astype(np.float32).tolist())
        pr.set_shader_value(shader, loc_right, rgt_ptr, pr.SHADER_UNIFORM_VEC4)

        up_ptr = pr.ffi.new("float[4]", cameraDirs[2].astype(np.float32).tolist())
        pr.set_shader_value(shader, loc_up, up_ptr, pr.SHADER_UNIFORM_VEC4)

        pr.begin_drawing()
        pr.clear_background(pr.BLACK)

        pr.begin_shader_mode(shader)
        pr.draw_rectangle(0, 0, width, height, pr.WHITE)
        pr.end_shader_mode()

        pr.end_drawing()

    pr.unload_shader(shader)
    pr.close_window()

if __name__ == '__main__':
    main()
