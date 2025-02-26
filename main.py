import vispy
vispy.use('pyqt5')
import numpy as np
from vispy import app, gloo
import time

vertex_shader = """
#version 120
attribute vec2 a_position;
void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
}
"""

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

class Canvas(app.Canvas):
    def __init__(self):
        app.Canvas.__init__(self, keys='interactive', size=(900, 600), title='S^3')
        self.program = gloo.Program(vertex_shader, fragment_shader)

        verts = np.array([[-1.0, -1.0],
                          [-1.0,  1.0],
                          [ 1.0, -1.0],
                          [ 1.0,  1.0]], dtype=np.float32)
        self.vbo = gloo.VertexBuffer(verts)
        self.program['a_position'] = self.vbo

        self.program['iResolution'] = np.array(self.size, dtype=np.float32)
        self.program['iTime'] = 0.0

        self.cameraPos = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        f = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
        r = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)
        u = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

        f = normalize_safe(project_to_tangent(self.cameraPos, f))
        r = project_to_tangent(self.cameraPos, r)
        r = r - f * np.dot(f, r)
        r = normalize_safe(r)
        u = project_to_tangent(self.cameraPos, u) - f*np.dot(f, u) - r*np.dot(r, u)
        u = normalize_safe(u)

        self.cameraDirs = [f, r, u]

        self.vv = self.cameraDirs[0].copy()

        self.moveSpeed = 1.2
        self.keyState = {}
        self.t0 = time.perf_counter()
        self.last_time = self.t0

        self.mouse_down = False
        self.last_mouse_pos = None
        self.mouse_sensitivity = 0.004

        self.timer = app.Timer('auto', connect=self.on_timer, start=True)
        gloo.set_viewport(0, 0, *self.size)

        self.program['cameraPos'] = self.cameraPos.astype(np.float32)
        self.program['forward'] = self.cameraDirs[0].astype(np.float32)
        self.program['right']   = self.cameraDirs[1].astype(np.float32)
        self.program['up']      = self.cameraDirs[2].astype(np.float32)

        self.show()

    def on_key_press(self, event):
        if event.key is None or event.key.name is None: return
        self.keyState[event.key.name.upper()] = True

    def on_key_release(self, event):
        if event.key is None or event.key.name is None: return
        self.keyState[event.key.name.upper()] = False

    def on_mouse_press(self, event):
        self.mouse_down = True
        self.last_mouse_pos = np.array(event.pos, dtype=np.float64)

    def on_mouse_release(self, event):
        self.mouse_down = False
        self.last_mouse_pos = None

    def on_mouse_move(self, event):
        if not self.mouse_down: return
        if self.last_mouse_pos is None:
            self.last_mouse_pos = np.array(event.pos, dtype=np.float64)
            return
        cur = np.array(event.pos, dtype=np.float64)
        delta = cur - self.last_mouse_pos
        self.last_mouse_pos = cur
        dx, dy = delta[0], -delta[1]
        yaw = -dx * self.mouse_sensitivity
        pitch = -dy * self.mouse_sensitivity

        f = self.cameraDirs[0]; r = self.cameraDirs[1]; u = self.cameraDirs[2]

        ca = np.cos(yaw); sa = np.sin(yaw)
        f1 = ca*f + sa*r
        r1 = -sa*f + ca*r

        cb = np.cos(pitch); sb = np.sin(pitch)
        f2 = cb*f1 + sb*u
        u2 = -sb*f1 + cb*u

        f2 = normalize_safe(project_to_tangent(self.cameraPos, f2))
        r2 = normalize_safe(project_to_tangent(self.cameraPos, r1))
        r2 = r2 - f2 * np.dot(f2, r2)
        r2 = normalize_safe(r2)
        u2 = project_to_tangent(self.cameraPos, u2)
        u2 = u2 - f2*np.dot(f2, u2) - r2*np.dot(r2, u2)
        u2 = normalize_safe(u2)

        self.cameraDirs = [f2, r2, u2]
        self.vv = f2.copy()

        self.program['forward'] = f2.astype(np.float32)
        self.program['right']   = r2.astype(np.float32)
        self.program['up']      = u2.astype(np.float32)

    def on_resize(self, event):
        w, h = event.physical_size
        gloo.set_viewport(0, 0, w, h)
        self.program['iResolution'] = np.array((w, h), dtype=np.float32)

    def update_camera(self, dt):
        local_disp = np.zeros(4, dtype=np.float64)
        speed = self.moveSpeed * dt

        if self.keyState.get('W', False):
            local_disp += self.cameraDirs[0] * speed
        if self.keyState.get('S', False):
            local_disp -= self.cameraDirs[0] * speed
        if self.keyState.get('D', False):
            local_disp += self.cameraDirs[1] * speed
        if self.keyState.get('A', False):
            local_disp -= self.cameraDirs[1] * speed
        if self.keyState.get('E', False):
            local_disp += self.cameraDirs[2] * speed
        if self.keyState.get('Q', False):
            local_disp -= self.cameraDirs[2] * speed

        if np.allclose(local_disp, 0.0):
            self.program['cameraPos'] = self.cameraPos.astype(np.float32)
            self.program['forward'] = self.cameraDirs[0].astype(np.float32)
            self.program['right']   = self.cameraDirs[1].astype(np.float32)
            self.program['up']      = self.cameraDirs[2].astype(np.float32)
            return

        old_pos = self.cameraPos.copy()
        new_dirs = [parallel_transport(old_pos, local_disp, d, 1.0) for d in self.cameraDirs]
        new_pos = retract(old_pos, local_disp)

        new_dirs = orthonormalize_at(new_pos, new_dirs)

        self.cameraPos = normalize_safe(new_pos)
        self.cameraDirs = new_dirs
        self.vv = self.cameraDirs[0].copy()

        self.program['cameraPos'] = self.cameraPos.astype(np.float32)
        self.program['forward'] = self.cameraDirs[0].astype(np.float32)
        self.program['right']   = self.cameraDirs[1].astype(np.float32)
        self.program['up']      = self.cameraDirs[2].astype(np.float32)

    def on_draw(self, event):
        gloo.clear(color='black')
        self.program.draw('triangle_strip')

    def on_timer(self, event):
        now = time.perf_counter()
        dt = event.dt if hasattr(event, 'dt') and event.dt is not None else (now - self.last_time)
        dt = min(dt, 0.05)
        self.last_time = now

        t = now - self.t0
        self.program['iTime'] = t
        self.update_camera(dt)
        self.update()

if __name__ == '__main__':
    c = Canvas()
    app.run()