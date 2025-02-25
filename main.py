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
uniform vec2 iResolution;
uniform float iTime;

vec4 retract(vec4 x, vec4 v)
{
    float normV = length(v);
    vec4 newX;
    if(normV > 0.0) {
        newX = cos(normV)*x + (sin(normV)/normV)*v;
    } else {
        newX = x;
    }
    return newX;
}

vec4 parallelTransport(vec4 x, vec4 v, vec4 u, float t)
{
    return u + v*(dot(v, u)/dot(v,v))*(cos(t*length(v)) - 1.0)
           - x*(sin(t*length(v))*dot(v, u)/length(v));
}

float getDist(vec4 x, vec4 y)
{
    float cosTheta = dot(x,y);
    return acos(cosTheta);
}

void main()
{
    vec4 cameraVelocity = vec4(0.0, 0.33, -0.33, 0.0);
    vec2 fragCoord = gl_FragCoord.xy;

    vec2 uv = fragCoord / iResolution.x;
    uv.x -= 0.5;
    uv.y -= 0.5 * (iResolution.y / iResolution.x);

    vec4 cameraPos = vec4(1.0, 0.0, 0.0, 0.0);
    vec4 initialCameraPos = cameraPos;

    float cameraTime = iTime;
    cameraPos = retract(cameraPos, cameraTime * cameraVelocity);

    vec4 targetPos[3];
    targetPos[0] = vec4(0.0, 0.7, 0.7, 0.0);
    targetPos[1] = vec4(0.0, 1.0, 0.0, 0.0);
    targetPos[2] = vec4(0.0, 0.0, 1.0, 0.0);

    float R[3];
    R[0] = 0.4; R[1] = 0.4; R[2] = 0.4;

    float MAX_DISTANCE = 6.4;

    vec4 v = vec4(0.0, 1.0, uv);
    v = parallelTransport(initialCameraPos, cameraVelocity, v, cameraTime);
    v = normalize(v);

    vec4 x = cameraPos;
    float dist = getDist(cameraPos, targetPos[0]) - R[0];
    for(int i=1; i<3; ++i)
        dist = min(dist, getDist(cameraPos, targetPos[i]) - R[i]);

    float t = 0.0;
    while(dist > 0.001 && t < MAX_DISTANCE) {
        t += dist;
        t = min(t, MAX_DISTANCE);
        x = retract(cameraPos, t*v);
        dist = getDist(x, targetPos[0]) - R[0];
        for(int i=1; i<3; ++i)
            dist = min(dist, getDist(x, targetPos[i]) - R[i]);
    }

    vec3 col = vec3(0.0);
    if(dist <= 0.001) {
        col = vec3(0.5);
        col.x  = 1.0 - 0.75*smoothstep(0.0, 1.0, (fract(abs(2.0*x.x))-0.01)/fwidth(x.x));
        col.y  = 1.0 - 0.75*smoothstep(0.0, 1.0, (fract(abs(2.0*x.z))-0.01)/fwidth(x.z));
        col.z  = 1.0 - 0.75*smoothstep(0.0, 1.0, (fract(abs(2.0*x.w))-0.01)/fwidth(x.w));
    }

    gl_FragColor = vec4(col, 1.0);
}
"""

class Canvas(app.Canvas):
    def __init__(self):
        app.Canvas.__init__(self, keys='interactive', size=(800, 600))
        self.program = gloo.Program(vertex_shader, fragment_shader)
        self.program['iTime'] = 0.0
        self.program['a_position'] = np.array(
            [[-1, -1], [-1, +1], [+1, -1], [+1, +1]], np.float32
        )
        self.program['iResolution'] = self.size
        self.t0 = time.perf_counter()
        self.timer = app.Timer('auto', connect=self.on_timer, start=True)
        gloo.set_viewport(0, 0, *self.size)
        self.show()

    def on_resize(self, event):
        gloo.set_viewport(0, 0, *self.physical_size)
        self.program['iResolution'] = self.physical_size

    def on_draw(self, event):
        gloo.clear(color='black', depth=True)
        self.program.draw('triangle_strip')

    def on_timer(self, event):
        t = time.perf_counter() - self.t0
        self.program['iTime'] = t
        self.update()

if __name__ == '__main__':
    c = Canvas()
    app.run()
