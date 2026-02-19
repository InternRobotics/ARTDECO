#version 330

// From https://github.com/john-chapman/im3d/blob/master/examples/OpenGL33/im3d.glsl

#define kAntialiasing 2

#if defined VERTEX_SHADER

in vec4 in_position_w;
in vec4 in_color;

uniform mat4 m_model;
uniform mat4 m_camera;
uniform mat4 m_proj;

out vec4 v_col;
noperspective out float v_size;

void main()
{
    v_col = in_color;
    v_col.a *= smoothstep(0.0, 1.0, in_position_w.w / kAntialiasing);
    v_size = max(in_position_w.w, kAntialiasing);
    gl_Position = m_proj * m_camera * m_model * vec4(in_position_w.xyz, 1.0);
}

#elif defined GEOMETRY_SHADER

layout(lines) in;
layout(triangle_strip, max_vertices = 4) out;

uniform vec2 viewport_size;

in vec4 v_col[];
noperspective in float v_size[];

noperspective out float g_edge_distance;
noperspective out float g_size;
out vec4 g_col;

void main()
{
    vec2 pos0 = gl_in[0].gl_Position.xy / gl_in[0].gl_Position.w;
    vec2 pos1 = gl_in[1].gl_Position.xy / gl_in[1].gl_Position.w;

    vec2 dir = pos0 - pos1;
    dir = normalize(vec2(dir.x, dir.y * viewport_size.y / viewport_size.x)); // correct for aspect ratio
    vec2 tng0 = vec2(-dir.y, dir.x);
    vec2 tng1 = tng0 * v_size[1] / viewport_size;
    tng0 = tng0 * v_size[0] / viewport_size;

    // line start
    gl_Position = vec4((pos0 - tng0) * gl_in[0].gl_Position.w, gl_in[0].gl_Position.zw);
    g_edge_distance = -v_size[0];
    g_size = v_size[0];
    g_col = v_col[0];
    EmitVertex();

    gl_Position = vec4((pos0 + tng0) * gl_in[0].gl_Position.w, gl_in[0].gl_Position.zw);
    g_col = v_col[0];
    g_edge_distance = v_size[0];
    g_size = v_size[0];
    EmitVertex();

    // line end
    gl_Position = vec4((pos1 - tng1) * gl_in[1].gl_Position.w, gl_in[1].gl_Position.zw);
    g_edge_distance = -v_size[1];
    g_size = v_size[1];
    g_col = v_col[1];
    EmitVertex();

    gl_Position = vec4((pos1 + tng1) * gl_in[1].gl_Position.w, gl_in[1].gl_Position.zw);
    g_col = v_col[1];
    g_size = v_size[1];
    g_edge_distance = v_size[1];
    EmitVertex();
}

#elif defined FRAGMENT_SHADER

in vec4 g_col;
noperspective in float g_edge_distance;
noperspective in float g_size;

out vec4 frag_color;
void main() {
    frag_color = g_col;
    float d = abs(g_edge_distance) / g_size;
    d = smoothstep(1.0, 1.0 - (kAntialiasing / g_size), d);
    frag_color.a *= d;
}

#endif
