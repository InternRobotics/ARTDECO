#version 330

#if defined VERTEX_SHADER

in vec4 in_position_w;
in vec4 in_color;

uniform mat4 m_model;
uniform mat4 m_camera;
uniform mat4 m_proj;
uniform vec2 viewport_size;

out vec4 v_col;
noperspective out float v_size;

void main()
{
    viewport_size;;
    v_col = in_color;
    gl_Position =  m_proj * m_camera * m_model * vec4(in_position_w.xyz, 1.0);
}

#elif defined FRAGMENT_SHADER

in vec4 v_col;
uniform vec2 viewport_size;

out vec4 frag_color;
void main() {
    frag_color = v_col;
}

#endif