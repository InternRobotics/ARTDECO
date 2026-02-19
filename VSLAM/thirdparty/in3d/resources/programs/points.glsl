#version 330

#if defined VERTEX_SHADER

in vec3 in_position;
in vec4 in_color;

uniform mat4 m_model;
uniform mat4 m_camera;
uniform mat4 m_proj;

out vec4 vcol;

void main() {
    mat4 m_view = m_camera * m_model;
    vec4 p = m_view * vec4(in_position, 1.0);
    gl_Position =  m_proj * p;
    vcol = in_color;
}

#elif defined FRAGMENT_SHADER

in vec4 vcol;
out vec4 fragColor;

void main() {
    fragColor = vcol;
}
#endif