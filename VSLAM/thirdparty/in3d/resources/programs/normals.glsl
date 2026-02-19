#version 330

#if defined VERTEX_SHADER

in vec3 in_position;
in vec3 in_normal;

uniform mat4 m_model;
uniform mat4 m_camera;
uniform mat4 m_proj;
uniform bool world = true;

out vec3 pos;
out vec3 normal;

void main() {
    mat4 m_view = m_camera * m_model;
    vec4 p = m_view * vec4(in_position, 1.0);
    gl_Position =  m_proj * p;
    if (world) {
        normal =  normalize(in_normal);
    } else {
        mat3 m_normal = inverse(transpose(mat3(m_view)));
        normal = m_normal * normalize(in_normal);
    }
    pos = p.xyz;
}

#elif defined FRAGMENT_SHADER

out vec4 fragColor;
uniform vec4 color;

in vec3 pos;
in vec3 normal;

void main() {
    color;
    fragColor = vec4(normal * 0.5 + 0.5, 1.0);
}
#endif