#version 330

#if defined VERTEX_SHADER

in vec3 in_position;
in vec3 in_normal;

uniform mat4 m_model;
uniform mat4 m_camera;
uniform mat4 m_proj;

out vec3 pos;
out vec3 normal;

void main() {
    mat4 m_view = m_camera * m_model;
    vec4 p = m_view * vec4(in_position, 1.0);
    gl_Position =  m_proj * p;
    mat3 m_normal = inverse(transpose(mat3(m_view)));
    normal = m_normal * normalize(in_normal);
    pos = p.xyz;
}

#elif defined FRAGMENT_SHADER

out vec4 fragColor;
uniform vec4 color = vec4(1.0, 1.0, 1.0, 1.0);
in vec3 pos; 
in vec3 normal;

void main() {
    float kA = 0.1;
    float kD = 0.2;
    float kS = 1;
    vec3 light_pos = vec3(1, 1, -1);
    vec3 L = normalize(light_pos - pos);
    vec3 N = normalize(normal);
    float lambertian = max(dot(N, L), 0.0);
    float specular = 0;
    if (lambertian > 0.0) {
        vec3 R = reflect(-L, N);
        vec3 V = normalize(-pos);
        specular = pow(max(dot(R, V), 0.0), 2);
    } 
    fragColor = vec4(vec3(color * (kA + lambertian * kD + kS * specular)), 1.0);
}
#endif