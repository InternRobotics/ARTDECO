#version 330

#if defined VERTEX_SHADER

void main(void) {}

#elif defined GEOMETRY_SHADER
layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

uniform mat4 m_model;
uniform mat4 m_camera;
uniform mat4 m_proj;
uniform int width;
uniform int height;
uniform float conf_threshold;
uniform float slant_threshold = 0.1;
uniform float depth_bias = 0.0;

uniform sampler2D pointmap;
uniform sampler2D confs;
uniform sampler2D img;

out fragData
{
  vec3 col;
  vec3 pos;
  vec3 normal;
} vertex;

vec3 normal(vec3 a, vec3 b, vec3 c) {
  return normalize(cross(b - a, c - a));
}

struct Pixel {
  vec3 pos;
  vec3 col;
  float conf;
};

void main(void) {
  int y = gl_PrimitiveIDIn / int(width);
  int x = gl_PrimitiveIDIn - y * int(width);
  if (x < 10 || x >= int(width) - 10 || y < 10 || y >= int(height) - 10) {
    return;
  }
  mat4 mv = m_camera * m_model;
  mat4 mvp = m_proj * mv;

  const int TL = 0; const int TR = 1; const int BL = 2; const int BR = 3;
  ivec2 offsets[4] = ivec2[4](ivec2(0, 0), ivec2(1, 0), ivec2(0, 1), ivec2(1, 1));
  Pixel pixels[4];
  for (int i = 0; i < 4; ++i) {
    ivec2 offset = offsets[i];
    pixels[i].pos = texelFetch(pointmap, ivec2(x, y) + offset, 0).xyz;
    pixels[i].col = texelFetch(img, ivec2(x, y) + offset, 0).xyz;
    pixels[i].conf = texelFetch(confs, ivec2(x, y), 0).x;
  }

  vec3 n1 = normal(pixels[TL].pos, pixels[BL].pos, pixels[TR].pos);
  vec3 n2 = normal(pixels[TR].pos, pixels[BL].pos, pixels[BR].pos);
  vec3 normal = mat3(mv) * ((n1 + n2) / 2.0);
  vec3 ray1 = normalize(pixels[TL].pos);
  vec3 ray2 = normalize(pixels[TR].pos);
  if (abs(dot(n1, ray1)) < slant_threshold) {
    return ;
  }
  if (abs(dot(n2, ray2)) < slant_threshold) {
    return ;
  }




  for (int i = 0; i < 4; ++i) {
    if (pixels[i].conf < conf_threshold) {
      return;
    }
    vec4 p = m_model * vec4(pixels[i].pos, 1.0);
  }
  // CCW
  int vertex_order[4] = int[4](TL, BL, TR, BR);
  for (int i = 0; i < 4; ++i) {
    vec4 pos = mvp * vec4(pixels[vertex_order[i]].pos, 1.0);
    pos.z -= depth_bias; 
    gl_Position = pos;
    vertex.pos = (mv * vec4(pixels[vertex_order[i]].pos, 1.0)).xyz;
    vertex.col = pixels[vertex_order[i]].col;
    vertex.normal = normal;
    EmitVertex();
  }
  EndPrimitive();
}

#elif defined FRAGMENT_SHADER
in fragData
{
  vec3 col;
  vec3 pos;
  vec3 normal;
} frag;

out vec4 out_color;

uniform vec3 lightpos = vec3(0.1, 0.1, 0);
uniform vec3 phong = vec3(0.3, 0.5, 0.4);
uniform float spec = 32;
uniform vec3 base_color = vec3(1, 1, 1);
uniform bool use_img = true;
uniform bool show_normal;

uniform mat4 m_model;
uniform mat4 m_camera;
uniform mat4 m_proj;

void main() {
  vec3 normal = normalize(frag.normal);
  vec3 position = frag.pos;
  mat4 mv = m_camera * m_model;
  vec3 light = lightpos;
  float kA = use_img ? phong[0] : 0.1;
  float kD = use_img ? phong[1] : 0.2;
  float kS = use_img ? phong[2] : 1.3;
  vec3 frag_color = use_img ? frag.col : base_color;
  float spec_power = use_img ? spec : 3;
  vec3 specular_color = vec3(1.0, 1.0, 1.0);

  vec3 L = normalize(light - position);
  vec3 N = normalize(normal);
  float lambertian = max(dot(N, L), 0.0);
  float specular = 0;
  if (lambertian > 0.0) {
    vec3 R = 2 * (L * N) * N - L;
    vec3 V = normalize(-position);
    specular = pow(max(dot(R, V), 0.0), spec_power);
  }
  vec3 color = frag_color * (kA + lambertian * kD ) + specular_color * kS * specular;
  out_color = vec4(color, 1.0);
  if (show_normal) {
    N = vec3(N.x, -N.y, -N.z);
    vec3 col = -N * 0.5 + 0.5;
    out_color = vec4(col, 1.0);
  }
}
#endif