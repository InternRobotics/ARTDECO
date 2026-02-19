#version 330

#if defined VERTEX_SHADER
void main() {}

#elif defined GEOMETRY_SHADER

layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

// Uniforms and textures
uniform mat4 m_model;
uniform mat4 m_camera;
uniform mat4 m_proj;
uniform vec4 K; // fx, fy, ux, uy
uniform int width;
uniform int height;
uniform float slt_thresh = 0.075;
uniform int crop_pix = 30;

uniform sampler2D image;
uniform sampler2D depth;
uniform sampler2D valid;

// Output to fragment shader
out fragData
{
  vec3 color;
  vec3 pos;
  vec3 normal;
} vertex;

vec3 normal(vec3 a, vec3 b, vec3 c)
{
  return normalize(cross(b - a, c - a));
}

vec4 unproject(vec4 K, vec2 point, float d)
{
  float fx = K[0];
  float fy = K[1];
  float cx = K[2];
  float cy = K[3];
  vec3 ray = vec3((point.x - cx) / fx, (point.y - cy) / fy, 1);
  return vec4(ray * d, 1);
}


struct Pixel
{
  vec3 color;
  float depth;
  bool valid;
  vec4 pos_C;
};

Pixel get(ivec2 loc, vec4 K)
{
  // texelFetch uses bottom-left as origin, but we've uploaded
  // the image flipped, so we can just indexing with top-left as origin
  Pixel data;
  data.color = texelFetch(image, loc, 0).xyz;
  data.depth = texelFetch(depth, loc, 0).x;
  data.valid = texelFetch(valid, loc, 0).x >= 0;
  data.pos_C = unproject(K, vec2(loc), data.depth);
  return data;
}

void main(void)
{
  mat4 mv = m_camera * m_model;
  mat4 mvp = m_proj * mv;

  // get (x,y) pixel location from primitive id
  int x = gl_PrimitiveIDIn % int(width);
  int y = gl_PrimitiveIDIn / int(width);

  if (x < crop_pix || x > int(width) - crop_pix || y < crop_pix || y > int(height) - crop_pix) {
    return;
  }

  // top-left, top-right, bottom-left, bottom-right
  const int TL = 0; const int TR = 1; const int BL = 2; const int BR = 3;
  ivec2 offsets[4] = ivec2[4](ivec2(0, 0), ivec2(1, 0), ivec2(0, 1), ivec2(1, 1));
  Pixel pixels[4];
  for (int i = 0; i < 4; ++i) {
    ivec2 offset = offsets[i];
    ivec2 coord = ivec2(x, y) + offset;
    pixels[i] = get(coord, K);
  }

  // Normal in the coordinate frame of the depth image
  vec3 n1 = normal(pixels[TL].pos_C.xyz, pixels[BL].pos_C.xyz, pixels[TR].pos_C.xyz);
  vec3 n2 = normal(pixels[TR].pos_C.xyz, pixels[BL].pos_C.xyz, pixels[BR].pos_C.xyz);

  vec3 ray = normalize(vec3(unproject(K, vec2(x, y), 1.0)));
  if (abs(dot(n1, ray)) < slt_thresh || abs(dot(n2, ray)) < slt_thresh) {
    return;
  }

  // Normal in the world coordinate frame
  vec3 normal = mat3(mv) * (n1 + n2) / 2.0;
  for (int i = 0; i < 4; ++i) {
    if (!pixels[i].valid) {
        return;
    } 
  }
  // CCW
  int vertex_order[4] = int[4](TL, BL, TR, BR);
  for (int i = 0; i < 4; ++i) {
    gl_Position = mvp * pixels[vertex_order[i]].pos_C;
    vertex.pos = (mv * pixels[vertex_order[i]].pos_C).xyz;
    vertex.color = pixels[vertex_order[i]].color;
    vertex.normal = normal;
    EmitVertex();
  }
  EndPrimitive();
}

#elif defined FRAGMENT_SHADER
uniform mat4 m_model;
uniform mat4 m_camera;
uniform vec3 lightpos = vec3(1, -1, -1);
uniform vec3 phong = vec3(0.1, 0.3, 3);
uniform float spec = 2;
uniform bool texmap = false;
uniform bool shownormal = true;
uniform vec3 basecolor = vec3(1, 1, 1);

in fragData
{
  vec3 color;
  vec3 pos;
  vec3 normal;
} frag;

out vec4 out_color;

void main() {
    if (shownormal) {
      vec3 N = normalize(frag.normal);
      // opengl normal so flip to opencv
      N = vec3(N.x, -N.y, -N.z);
      // Set the normal to be outward normals
      out_color = vec4(-N * 0.5 + 0.5, 1.0);
    } else {
      float kA = phong[0];
      float kD = phong[1];
      float kS = phong[2];

      vec3 color = texmap ? frag.color : basecolor;

      vec3 L = normalize(lightpos - frag.pos);
      vec3 N = normalize(frag.normal);
      float lambertian = max(dot(N, L), 0.0);
      float specular = 0;
      if (lambertian > 0.0) {
          vec3 R = 2 * (L * N) * N - L;
          vec3 V = normalize(-frag.pos);
          specular = pow(max(dot(R, V), 0.0), spec);
      // out_color = vec4(color * kA + basecolor * lambertian * kD + kS * specular, 1.0);
      // } else {
      //   out_color = vec4(1, 0, 0, 1);
      }
      out_color = vec4(vec3(color * (kA +lambertian * kD) + basecolor * (kS * specular)), 1.0);
    }
}
#endif