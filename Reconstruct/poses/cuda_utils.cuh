/*
 * Copyright (C) 2025, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <cuda_fp16.h>
#include <cuda/std/utility>

#ifndef CHANNELS
    #define CHANNELS 64
#endif
#ifndef NUM_CAMS
    #define NUM_CAMS 6
#endif
#ifndef NUM_DEPTH_CANDIDATES
    #define NUM_DEPTH_CANDIDATES 128
#endif

class Matx33f {
public:
    __device__ Matx33f(float* data) {
        memcpy(this->data, data, 9 * sizeof(float));
    }

    __device__ Matx33f(
        float c00=0.f, float c01=0.f, float c02=0.f,
        float c10=0.f, float c11=0.f, float c12=0.f,
        float c20=0.f, float c21=0.f, float c22=0.f
    ) {
        data[0][0] = c00; data[0][1] = c01; data[0][2] = c02;
        data[1][0] = c10; data[1][1] = c11; data[1][2] = c12;
        data[2][0] = c20; data[2][1] = c21; data[2][2] = c22;
    }

    __device__ Matx33f(float3 r0, float3 r1, float3 r2) {
        data[0][0] = r0.x; data[0][1] = r0.y; data[0][2] = r0.z;
        data[1][0] = r1.x; data[1][1] = r1.y; data[1][2] = r1.z;
        data[2][0] = r2.x; data[2][1] = r2.y; data[2][2] = r2.z;
    }

    // Accessor for element at (i, j)
    __device__ float& operator()(int i, int j) {
        return data[i][j];
    }

    __device__ const float& operator()(int i, int j) const {
        return data[i][j];
    }

    __device__ const float3 operator*(const float3& v) const {
        return {
            data[0][0] * v.x + data[0][1] * v.y + data[0][2] * v.z,
            data[1][0] * v.x + data[1][1] * v.y + data[1][2] * v.z,
            data[2][0] * v.x + data[2][1] * v.y + data[2][2] * v.z
        };
    }

    __device__ const Matx33f operator*(const Matx33f& m) const {
        Matx33f result;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                result(i, j) = 0;
                for (int k = 0; k < 3; ++k) {
                    result(i, j) += data[i][k] * m(k, j);
                }
            }
        }
        return result;
    }
    // Transpose the matrix
    __device__ Matx33f t() const {
        Matx33f result;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                result(i, j) = data[j][i];
            }
        }
        return result;
    }

    __device__ float3 col(int i) const {
        return {data[0][i], data[1][i], data[2][i]};
    }

    __device__ float3 row(int i) const {
        return {data[i][0], data[i][1], data[i][2]};
    }

    __device__ Matx33f operator*(float s) const {
        Matx33f result;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                result(i, j) = data[i][j] * s;
            }
        }
        return result;
    }

    // Inverse the matrix using Gaussian elimination
    __device__ Matx33f inv() const {
        Matx33f result = *this;
        Matx33f identity(1, 0, 0,
                         0, 1, 0,
                         0, 0, 1);
    
        for (int i = 0; i < 3; ++i) {
            // Partial Pivoting
            int pivotRow = i;
            for (int k = i + 1; k < 3; ++k) {
                if (abs(result(k, i)) > abs(result(pivotRow, i))) {
                    pivotRow = k;
                }
            }
            
            // Swap rows if necessary
            if (pivotRow != i) {
                for (int j = 0; j < 3; ++j) {
                    cuda::std::swap(result(i, j), result(pivotRow, j));
                    cuda::std::swap(identity(i, j), identity(pivotRow, j));
                }
            }
    
            float diag = result(i, i);
            if (abs(diag) < 1e-6f) {
                return identity;
            }
    
            // Normalize pivot row
            for (int j = 0; j < 3; ++j) {
                result(i, j) /= diag;
                identity(i, j) /= diag;
            }
    
            // Eliminate other rows
            for (int k = 0; k < 3; ++k) {
                if (k != i) {
                    float factor = result(k, i);
                    for (int j = 0; j < 3; ++j) {
                        result(k, j) -= factor * result(i, j);
                        identity(k, j) -= factor * identity(i, j);
                    }
                }
            }
        }
        return identity;
    }
        
    float data[3][3];
};

class Pose {
public:
    __device__ Pose(Matx33f r, float3 t) {
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                data[i][j] = r(i, j);
            }
        }
        data[0][3] = t.x;
        data[1][3] = t.y;
        data[2][3] = t.z;
    }

    __device__ float3 apply(const float3& p) const {
        return {
            data[0][0] * p.x + data[0][1] * p.y + data[0][2] * p.z + data[0][3],
            data[1][0] * p.x + data[1][1] * p.y + data[1][2] * p.z + data[1][3],
            data[2][0] * p.x + data[2][1] * p.y + data[2][2] * p.z + data[2][3]
        };
    }

    float data[3][4];
};

struct Intrinsics {float f, cx, cy;};

inline __device__ float3 operator*(float3 a, float b)
{
    return {a.x * b, a.y * b, a.z * b};
}

inline __device__ float3 operator*(float a, float3 b)
{
    return {a * b.x, a * b.y, a * b.z};
}

inline __device__ float3 operator+(float3 a, float3 b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

inline __device__ float2 operator-(float2 a, float2 b)
{
    return {a.x - b.x, a.y - b.y};
}

inline __device__ float3 operator-(float3 a, float3 b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

inline __device__ float3 operator/(float3 a, float b)
{
    return {a.x / b, a.y / b, a.z / b};
}

inline __device__ void operator/=(float3& a, float b)
{
    a.x /= b;
    a.y /= b;
    a.z /= b;
}

inline __device__ float norm2(float2 a)
{
    return (a.x * a.x + a.y * a.y);
}

inline __device__ float norm2(float3 a)
{
    return (a.x * a.x + a.y * a.y + a.z * a.z);
}

inline __device__ float norm(float3 a)
{
    return sqrtf(norm2(a));
}

inline __device__ float dist2(float2 a, float2 b)
{
    return norm2(a - b);
}

inline __device__ float dot(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __device__ float3 cross(float3 a, float3 b)
{
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

inline __device__ float2 project(const float3& xyz, const Intrinsics& intrinsics)
{
    return {intrinsics.f * (xyz.x / xyz.z) + intrinsics.cx, intrinsics.f * (xyz.y / xyz.z) + intrinsics.cy};
}

inline __device__ float2 project(const float3& xyz, const Intrinsics& intrinsics, const Pose& Rt)
{
    return project(Rt.apply(xyz), intrinsics);
}


struct float_C
{
    float data[CHANNELS];
    inline __device__ float_C operator*(float b) const
    {
        float_C output;
        #pragma unroll
        for (int i = 0; i < CHANNELS; i++)
        {
            output[i] = data[i] * b;
        }
        return output;
    }
    inline __device__ float_C operator+(const float_C& b) const
    {
        float_C output;
        #pragma unroll
        for (int i = 0; i < CHANNELS; i++)
        {
            output[i] = data[i] + b[i];
        }
        return output;
    }
    __device__ float& operator[](int i) { return data[i]; }
    __device__ const float& operator[](int i) const { return data[i]; }
};
// __align__(sizeof(half)*CHANNELS)
struct half_C
{
    half data[CHANNELS];
    inline __device__ float_C operator*(float b) const
    {
        float_C output;
        #pragma unroll
        for (int i = 0; i < CHANNELS; i++)
        {
            output[i] = float(data[i]) * b;
        }
        return output;
    }
    inline __device__ half_C operator*(half b) const
    {
        half_C output;
        #pragma unroll
        for (int i = 0; i < CHANNELS; i++)
        {
            output[i] = data[i] * b;
        }
        return output;
    }
    inline __device__ half_C operator+(const half_C& b) const
    {
        half_C output;
        #pragma unroll
        for (int i = 0; i < CHANNELS; i++)
        {
            output[i] = data[i] + b[i];
        }
        return output;
    }
    __device__ half& operator[](int i) { return data[i]; }
    __device__ const half& operator[](int i) const { return data[i]; }
};

inline __device__ float dist1(const half_C& a, const half_C& b)
{
    float output = 0.f;
    #pragma unroll
    for (int i = 0; i < CHANNELS; i++)
    {
        output += __half2float(__habs(a[i] - b[i]));
    }
    return output;
}

inline __device__ float dist1(const float_C& a, const float_C& b)
{
    float output = 0.f;
    #pragma unroll
    for (int i = 0; i < CHANNELS; i++)
    {
        output += abs(a[i] - b[i]);
    }
    return output;
}

inline __device__ float_C interp(const half_C* featMap, float2 uv, int featMapW, int featMapH)
{
    uv.x = min(max(uv.x, 0.1f), featMapW - 1.1f);
    uv.y = min(max(uv.y, 0.1f), featMapH - 1.1f);
    int x = __float2int_rd(uv.x);
    int y = __float2int_rd(uv.y); 
    float2 uvFrac = {uv.x - x, uv.y - y};

    const half_C sample00 = featMap[(y) * featMapW + x];
    const half_C sample01 = featMap[(y + 1) * featMapW + x];
    const half_C sample10 = featMap[(y) * featMapW + x + 1];
    const half_C sample11 = featMap[(y + 1) * featMapW + x + 1];

    float_C output = sample00 * ((1.f - uvFrac.x) * (1.f - uvFrac.y)) +
                     sample01 * ((1.f - uvFrac.x) * uvFrac.y) +
                     sample10 * (uvFrac.x * (1.f - uvFrac.y)) +
                     sample11 * (uvFrac.x * uvFrac.y);

    return output;
}

inline __device__ void interp_ptr(float_C* target, const half_C* featMap, float2 uv, int featMapW, int featMapH)
{
    uv.x = min(max(uv.x, 0.1f), featMapW - 1.1f);
    uv.y = min(max(uv.y, 0.1f), featMapH - 1.1f);
    int x = __float2int_rd(uv.x);
    int y = __float2int_rd(uv.y); 
    float2 uvFrac = {uv.x - x, uv.y - y};

    const half_C* sample00 = &featMap[(y) * featMapW + x];
    const half_C* sample01 = &featMap[(y + 1) * featMapW + x];
    const half_C* sample10 = &featMap[(y) * featMapW + x + 1];
    const half_C* sample11 = &featMap[(y + 1) * featMapW + x + 1];

    for(int i = 0; i < CHANNELS; i++)
    {
        target->data[i] = 
        ((float)sample00->data[i]) * ((1.f - uvFrac.x) * (1.f - uvFrac.y)) +
        ((float)sample01->data[i]) * ((1.f - uvFrac.x) * (uvFrac.y)) +
        ((float)sample10->data[i]) * ((uvFrac.x) * (1.f - uvFrac.y)) +
        ((float)sample11->data[i]) * ((uvFrac.x) * (uvFrac.y));
    }
}

inline __device__ float interp_dist_ptr(const float_C* ref, const half_C* featMap, float2 uv, int featMapW, int featMapH)
{
    uv.x = min(max(uv.x, 0.1f), featMapW - 1.1f);
    uv.y = min(max(uv.y, 0.1f), featMapH - 1.1f);
    int x = __float2int_rd(uv.x);
    int y = __float2int_rd(uv.y); 
    float2 uvFrac = {uv.x - x, uv.y - y};

    const half_C* sample00 = &featMap[(y) * featMapW + x];
    const half_C* sample01 = &featMap[(y + 1) * featMapW + x];
    const half_C* sample10 = &featMap[(y) * featMapW + x + 1];
    const half_C* sample11 = &featMap[(y + 1) * featMapW + x + 1];

    float output = 0.f;
    for(int i = 0; i < CHANNELS; i++)
    {
        output += abs(ref->data[i] -
        (((float)sample00->data[i]) * ((1.f - uvFrac.x) * (1.f - uvFrac.y)) +
        ((float)sample01->data[i]) * ((1.f - uvFrac.x) * (uvFrac.y)) +
        ((float)sample10->data[i]) * ((uvFrac.x) * (1.f - uvFrac.y)) +
        ((float)sample11->data[i]) * ((uvFrac.x) * (uvFrac.y)))
        );
    }
    return output;
}

inline __device__ float interp(const float* featMap, float2 uv, int featMapW, int featMapH)
{
    uv.x = min(max(uv.x, 0.1f), featMapW - 1.1f);
    uv.y = min(max(uv.y, 0.1f), featMapH - 1.1f);
    int x = __float2int_rd(uv.x);
    int y = __float2int_rd(uv.y); 
    float2 uvFrac = {uv.x - x, uv.y - y};

    const float sample00 = featMap[(y) * featMapW + x];
    const float sample01 = featMap[(y + 1) * featMapW + x];
    const float sample10 = featMap[(y) * featMapW + x + 1];
    const float sample11 = featMap[(y + 1) * featMapW + x + 1];

    float output = sample00 * ((1.f - uvFrac.x) * (1.f - uvFrac.y)) +
                     sample01 * ((1.f - uvFrac.x) * uvFrac.y) +
                     sample10 * (uvFrac.x * (1.f - uvFrac.y)) +
                     sample11 * (uvFrac.x * uvFrac.y);

    return output;
}

inline __device__ float2 makeSamplingUV(float2 uv, int featMapW, int featMapH, int W, int H)
{
    return {
        (uv.x + 0.5f) * (float(featMapW) / W) - 0.5f,
        (uv.y + 0.5f) * (float(featMapH) / H) - 0.5f
    };
}
