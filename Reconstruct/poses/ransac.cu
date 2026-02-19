#include "cuda_utils.cuh"

#include <cuda/std/utility>


// Based on https://github.com/opencv/opencv/tree/4.x/modules/calib3d/src/usac
// Modified for CUDA and to interface with CuPy

// https://github.com/opencv/opencv/blob/42a132088cfc2060e9ae816bbcf7ebfcab4f1de8/modules/calib3d/src/usac/utils.cpp#L468
inline __device__ bool eliminateUpperTriangular(float *a, int m, int n) {
    for (int r = 0; r < m; r++){
        float pivot = a[r*n+r];
        int row_with_pivot = r;

        // find the maximum pivot value among r-th column
        for (int k = r+1; k < m; k++)
            if (fabs(pivot) < fabs(a[k*n+r])) {
                pivot = a[k*n+r];
                row_with_pivot = k;
            }

        // if pivot value is 0 continue
        if (fabs(pivot) < 1e-8)
            continue;

        // swap row with maximum pivot value with current row
        for (int c = r; c < n; c++)
            cuda::std::swap(a[row_with_pivot*n+c], a[r*n+c]);

        // eliminate other rows
        for (int j = r+1; j < m; j++){
            const int row_idx1 = j*n, row_idx2 = r*n;
            const auto fac = a[row_idx1+r] / pivot;
            a[row_idx1+r] = 0; // zero eliminated element
            for (int c = r+1; c < n; c++)
                a[row_idx1+c] -= fac * a[row_idx2+c];
        }
    }
    return true;
}

// https://github.com/opencv/opencv/blob/42a132088cfc2060e9ae816bbcf7ebfcab4f1de8/modules/calib3d/src/usac/fundamental_solver.cpp#L162
extern "C" __global__ void batchFundMat8pts(
    const float2* mkpts1,
    const float2* mkpts2,
    const int64_t* indices,
    Matx33f* Fs,
    bool* isValid, 
    int batchSize
)
{
    int batch = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch >= batchSize)
    {
        return;
    }

    const int m = 8, n = 9; // rows, cols
    float a[72];
    for (int row = 0; row < m; row++)
    {
        int64_t index = indices[batch * m + row];
        float2 uv1 = mkpts1[index];
        float2 uv2 = mkpts2[index];
        float x1 = uv1.x, y1 = uv1.y, x2 = uv2.x, y2 = uv2.y;   

        a[row * 9 + 0] = x2*x1;
        a[row * 9 + 1] = x2*y1;
        a[row * 9 + 2] = x2;
        a[row * 9 + 3] = y2*x1;
        a[row * 9 + 4] = y2*y1;
        a[row * 9 + 5] = y2;
        a[row * 9 + 6] = x1;
        a[row * 9 + 7] = y1;
        a[row * 9 + 8] = 1;
    }
    
    if(!eliminateUpperTriangular(a, m, n))
    {
        isValid[batch] = false;
        return;
    }

    float f[9] = {0};
    f[8] = 1;

    // start from the last row
    for (int i = m-1; i >= 0; i--) {
        float acc = 0;
        for (int j = i+1; j < n; j++){
            acc -= a[i*n+j]*f[j];
        }

        f[i] = acc / a[i*n+i];
    }
    memcpy(Fs[batch].data, f, sizeof(Matx33f));
    isValid[batch] = true;
}

// https://github.com/opencv/opencv/blob/42a132088cfc2060e9ae816bbcf7ebfcab4f1de8/modules/calib3d/src/usac/estimator.cpp#L337
extern "C" __global__ void sampsonInliers(
    const float2* mkpts1,
    const float2* mkpts2,
    const Matx33f* Fs,
    const bool* isValid,
    bool* inliers,
    float threshold,
    int batchSize,
    int nPts
)
{
    int global_index = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = global_index / nPts;
    int index = global_index % nPts;
    

    if (batch >= batchSize || index >= nPts)
    {
        return;
    }
    if (!isValid[batch])
    {
        inliers[batch * nPts + index] = false;
        return;
    }

    Matx33f F = Fs[batch];
    float m11=F(0, 0), m12=F(0, 1), m13=F(0, 2);
    float m21=F(1, 0), m22=F(1, 1), m23=F(1, 2);
    float m31=F(2, 0), m32=F(2, 1), m33=F(2, 2);

    float2 uv1 = mkpts1[index];
    float2 uv2 = mkpts2[index];
    float x1 = uv1.x, y1 = uv1.y, x2 = uv2.x, y2 = uv2.y;

    const float F_pt1_x = m11 * x1 + m12 * y1 + m13,
                F_pt1_y = m21 * x1 + m22 * y1 + m23;
    const float pt2_F_x = x2 * m11 + y2 * m21 + m31,
                pt2_F_y = x2 * m12 + y2 * m22 + m32;
    const float pt2_F_pt1 = x2 * F_pt1_x + y2 * F_pt1_y + m31 * x1 + m32 * y1 + m33;
    float error =  pt2_F_pt1 * pt2_F_pt1 / (F_pt1_x * F_pt1_x + F_pt1_y * F_pt1_y +
                                        pt2_F_x * pt2_F_x + pt2_F_y * pt2_F_y);

    if (error < threshold)
    {
        inliers[batch * nPts + index] = true;
    }
}