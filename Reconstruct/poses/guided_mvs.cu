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

#include "cuda_utils.cuh"

#define min_iz 1e-4f
#define max_iz 1e1f

/**
 * @brief Estimate depth values of for the UV coordinates using multi-view stereo guided with a coarse initial depth map.
 * This kernel finds the best depth candidates and refines using quadratic fitting.
 * Does not update the depth if invalid (not enough baseline, out of image bounds, or not weak maximum).
 *
 * @param uvs Array of UV coordinates.
 * @param refFeatMap Feature map of the reference camera.
 * @param otherFeatMaps Feature maps from other camera views.
 * @param Rts Array of relative camera poses. IMPORTANT: the format is [R0, t0, R1, t1, ...].
 * @param intrinsics_ Camera intrinsics [f, cx, cy].
 * @param idepthMap Input coarse inverse depth map used to guide the depth estimation. Typically from monocular depth.
 * @param depth Output array for depth values.
 * @param idist Output array for discrepancy between the input idepthMap and the inverse of the result depth.
 * @param range Inverse depth range for candidate generation.
 * @param nPts Number of points.
 * @param featMapH Height of the feature map.
 * @param featMapW Width of the feature map.
 * @param depthMapH Height of the depth map.
 * @param depthMapW Width of the depth map.
 * @param H Height of the image. Should match intrinsics_'s scale.
 * @param W Width of the image.
 */
extern "C" __global__ void uvToDepth(
    const float2* uvs,
    const half_C* refFeatMap,
    const half_C* otherFeatMaps,
    const Pose* Rts,
    const Intrinsics* intrinsics_,
    const float* idepthMap,
    float* depth,
    float* idist,
    float range,
    int nPts,
    int featMapH,
    int featMapW,
    int depthMapH,
    int depthMapW,
    int H,
    int W
)
{
    int index = blockIdx.x;

    if (index >= nPts)
    {
        return;
    }

    Intrinsics intrinsics = *intrinsics_;
    float2 uv = uvs[index];
    float3 unit_xyz = {
        (uv.x - intrinsics.cx) / intrinsics.f,
        (uv.y - intrinsics.cy) / intrinsics.f,
        1 
    };

    __shared__ int bestCamIdx_s;
    __shared__ bool valid[NUM_CAMS];
    __shared__ float costs[NUM_DEPTH_CANDIDATES];
    __shared__ float idepthCandidates[NUM_DEPTH_CANDIDATES];
    __shared__ float_C refFeat;
    __shared__ float refiDepth;
    __shared__ float step;

    if(threadIdx.x == 0)
    {
        step = 2 * range / (NUM_DEPTH_CANDIDATES - 1);
        float2 samplingUV = makeSamplingUV(uv, depthMapW, depthMapH, W, H);
        refiDepth = interp(idepthMap, samplingUV, depthMapW, depthMapH);
        refiDepth = max(refiDepth, 1e-6f);
        if (refiDepth >= max_iz)
            return;
        float3 min_xyz = unit_xyz * (1.f / min(refiDepth + range, max_iz));
        float3 max_xyz = unit_xyz * (1.f / max(refiDepth - range, min_iz));

        // Check the validity of the neighboring cameras
        bestCamIdx_s = -1;
        int bestCamIdx = -1;
        float maxDist = 0.0f;
        for (int camIdx(0); camIdx < NUM_CAMS; camIdx++)
        {
            float2 uvCamNear = project(min_xyz, intrinsics, Rts[camIdx]);
            float2 uvCamFar = project(max_xyz, intrinsics, Rts[camIdx]);
            float dist = dist2(uvCamNear, uvCamFar);
            valid[camIdx] = uvCamNear.x > 0 && uvCamNear.y > 0 && uvCamNear.x < W - 1 && uvCamNear.y < H - 1 &&
                            uvCamFar.x > 0 && uvCamFar.y > 0 && uvCamFar.x < W - 1 && uvCamFar.y < H - 1 && dist > 100.f;

            if (valid[camIdx] && maxDist < dist) 
            {
                maxDist = dist;
                bestCamIdx = camIdx;
            }
        }
        bestCamIdx_s = bestCamIdx;
    }
    __syncthreads();

    if (bestCamIdx_s == -1)
    {
        if(threadIdx.x == 0)
        {
            depth[index] = 1.f / refiDepth;
        }
        return;
    }

    // Feature of the reference camera
    if(threadIdx.x == 0)
    {
        float2 samplingUV = makeSamplingUV(uv, featMapW, featMapH, W, H);
        interp_ptr(&refFeat, refFeatMap, samplingUV, featMapW, featMapH);
    }
    __syncthreads();

    float minCost = 1e15f, secondMinCost = 1e15f, maxCost = 0.f; 
    int bestDepthIdx = 0, secondBestDepthIdx = 0;

    // Get the depth candidate around the input depth
    int candidateIdx = threadIdx.x;
    float idepthCandidate = refiDepth + candidateIdx * step - range;
    idepthCandidate = max(min(idepthCandidate, max_iz), min_iz);
    idepthCandidates[candidateIdx] = idepthCandidate;
    float depthCandidate = 1.f / idepthCandidate;
    float3 xyz = unit_xyz * depthCandidate;
    float cost = 0.f;

    // Compute the cost for this depth candidate given the reprojected feature of the neighoring cameras
    for (int camIdx(0); camIdx < NUM_CAMS; camIdx++)
    {
        if (!valid[camIdx])
            continue;
        
        float2 uvCam = project(xyz, intrinsics, Rts[camIdx]);
        float2 samplingUV = makeSamplingUV(uvCam, featMapW, featMapH, W, H);
        cost += interp_dist_ptr(&refFeat, otherFeatMaps + camIdx * featMapH * featMapW, samplingUV, featMapW, featMapH);
    }
    costs[candidateIdx] = cost;

    __syncthreads();
    if(threadIdx.x != 0)
    {
        return;
    }

    // Best depth candidate selection
    for (int candidateIdx(0); candidateIdx < NUM_DEPTH_CANDIDATES; candidateIdx++)
    {
        float cost = costs[candidateIdx];
        if (cost < minCost)
        {
            minCost = cost;
            bestDepthIdx = candidateIdx;
        }
        if (cost > maxCost)
        {
            maxCost = cost;
        }
    }

    if ((maxCost > 1.1f * minCost))
    {
        // Quadratic interpolation
        int leftIdx = max(bestDepthIdx - 1, 0);
        int rightIdx = min(bestDepthIdx + 1, NUM_DEPTH_CANDIDATES - 1);
        float leftCost = costs[leftIdx];
        float rightCost = costs[rightIdx];
        float variation = 0.5 * (leftCost - rightCost) / ((leftCost + rightCost) - 2. * minCost + 1e-8);
        variation = min(max(variation, -0.5f), 0.5f);

        float bestiDepth; 
        if (variation > 0)
        {
            bestiDepth = idepthCandidates[bestDepthIdx] * (1 - variation) + 
                        idepthCandidates[rightIdx] * variation;
        }
        else
        {
            variation = -variation;
            bestiDepth = idepthCandidates[bestDepthIdx] * (1 - variation) + 
                        idepthCandidates[leftIdx] * variation;
        }

        bestiDepth = max(min(bestiDepth, max_iz), min_iz);
        depth[index] = 1.f / bestiDepth;
        idist[index] = abs(bestiDepth - refiDepth);
    }
}