/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#define BOX_SIZE 1024
#define BOX_SIZE2 128

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "simple_knn.h"
#include <cfloat>
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <vector>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#define __CUDACC__
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

struct CustomMin
{
	__device__ __forceinline__
		float3 operator()(const float3& a, const float3& b) const {
		return { min(a.x, b.x), min(a.y, b.y), min(a.z, b.z) };
	}
};

struct CustomMax
{
	__device__ __forceinline__
		float3 operator()(const float3& a, const float3& b) const {
		return { max(a.x, b.x), max(a.y, b.y), max(a.z, b.z) };
	}
};

__host__ __device__ uint32_t prepMorton(uint32_t x)
{
	x = (x | (x << 16)) & 0x030000FF;
	x = (x | (x << 8)) & 0x0300F00F;
	x = (x | (x << 4)) & 0x030C30C3;
	x = (x | (x << 2)) & 0x09249249;
	return x;
}

__host__ __device__ uint32_t coord2Morton(float3 coord, float3 minn, float3 maxx)
{
	uint32_t x = prepMorton(((coord.x - minn.x) / (maxx.x - minn.x)) * ((1 << 10) - 1));
	uint32_t y = prepMorton(((coord.y - minn.y) / (maxx.y - minn.y)) * ((1 << 10) - 1));
	uint32_t z = prepMorton(((coord.z - minn.z) / (maxx.z - minn.z)) * ((1 << 10) - 1));

	return x | (y << 1) | (z << 2);
}

__global__ void coord2Morton(int P, const float3* points, float3 minn, float3 maxx, uint32_t* codes)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	codes[idx] = coord2Morton(points[idx], minn, maxx);
}

struct MinMax
{
	float3 minn;
	float3 maxx;
};

template <uint32_t B>
__global__ void boxMinMax(uint32_t P, float3* points, uint32_t* indices, MinMax* boxes)
{
	auto idx = cg::this_grid().thread_rank();

	MinMax me;
	if (idx < P)
	{
		me.minn = points[indices[idx]];
		me.maxx = points[indices[idx]];
	}
	else
	{
		me.minn = { FLT_MAX, FLT_MAX, FLT_MAX };
		me.maxx = { -FLT_MAX,-FLT_MAX,-FLT_MAX };
	}

	__shared__ MinMax redResult[BOX_SIZE];

	for (int off = BOX_SIZE / 2; off >= 1; off /= 2)
	{
		if (threadIdx.x < 2 * off)
			redResult[threadIdx.x] = me;
		__syncthreads();

		if (threadIdx.x < off)
		{
			MinMax other = redResult[threadIdx.x + off];
			me.minn.x = min(me.minn.x, other.minn.x);
			me.minn.y = min(me.minn.y, other.minn.y);
			me.minn.z = min(me.minn.z, other.minn.z);
			me.maxx.x = max(me.maxx.x, other.maxx.x);
			me.maxx.y = max(me.maxx.y, other.maxx.y);
			me.maxx.z = max(me.maxx.z, other.maxx.z);
		}
		__syncthreads();
	}

	if (threadIdx.x == 0)
		boxes[blockIdx.x] = me;
}

__device__ __host__ float distBoxPoint(const MinMax& box, const float3& p)
{
	float3 diff = { 0, 0, 0 };
	if (p.x < box.minn.x || p.x > box.maxx.x)
		diff.x = min(abs(p.x - box.minn.x), abs(p.x - box.maxx.x));
	if (p.y < box.minn.y || p.y > box.maxx.y)
		diff.y = min(abs(p.y - box.minn.y), abs(p.y - box.maxx.y));
	if (p.z < box.minn.z || p.z > box.maxx.z)
		diff.z = min(abs(p.z - box.minn.z), abs(p.z - box.maxx.z));
	return diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
}

template<int K>
__device__ void updateKBest(const float3& ref, const float3& point, float* knn)
{
	float3 d = { point.x - ref.x, point.y - ref.y, point.z - ref.z };
	float dist = d.x * d.x + d.y * d.y + d.z * d.z;
	for (int j = 0; j < K; j++)
	{
		if (knn[j] > dist)
		{
			float t = knn[j];
			knn[j] = dist;
			dist = t;
		}
	}
}

__global__ void boxMeanDist(uint32_t P, float3* points, uint32_t* indices, MinMax* boxes, float* dists)
{
	int idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 point = points[indices[idx]];
	float best[3] = { FLT_MAX, FLT_MAX, FLT_MAX };

	for (int i = max(0, idx - 3); i <= min(P - 1, idx + 3); i++)
	{
		if (i == idx)
			continue;
		updateKBest<3>(point, points[indices[i]], best);
	}

	float reject = best[2];
	best[0] = FLT_MAX;
	best[1] = FLT_MAX;
	best[2] = FLT_MAX;

	for (int b = 0; b < (P + BOX_SIZE - 1) / BOX_SIZE; b++)
	{
		MinMax box = boxes[b];
		float dist = distBoxPoint(box, point);
		if (dist > reject || dist > best[2])
			continue;

		for (int i = b * BOX_SIZE; i < min(P, (b + 1) * BOX_SIZE); i++)
		{
			if (i == idx)
				continue;
			updateKBest<3>(point, points[indices[i]], best);
		}
	}
	dists[indices[idx]] = (best[0] + best[1] + best[2]) / 3.0f;
}

void SimpleKNN::knn(int P, float3* points, float* meanDists)
{
	float3* result;
	cudaMalloc(&result, sizeof(float3));
	size_t temp_storage_bytes;

	float3 init = { 0, 0, 0 }, minn, maxx;

	cub::DeviceReduce::Reduce(nullptr, temp_storage_bytes, points, result, P, CustomMin(), init);
	thrust::device_vector<char> temp_storage(temp_storage_bytes);

	cub::DeviceReduce::Reduce(temp_storage.data().get(), temp_storage_bytes, points, result, P, CustomMin(), init);
	cudaMemcpy(&minn, result, sizeof(float3), cudaMemcpyDeviceToHost);

	cub::DeviceReduce::Reduce(temp_storage.data().get(), temp_storage_bytes, points, result, P, CustomMax(), init);
	cudaMemcpy(&maxx, result, sizeof(float3), cudaMemcpyDeviceToHost);

	thrust::device_vector<uint32_t> morton(P);
	thrust::device_vector<uint32_t> morton_sorted(P);
	coord2Morton << <(P + 255) / 256, 256 >> > (P, points, minn, maxx, morton.data().get());

	thrust::device_vector<uint32_t> indices(P);
	thrust::sequence(indices.begin(), indices.end());
	thrust::device_vector<uint32_t> indices_sorted(P);

	cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes, morton.data().get(), morton_sorted.data().get(), indices.data().get(), indices_sorted.data().get(), P);
	temp_storage.resize(temp_storage_bytes);

	cub::DeviceRadixSort::SortPairs(temp_storage.data().get(), temp_storage_bytes, morton.data().get(), morton_sorted.data().get(), indices.data().get(), indices_sorted.data().get(), P);

	uint32_t num_boxes = (P + BOX_SIZE - 1) / BOX_SIZE;
	thrust::device_vector<MinMax> boxes(num_boxes);
	boxMinMax<BOX_SIZE> << <num_boxes, BOX_SIZE >> > (P, points, indices_sorted.data().get(), boxes.data().get());
	boxMeanDist << <num_boxes, BOX_SIZE >> > (P, points, indices_sorted.data().get(), boxes.data().get(), meanDists);

	cudaFree(result);
}


__device__ void updateKBest(int K, int index, const float3& ref, const float3& point, float* knn, int* indices)
{
	float3 d = { point.x - ref.x, point.y - ref.y, point.z - ref.z };
	float dist = d.x * d.x + d.y * d.y + d.z * d.z;
	int ind = index;

	for (int j = 0; j < K; j++)
	{
		if (dist < knn[j])
		{
			float t = knn[j];
			int _i = indices[j];
			knn[j] = dist;
			indices[j] = ind;
			dist = t;
			ind = _i;
		}
	}
}

__global__ void boxKnn(int K, uint32_t P, float3* points, uint32_t* indices, MinMax* boxes, float* dists, int* index_space)
{
	int idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 point = points[indices[idx]];
	
	float* best = dists + indices[idx] * K;
	int* best_ind = index_space + indices[idx] * K;
	for(int i = 0; i < K; i++)
		best[i] = FLT_MAX;

	for (int i = max(0, idx - K); i <= min(P - 1, idx + K); i++)
	{
		if (i == idx)
			continue;
		updateKBest(K, indices[i], point, points[indices[i]], best, best_ind);
	}

	float reject = best[K-1];
	for(int i = 0; i < K; i++)
		best[i] = FLT_MAX;

	for (int b = 0; b < (P + BOX_SIZE - 1) / BOX_SIZE; b++)
	{
		MinMax box = boxes[b];
		float dist = distBoxPoint(box, point);
		if (dist > reject || dist > best[K-1])
			continue;

		for (int i = b * BOX_SIZE; i < min(P, (b + 1) * BOX_SIZE); i++)
		{
			if (i == idx)
				continue;
			updateKBest(K, indices[i], point, points[indices[i]], best, best_ind);
		}
	}
}

void SimpleKNN::knn_index(int K, int P, float3* points, float* dists, int* index_space)
{
	float3* result;
	cudaMalloc(&result, sizeof(float3));
	size_t temp_storage_bytes;

	float3 init = { 0, 0, 0 }, minn, maxx;

	cub::DeviceReduce::Reduce(nullptr, temp_storage_bytes, points, result, P, CustomMin(), init);
	thrust::device_vector<char> temp_storage(temp_storage_bytes);

	cub::DeviceReduce::Reduce(temp_storage.data().get(), temp_storage_bytes, points, result, P, CustomMin(), init);
	cudaMemcpy(&minn, result, sizeof(float3), cudaMemcpyDeviceToHost);

	cub::DeviceReduce::Reduce(temp_storage.data().get(), temp_storage_bytes, points, result, P, CustomMax(), init);
	cudaMemcpy(&maxx, result, sizeof(float3), cudaMemcpyDeviceToHost);

	thrust::device_vector<uint32_t> morton(P);
	thrust::device_vector<uint32_t> morton_sorted(P);
	coord2Morton << <(P + 255) / 256, 256 >> > (P, points, minn, maxx, morton.data().get());

	thrust::device_vector<uint32_t> indices(P);
	thrust::sequence(indices.begin(), indices.end());
	thrust::device_vector<uint32_t> indices_sorted(P);

	cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes, morton.data().get(), morton_sorted.data().get(), indices.data().get(), indices_sorted.data().get(), P);
	temp_storage.resize(temp_storage_bytes);

	cub::DeviceRadixSort::SortPairs(temp_storage.data().get(), temp_storage_bytes, morton.data().get(), morton_sorted.data().get(), indices.data().get(), indices_sorted.data().get(), P);

	// cudaEvent_t ev1, ev2, ev3;
	// cudaEventCreate(&ev1);
	// cudaEventCreate(&ev2);
	// cudaEventCreate(&ev3);

	uint32_t num_boxes = (P + BOX_SIZE - 1) / BOX_SIZE;
	thrust::device_vector<MinMax> boxes(num_boxes);
	// cudaEventRecord(ev1);
	boxMinMax<BOX_SIZE> << <num_boxes, BOX_SIZE >> > (P, points, indices_sorted.data().get(), boxes.data().get());
	// cudaEventRecord(ev2);
	boxKnn << <num_boxes, BOX_SIZE >> > (K, P, points, indices_sorted.data().get(), boxes.data().get(), dists, index_space);
	// cudaEventRecord(ev3);

	// cudaEventSynchronize(ev3);
	// float ms1, ms2;
	// cudaEventElapsedTime(&ms1, ev1, ev2);
	// cudaEventElapsedTime(&ms2, ev2, ev3);

	// std::cout << "First part: " << ms1 << std::endl;
	// std::cout << "Second part: " << ms2 << std::endl;

	cudaFree(result);
}

__device__ float get4FromK(
	int K,
	float4& dist4,
	int4& ind4,
	float* __restrict__ knn,
	int* __restrict__ indices)
{
	dist4 = { FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX };
	ind4 = { -1, -1, -1, -1 };
	for (int j = 0; j < K; j++)
	{
		float v = knn[j];
		int ind = indices[j];
		float w = v;
		int indx = ind;
		if (v < dist4.w)
		{
			if (v < dist4.z)
			{
				if (v < dist4.y)
				{
					if (v < dist4.x)
					{
						w = dist4.x;
						indx = ind4.x;
						dist4.x = v;
						ind4.x = ind;
						v = w;
						ind = indx;
					}
					w = dist4.y;
					indx = ind4.y;
					dist4.y = v;
					ind4.y = ind;
					v = w;
					ind = indx;
				}
				w = dist4.z;
				indx = ind4.z;
				dist4.z = v;
				ind4.z = ind;
				v = w;
				ind = indx;
			}
			dist4.w = v;
			ind4.w = ind;
		}
	}
}

__device__ void updateKBest2(
	float& reject,
	int K,
	int index,
	const float3& ref,
	const float3& point,
	float* __restrict__ knn,
	int* __restrict__ indices)
{
	float3 d = { point.x - ref.x, point.y - ref.y, point.z - ref.z };
	float dist = d.x * d.x + d.y * d.y + d.z * d.z;
	if (dist >= reject)
		return;

	float test_reject = dist;
	int maxint = -1;
	for (int j = 0; j < K; j++)
	{
		if (test_reject < knn[j])
		{
			test_reject = knn[j];
			maxint = j;
		}
	}
	if (maxint != -1)
	{
		knn[maxint] = dist;
		indices[maxint] = index;
	}
	reject = min(reject, test_reject);
}

__global__ void boxKnn2(
	int K,
	uint32_t P,
	const float3* __restrict__ points,
	const uint32_t* __restrict__ indices,
	const MinMax* __restrict__ boxes,
	float* __restrict__ dists,
	int* __restrict__ index_space)
{
	int idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	const float3 point = points[indices[idx]];

	float* best = dists + indices[idx] * K;
	int* best_ind = index_space + indices[idx] * K;
	for (int i = 0; i < K; i++)
		best[i] = FLT_MAX;

	float reject = FLT_MAX;
	int b = idx / BOX_SIZE2;
	int lo = b, hi = b;

	const int num_boxes = (P + BOX_SIZE2 - 1) / BOX_SIZE2;

	for (int iter = 0; iter < num_boxes; iter++)
	{
		MinMax box = boxes[b];
		float dist = distBoxPoint(box, point);
		if (dist < reject)
		{
			for (int i = b * BOX_SIZE2; i < min(P, (b + 1) * BOX_SIZE2); i++)
			{
				if (i == idx)
					continue;
				const int other_idx = indices[i];
				updateKBest2(reject, K, other_idx, point, points[other_idx], best, best_ind);
			}
		}
		bool odd = iter & 1;
		b = (odd && hi == num_boxes - 1) || (!odd && lo > 0) ? --lo : ++hi;
	}
}

void SimpleKNN::knn_index2(int K, int P, float3* points, float* dists, int* index_space)
{
	float3* result;
	cudaMalloc(&result, sizeof(float3));
	size_t temp_storage_bytes;

	float3 init = { 0, 0, 0 }, minn, maxx;

	cub::DeviceReduce::Reduce(nullptr, temp_storage_bytes, points, result, P, CustomMin(), init);
	thrust::device_vector<char> temp_storage(temp_storage_bytes);

	cub::DeviceReduce::Reduce(temp_storage.data().get(), temp_storage_bytes, points, result, P, CustomMin(), init);
	cudaMemcpy(&minn, result, sizeof(float3), cudaMemcpyDeviceToHost);

	cub::DeviceReduce::Reduce(temp_storage.data().get(), temp_storage_bytes, points, result, P, CustomMax(), init);
	cudaMemcpy(&maxx, result, sizeof(float3), cudaMemcpyDeviceToHost);

	thrust::device_vector<uint32_t> morton(P);
	thrust::device_vector<uint32_t> morton_sorted(P);
	coord2Morton << <(P + 255) / 256, 256 >> > (P, points, minn, maxx, morton.data().get());

	thrust::device_vector<uint32_t> indices(P);
	thrust::sequence(indices.begin(), indices.end());
	thrust::device_vector<uint32_t> indices_sorted(P);

	cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes, morton.data().get(), morton_sorted.data().get(), indices.data().get(), indices_sorted.data().get(), P);
	temp_storage.resize(temp_storage_bytes);

	cub::DeviceRadixSort::SortPairs(temp_storage.data().get(), temp_storage_bytes, morton.data().get(), morton_sorted.data().get(), indices.data().get(), indices_sorted.data().get(), P);

	// cudaEvent_t ev1, ev2, ev3;
	// cudaEventCreate(&ev1);
	// cudaEventCreate(&ev2);
	// cudaEventCreate(&ev3);

	uint32_t num_boxes2 = (P + BOX_SIZE2 - 1) / BOX_SIZE2;
	thrust::device_vector<MinMax> boxes(num_boxes2);
	// cudaEventRecord(ev1);
	boxMinMax<BOX_SIZE2> << <num_boxes2, BOX_SIZE2 >> > (P, points, indices_sorted.data().get(), boxes.data().get());
	// cudaEventRecord(ev2);

	int num_blocks = (P + 255) / 256;
	boxKnn2 << <num_blocks, 256>> > (K, P, points, indices_sorted.data().get(), boxes.data().get(), dists, index_space);
	// cudaEventRecord(ev3);

	// cudaEventSynchronize(ev3);
	// float ms1, ms2;
	// cudaEventElapsedTime(&ms1, ev1, ev2);
	// cudaEventElapsedTime(&ms2, ev2, ev3);

	// std::cout << "First part: " << ms1 << std::endl;
	// std::cout << "Second part: " << ms2 << std::endl;

	cudaFree(result);
}

__global__ void boxKnnQ(
	int K,
	uint32_t P,
	const float3* __restrict__ points,
	uint32_t Q,
	const int* __restrict__ query_indices,
	const bool* __restrict__ is_neighbor,
	const uint32_t* __restrict__ indices,
	const uint32_t* __restrict__ i2p,
	const MinMax* __restrict__ boxes,
	float* __restrict__ dists,
	int* __restrict__ index_space)
{
	int q_idx = cg::this_grid().thread_rank();
	if (q_idx >= Q)
		return;

	int i_idx = query_indices[q_idx];
	const float3 point = points[i_idx];

	float* best = dists + q_idx * K;
	int* best_ind = index_space + q_idx * K;
	for (int i = 0; i < K; i++)
		best[i] = FLT_MAX;
	float reject = FLT_MAX;

	int p_idx = i2p[i_idx];

	int b = p_idx / BOX_SIZE2;
	int lo = b, hi = b;

	const int num_boxes = (P + BOX_SIZE2 - 1) / BOX_SIZE2;

	for (int iter = 0; iter < num_boxes; iter++)
	{
		MinMax box = boxes[b];
		float dist = distBoxPoint(box, point);
		if (dist < reject)
		{
			for (int i = b * BOX_SIZE2; i < min(P, (b + 1) * BOX_SIZE2); i++)
			{
				if (i == p_idx)
					continue;
				const int other_idx = indices[i];

				if(is_neighbor[other_idx])
					updateKBest2(reject, K, other_idx, point, points[other_idx], best, best_ind);
			}
		}
		bool odd = iter & 1;
		b = (odd && hi == num_boxes - 1) || (!odd && lo > 0) ? --lo : ++hi;
	}
}

__global__ void fillIndex2Pos(int P, int N, int* neighbor_indices, bool* __restrict__ is_neighbor, uint32_t* __restrict__ ind_sorted, uint32_t* __restrict__ ind2pos)
{
	int idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;
	int ind = ind_sorted[idx];
	ind2pos[ind] = idx;

	if (idx < N)
	{
		is_neighbor[neighbor_indices[idx]] = true;
	}
}

void SimpleKNN::knn_indexQ(int K, int P, float3* points, int Q, int* query_indices, int N, int* neighbor_indices, float* dists, int* index_space)
{
	float3* result;
	cudaMalloc(&result, sizeof(float3));
	size_t temp_storage_bytes;

	float3 init = { 0, 0, 0 }, minn, maxx;

	cub::DeviceReduce::Reduce(nullptr, temp_storage_bytes, points, result, P, CustomMin(), init);
	thrust::device_vector<char> temp_storage(temp_storage_bytes);

	cub::DeviceReduce::Reduce(temp_storage.data().get(), temp_storage_bytes, points, result, P, CustomMin(), init);
	cudaMemcpy(&minn, result, sizeof(float3), cudaMemcpyDeviceToHost);

	cub::DeviceReduce::Reduce(temp_storage.data().get(), temp_storage_bytes, points, result, P, CustomMax(), init);
	cudaMemcpy(&maxx, result, sizeof(float3), cudaMemcpyDeviceToHost);

	thrust::device_vector<uint32_t> morton(P);
	thrust::device_vector<uint32_t> morton_sorted(P);
	coord2Morton << <(P + 255) / 256, 256 >> > (P, points, minn, maxx, morton.data().get());

	thrust::device_vector<uint32_t> indices(P);
	thrust::sequence(indices.begin(), indices.end());
	thrust::device_vector<uint32_t> indices_sorted(P);

	cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes, morton.data().get(), morton_sorted.data().get(), indices.data().get(), indices_sorted.data().get(), P);
	temp_storage.resize(temp_storage_bytes);

	cub::DeviceRadixSort::SortPairs(temp_storage.data().get(), temp_storage_bytes, morton.data().get(), morton_sorted.data().get(), indices.data().get(), indices_sorted.data().get(), P);

	// cudaEvent_t ev1, ev2, ev3;
	// cudaEventCreate(&ev1);
	// cudaEventCreate(&ev2);
	// cudaEventCreate(&ev3);

	uint32_t num_boxes2 = (P + BOX_SIZE2 - 1) / BOX_SIZE2;
	thrust::device_vector<MinMax> boxes(num_boxes2);
	// cudaEventRecord(ev1);
	boxMinMax<BOX_SIZE2> << <num_boxes2, BOX_SIZE2 >> > (P, points, indices_sorted.data().get(), boxes.data().get());
	// cudaEventRecord(ev2);

	int num_blocks = (P + 255) / 256;
	thrust::device_vector<uint32_t> index2pos(P);
	thrust::device_vector<bool> is_neighbor(P, false);
	fillIndex2Pos << <num_blocks, 256 >> > (P, N, neighbor_indices, is_neighbor.data().get(), indices_sorted.data().get(), index2pos.data().get());

	int num_blocks2 = (Q + 255) / 256;
	boxKnnQ << <num_blocks2, 256 >> > (K, P, points, Q, query_indices, is_neighbor.data().get(), indices_sorted.data().get(), index2pos.data().get(), boxes.data().get(), dists, index_space);
	// cudaEventRecord(ev3);

	// cudaEventSynchronize(ev3);
	// float ms1, ms2;
	// cudaEventElapsedTime(&ms1, ev1, ev2);
	// cudaEventElapsedTime(&ms2, ev2, ev3);

	// std::cout << "First part: " << ms1 << std::endl;
	// std::cout << "Second part: " << ms2 << std::endl;

	cudaFree(result);
}
