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

#ifndef SIMPLEKNN_H_INCLUDED
#define SIMPLEKNN_H_INCLUDED

class SimpleKNN
{
public:
	static void knn(int P, float3* points, float* meanDists);
	static void knn_index(int K, int P, float3* points, float* dists, int* indices);
	static void knn_index2(int K, int P, float3* points, float* dists, int* indices);
	static void knn_indexQ(int K, int P, float3* points, int Q, int* query_indices, int N, int* neighbor_indices, float* dists, int* index_space);
};

#endif