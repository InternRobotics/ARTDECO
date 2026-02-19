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

#include "spatial.h"
#include "simple_knn.h"

torch::Tensor
distCUDA2(const torch::Tensor& points)
{
  const int P = points.size(0);

  auto float_opts = points.options().dtype(torch::kFloat32);
  torch::Tensor means = torch::full({P}, 0.0, float_opts);
  
  SimpleKNN::knn(P, (float3*)points.contiguous().data<float>(), means.contiguous().data<float>());

  return means;
}

std::vector<torch::Tensor>
distCudaIndices2(const torch::Tensor& points, int K)
{
  const int P = points.size(0);

  auto float_opts = points.options().dtype(torch::kFloat32);
  auto int_opts = points.options().dtype(torch::kInt32);
  torch::Tensor dists = torch::full({P * K}, 0.0, float_opts);
  torch::Tensor indices = torch::full({P * K}, -1, int_opts);
  
  SimpleKNN::knn_index2(K, P, (float3*)points.contiguous().data<float>(), dists.contiguous().data<float>(), indices.contiguous().data<int>());

  return {dists, indices};
}

std::vector<torch::Tensor> 
distCudaIndicesQ(const torch::Tensor& points, const torch::Tensor& q_indices, const torch::Tensor& n_indices, int K)
{
  const int P = points.size(0);
  const int Q = q_indices.size(0);
  const int N = n_indices.size(0);

  auto float_opts = points.options().dtype(torch::kFloat32);
  auto int_opts = points.options().dtype(torch::kInt32);
  torch::Tensor dists = torch::full({Q * K}, 0.0, float_opts);
  torch::Tensor indices = torch::full({Q * K}, -1, int_opts);
  
  SimpleKNN::knn_indexQ(K, P, (float3*)points.contiguous().data<float>(), Q, q_indices.contiguous().data<int>(), N, n_indices.contiguous().data<int>(), dists.contiguous().data<float>(), indices.contiguous().data<int>());

  return {dists, indices};
}
