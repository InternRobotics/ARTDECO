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

#include <torch/extension.h>

torch::Tensor distCUDA2(const torch::Tensor& points);
std::vector<torch::Tensor> distCudaIndices2(const torch::Tensor& points, int K);
std::vector<torch::Tensor> distCudaIndicesQ(const torch::Tensor& points, const torch::Tensor& q_indices, const torch::Tensor& n_indices, int K);