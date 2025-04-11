// corerec/csrc/ops/embedding_ops.h
#pragma once

#include "../tensor/tensor.h"

namespace corerec {

/**
 * Fast embedding lookup operation.
 * 
 * @param weight The embedding table (2D tensor)
 * @param indices The indices to look up (1D tensor)
 * @return A tensor containing the embeddings for each index
 */
Tensor embedding_lookup(const Tensor& weight, const Tensor& indices);

/**
 * Fast embedding bag operation (lookup with pooling).
 * 
 * @param weight The embedding table (2D tensor)
 * @param indices The indices to look up (1D tensor)
 * @param offsets The offsets for pooling (1D tensor)
 * @param mode Pooling mode (0=sum, 1=mean, 2=max)
 * @return A tensor containing the pooled embeddings
 */
Tensor embedding_bag(const Tensor& weight, const Tensor& indices, 
                    const Tensor& offsets, int mode);

} // namespace corerec