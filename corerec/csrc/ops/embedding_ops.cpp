// corerec/csrc/ops/embedding_ops.cpp
#include "embedding_ops.h"
#include <algorithm>
#include <limits>

namespace corerec {

Tensor embedding_lookup(const Tensor& weight, const Tensor& indices) {
    // Validate inputs
    if (weight.shape().size() != 2) {
        throw std::runtime_error("Weight must be a 2D tensor");
    }
    if (indices.shape().size() != 1) {
        throw std::runtime_error("Indices must be a 1D tensor");
    }
    if (indices.dtype() != DataType::INT64 && indices.dtype() != DataType::INT32) {
        throw std::runtime_error("Indices must be integer type");
    }
    
    // Get dimensions
    int64_t num_embeddings = weight.shape()[0];
    int64_t embedding_dim = weight.shape()[1];
    int64_t num_indices = indices.shape()[0];
    
    // Create output tensor
    std::vector<int64_t> output_shape = {num_indices, embedding_dim};
    Tensor output(output_shape, weight.dtype());
    
    // Perform lookup based on data type
    if (weight.dtype() == DataType::FLOAT32) {
        const float* weight_data = weight.data<float>();
        float* output_data = output.data<float>();
        
        if (indices.dtype() == DataType::INT64) {
            const int64_t* indices_data = indices.data<int64_t>();
            
            // Perform lookup
            for (int64_t i = 0; i < num_indices; ++i) {
                int64_t idx = indices_data[i];
                if (idx < 0 || idx >= num_embeddings) {
                    throw std::out_of_range("Embedding index out of bounds");
                }
                
                // Copy embedding
                std::copy(
                    weight_data + idx * embedding_dim,
                    weight_data + (idx + 1) * embedding_dim,
                    output_data + i * embedding_dim
                );
            }
        } else { // INT32
            const int32_t* indices_data = indices.data<int32_t>();
            
            // Perform lookup
            for (int64_t i = 0; i < num_indices; ++i) {
                int32_t idx = indices_data[i];
                if (idx < 0 || idx >= num_embeddings) {
                    throw std::out_of_range("Embedding index out of bounds");
                }
                
                // Copy embedding
                std::copy(
                    weight_data + idx * embedding_dim,
                    weight_data + (idx + 1) * embedding_dim,
                    output_data + i * embedding_dim
                );
            }
        }
    } else if (weight.dtype() == DataType::FLOAT64) {
        // Similar implementation for double
        // ...
    } else {
        throw std::runtime_error("Unsupported weight data type");
    }
    
    return output;
}

Tensor embedding_bag(const Tensor& weight, const Tensor& indices, 
                    const Tensor& offsets, int mode) {
    // Validate inputs
    if (weight.shape().size() != 2) {
        throw std::runtime_error("Weight must be a 2D tensor");
    }
    if (indices.shape().size() != 1) {
        throw std::runtime_error("Indices must be a 1D tensor");
    }
    if (offsets.shape().size() != 1) {
        throw std::runtime_error("Offsets must be a 1D tensor");
    }
    if (indices.dtype() != DataType::INT64 && indices.dtype() != DataType::INT32) {
        throw std::runtime_error("Indices must be integer type");
    }
    if (offsets.dtype() != DataType::INT64 && offsets.dtype() != DataType::INT32) {
        throw std::runtime_error("Offsets must be integer type");
    }
    
    // Get dimensions
    int64_t num_embeddings = weight.shape()[0];
    int64_t embedding_dim = weight.shape()[1];
    int64_t num_bags = offsets.shape()[0] - 1; // Last offset is the end
    
    // Create output tensor
    std::vector<int64_t> output_shape = {num_bags, embedding_dim};
    Tensor output(output_shape, weight.dtype());
    
    // Perform embedding bag operation based on data type
    if (weight.dtype() == DataType::FLOAT32) {
        const float* weight_data = weight.data<float>();
        float* output_data = output.data<float>();
        
        // Initialize output to zeros
        std::fill(output_data, output_data + num_bags * embedding_dim, 0.0f);
        
        // Get offsets data
        std::vector<int64_t> offset_values(num_bags + 1);
        if (offsets.dtype() == DataType::INT64) {
            const int64_t* offsets_data = offsets.data<int64_t>();
            std::copy(offsets_data, offsets_data + num_bags + 1, offset_values.begin());
        } else { // INT32
            const int32_t* offsets_data = offsets.data<int32_t>();
            for (int64_t i = 0; i <= num_bags; ++i) {
                offset_values[i] = static_cast<int64_t>(offsets_data[i]);
            }
        }
        
        // Process each bag
        for (int64_t bag = 0; bag < num_bags; ++bag) {
            int64_t start_offset = offset_values[bag];
            int64_t end_offset = offset_values[bag + 1];
            int64_t bag_size = end_offset - start_offset;
            
            if (bag_size == 0) {
                continue; // Empty bag
            }
            
            // For max pooling, initialize with minimum value
            if (mode == 2) { // max
                std::fill(
                    output_data + bag * embedding_dim,
                    output_data + (bag + 1) * embedding_dim,
                    -std::numeric_limits<float>::infinity()
                );
            }
            
            // Process indices in this bag
            for (int64_t idx_pos = start_offset; idx_pos < end_offset; ++idx_pos) {
                int64_t idx;
                if (indices.dtype() == DataType::INT64) {
                    idx = indices.data<int64_t>()[idx_pos];
                } else { // INT32
                    idx = static_cast<int64_t>(indices.data<int32_t>()[idx_pos]);
                }
                
                if (idx < 0 || idx >= num_embeddings) {
                    throw std::out_of_range("Embedding index out of bounds");
                }
                
                // Get embedding
                const float* embedding = weight_data + idx * embedding_dim;
                
                // Apply pooling
                if (mode == 0) { // sum
                    for (int64_t d = 0; d < embedding_dim; ++d) {
                        output_data[bag * embedding_dim + d] += embedding[d];
                    }
                } else if (mode == 1) { // mean
                    for (int64_t d = 0; d < embedding_dim; ++d) {
                        output_data[bag * embedding_dim + d] += embedding[d] / bag_size;
                    }
                } else if (mode == 2) { // max
                    for (int64_t d = 0; d < embedding_dim; ++d) {
                        output_data[bag * embedding_dim + d] = 
                            std::max(output_data[bag * embedding_dim + d], embedding[d]);
                    }
                }
            }
        }
    } else if (weight.dtype() == DataType::FLOAT64) {
        // Similar implementation for double
        // ...
    } else {
        throw std::runtime_error("Unsupported weight data type");
    }
    
    return output;
}

} // namespace corerec