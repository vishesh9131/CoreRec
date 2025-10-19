// corerec/csrc/tensor/tensor.cpp
#include "tensor.h"
#include <numeric>
#include <functional>

namespace corerec {

Tensor::Tensor(const std::vector<int64_t>& shape, DataType dtype) 
    : shape_(shape), dtype_(dtype) {
    
    // Calculate total size
    size_ = std::accumulate(shape.begin(), shape.end(), 
                           static_cast<size_t>(1), std::multiplies<size_t>());
    
    // Allocate memory based on data type
    switch (dtype) {
        case DataType::FLOAT32:
            data_ = std::shared_ptr<void>(new float[size_], [](void* p) { delete[] static_cast<float*>(p); });
            break;
        case DataType::FLOAT64:
            data_ = std::shared_ptr<void>(new double[size_], [](void* p) { delete[] static_cast<double*>(p); });
            break;
        case DataType::INT32:
            data_ = std::shared_ptr<void>(new int32_t[size_], [](void* p) { delete[] static_cast<int32_t*>(p); });
            break;
        case DataType::INT64:
            data_ = std::shared_ptr<void>(new int64_t[size_], [](void* p) { delete[] static_cast<int64_t*>(p); });
            break;
        default:
            throw std::runtime_error("Unsupported data type");
    }
}

template<typename T>
Tensor::Tensor(const std::vector<int64_t>& shape, const T* data) 
    : shape_(shape), dtype_(get_dtype<T>()) {
    
    // Calculate total size
    size_ = std::accumulate(shape.begin(), shape.end(), 
                           static_cast<size_t>(1), std::multiplies<size_t>());
    
    // Allocate and copy data
    T* new_data = new T[size_];
    std::copy(data, data + size_, new_data);
    data_ = std::shared_ptr<void>(new_data, [](void* p) { delete[] static_cast<T*>(p); });
}

size_t Tensor::offset(const std::vector<int64_t>& indices) const {
    if (indices.size() != shape_.size()) {
        throw std::runtime_error("Indices dimension mismatch");
    }
    
    size_t offset = 0;
    size_t stride = 1;
    
    for (int i = shape_.size() - 1; i >= 0; --i) {
        if (indices[i] < 0 || indices[i] >= shape_[i]) {
            throw std::out_of_range("Index out of bounds");
        }
        offset += indices[i] * stride;
        stride *= shape_[i];
    }
    
    return offset;
}

// Explicit template instantiations
template Tensor::Tensor(const std::vector<int64_t>&, const float*);
template Tensor::Tensor(const std::vector<int64_t>&, const double*);
template Tensor::Tensor(const std::vector<int64_t>&, const int32_t*);
template Tensor::Tensor(const std::vector<int64_t>&, const int64_t*);

} // namespace corerec