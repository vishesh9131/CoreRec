// corerec/csrc/tensor/tensor.h
#pragma once

#include <vector>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

namespace corerec {

enum class DataType {
    FLOAT32,
    FLOAT64,
    INT32,
    INT64
};

// Forward declaration of get_dtype template function
template<typename T>
DataType get_dtype();

class Tensor {
private:
    std::vector<int64_t> shape_;
    DataType dtype_;
    std::shared_ptr<void> data_;
    size_t size_;

public:
    // Constructor for creating a tensor with given shape and type
    Tensor(const std::vector<int64_t>& shape, DataType dtype);
    
    // Constructor from raw data
    template<typename T>
    Tensor(const std::vector<int64_t>& shape, const T* data);
    
    // Destructor
    ~Tensor() = default;
    
    // Get shape
    const std::vector<int64_t>& shape() const { return shape_; }
    
    // Get data type
    DataType dtype() const { return dtype_; }
    
    // Get raw data pointer
    template<typename T>
    T* data() {
        return static_cast<T*>(data_.get());
    }
    
    template<typename T>
    const T* data() const {
        return static_cast<const T*>(data_.get());
    }
    
    // Get total number of elements
    size_t size() const { return size_; }
    
    // Calculate element offset from indices
    size_t offset(const std::vector<int64_t>& indices) const;
    
    // Get element at indices
    template<typename T>
    T& at(const std::vector<int64_t>& indices) {
        return data<T>()[offset(indices)];
    }
    
    template<typename T>
    const T& at(const std::vector<int64_t>& indices) const {
        return data<T>()[offset(indices)];
    }
};

// Template specializations for get_dtype
template<> inline DataType get_dtype<float>() { return DataType::FLOAT32; }
template<> inline DataType get_dtype<double>() { return DataType::FLOAT64; }
template<> inline DataType get_dtype<int32_t>() { return DataType::INT32; }
template<> inline DataType get_dtype<int64_t>() { return DataType::INT64; }

} // namespace corerec