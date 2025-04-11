#include "Python.h"
#include "numpy/ndarrayobject.h"

// corerec/csrc/python/module.cpp
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include "../tensor/tensor.h"
#include "../ops/embedding_ops.h"
#include "module.h"

// Convert NumPy array to CoreRec Tensor
corerec::Tensor numpy_to_tensor(PyArrayObject* array) {
    // Get array info
    int ndim = PyArray_NDIM(array);
    npy_intp* dims = PyArray_DIMS(array);
    int dtype = PyArray_TYPE(array);
    void* data = PyArray_DATA(array);
    
    // Convert dimensions
    std::vector<int64_t> shape(ndim);
    for (int i = 0; i < ndim; ++i) {
        shape[i] = static_cast<int64_t>(dims[i]);
    }
    
    // Create tensor based on dtype
    corerec::DataType tensor_dtype;
    switch (dtype) {
        case NPY_FLOAT32:
            tensor_dtype = corerec::DataType::FLOAT32;
            return corerec::Tensor(shape, static_cast<float*>(data));
        case NPY_FLOAT64:
            tensor_dtype = corerec::DataType::FLOAT64;
            return corerec::Tensor(shape, static_cast<double*>(data));
        case NPY_INT32:
            tensor_dtype = corerec::DataType::INT32;
            return corerec::Tensor(shape, static_cast<int32_t*>(data));
        case NPY_INT64:
            tensor_dtype = corerec::DataType::INT64;
            return corerec::Tensor(shape, static_cast<int64_t*>(data));
        default:
            throw std::runtime_error("Unsupported NumPy data type");
    }
}

// Convert CoreRec Tensor to NumPy array
PyObject* tensor_to_numpy(const corerec::Tensor& tensor) {
    // Get tensor info
    const std::vector<int64_t>& shape = tensor.shape();
    corerec::DataType dtype = tensor.dtype();
    
    // Convert shape to NumPy dimensions
    npy_intp dims[shape.size()];
    for (size_t i = 0; i < shape.size(); ++i) {
        dims[i] = static_cast<npy_intp>(shape[i]);
    }
    
    // Create NumPy array based on dtype
    int numpy_dtype;
    PyObject* array = nullptr;
    
    switch (dtype) {
        case corerec::DataType::FLOAT32:
            numpy_dtype = NPY_FLOAT32;
            array = PyArray_SimpleNew(shape.size(), dims, numpy_dtype);
            std::memcpy(PyArray_DATA((PyArrayObject*)array), 
                       tensor.data<float>(), 
                       tensor.size() * sizeof(float));
            break;
        case corerec::DataType::FLOAT64:
            numpy_dtype = NPY_FLOAT64;
            array = PyArray_SimpleNew(shape.size(), dims, numpy_dtype);
            std::memcpy(PyArray_DATA((PyArrayObject*)array), 
                       tensor.data<double>(), 
                       tensor.size() * sizeof(double));
            break;
        case corerec::DataType::INT32:
            numpy_dtype = NPY_INT32;
            array = PyArray_SimpleNew(shape.size(), dims, numpy_dtype);
            std::memcpy(PyArray_DATA((PyArrayObject*)array), 
                       tensor.data<int32_t>(), 
                       tensor.size() * sizeof(int32_t));
            break;
        case corerec::DataType::INT64:
            numpy_dtype = NPY_INT64;
            array = PyArray_SimpleNew(shape.size(), dims, numpy_dtype);
            std::memcpy(PyArray_DATA((PyArrayObject*)array), 
                       tensor.data<int64_t>(), 
                       tensor.size() * sizeof(int64_t));
            break;
        default:
            throw std::runtime_error("Unsupported tensor data type");
    }
    
    return array;
}

// Python wrapper for embedding_lookup
static PyObject* py_embedding_lookup(PyObject* self, PyObject* args) {
    PyArrayObject *weight_array, *indices_array;
    
    // Parse arguments
    if (!PyArg_ParseTuple(args, "O!O!", 
                         &PyArray_Type, &weight_array,
                         &PyArray_Type, &indices_array)) {
        return NULL;
    }
    
    try {
        // Convert NumPy arrays to tensors
        corerec::Tensor weight = numpy_to_tensor(weight_array);
        corerec::Tensor indices = numpy_to_tensor(indices_array);
        
        // Perform embedding lookup
        corerec::Tensor result = corerec::embedding_lookup(weight, indices);
        
        // Convert result back to NumPy array
        return tensor_to_numpy(result);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

// Python wrapper for embedding_bag
static PyObject* py_embedding_bag(PyObject* self, PyObject* args) {
    PyArrayObject *weight_array, *indices_array, *offsets_array;
    int mode;
    
    // Parse arguments
    if (!PyArg_ParseTuple(args, "O!O!O!i", 
                         &PyArray_Type, &weight_array,
                         &PyArray_Type, &indices_array,
                         &PyArray_Type, &offsets_array,
                         &mode)) {
        return NULL;
    }
    
    try {
        // Convert NumPy arrays to tensors
        corerec::Tensor weight = numpy_to_tensor(weight_array);
        corerec::Tensor indices = numpy_to_tensor(indices_array);
        corerec::Tensor offsets = numpy_to_tensor(offsets_array);
        
        // Perform embedding bag operation
        corerec::Tensor result = corerec::embedding_bag(weight, indices, offsets, mode);
        
        // Convert result back to NumPy array
        return tensor_to_numpy(result);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

// Method definitions
static PyMethodDef FastEmbeddingMethods[] = {
    {"embedding_lookup", py_embedding_lookup, METH_VARARGS, 
     "Fast embedding lookup operation"},
    {"embedding_bag", py_embedding_bag, METH_VARARGS, 
     "Fast embedding bag operation (lookup with pooling)"},
    {NULL, NULL, 0, NULL}  // Sentinel
};

// Module definition
static struct PyModuleDef fast_embedding_module = {
    PyModuleDef_HEAD_INIT,
    "fast_embedding_ops",   // Module name
    "Fast embedding operations for CoreRec",  // Module docstring
    -1,                     // Size of per-interpreter state or -1
    FastEmbeddingMethods    // Method definitions
};

// Module initialization function
PyMODINIT_FUNC PyInit_fast_embedding_ops(void) {
    import_array();  // Initialize NumPy
    return PyModule_Create(&fast_embedding_module);
}

// Add the PyInit_tensor_ops function that Python is looking for
PyMODINIT_FUNC PyInit_tensor_ops(void) {
    // Return the module created by your existing initialization code
    // If you already have a function that creates the module, call it here
    // Otherwise, implement the module creation logic in this function
    return PyModule_Create(&fast_embedding_module);  // Replace moduledef with your module definition
}
