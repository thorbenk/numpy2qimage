#define Py_ARRAY_UNIQUE_SYMBOL numpy2qimage_array

#include <Python.h>
#include <vigra/numpy_array.hxx> /*included first to avoid redefinition warnings*/

#include "numpy2qimage.h"

#include <iostream>

#include <QtGui/QImage>

extern "C" {
    void numpy2qimage_import_array() {
        _import_array();
    }
}

void array2gray_uint8(PyArrayObject* array, QImage& img, unsigned char l, unsigned char h) {
    unsigned char* data = static_cast<unsigned char*>(PyArray_DATA(array));
    unsigned char* imgData = img.bits();
    const unsigned char* dataEnd = data+PyArray_SIZE(array);
    unsigned char pixel;
    if(l == 0 && h == 255) {
        while(data < dataEnd) {
            pixel = *data;
            *imgData = pixel; ++imgData; //B
            *imgData = pixel; ++imgData; //G
            *imgData = pixel; ++imgData; //R
            *imgData = 255;   ++imgData; //A
            ++data;
        }
    }
    else {
        const float div = static_cast<float>(h-l);
        while(data < dataEnd) {
            pixel = *data;
            
            if(pixel < l) pixel = l;
            else if(pixel > h) pixel = h;
            
            pixel = 255*(pixel-l)/div;
            *imgData = pixel; ++imgData; //B
            *imgData = pixel; ++imgData; //G
            *imgData = pixel; ++imgData; //R
            *imgData = 255;   ++imgData; //A
            ++data;
        }
    }
}

void array2gray_float32(PyArrayObject* array, QImage& img, float l, float h) {
    float* data = static_cast<float*>(PyArray_DATA(array));
    unsigned char* imgData = img.bits();
    const float* dataEnd = data+PyArray_SIZE(array);
    
    float pixelF;
    unsigned  char pixel;

    const float div = static_cast<float>(h-l);
    while(data < dataEnd) {
        pixelF = *data;
        
        if(pixelF < l) pixelF = l;
        else if(pixelF > h) pixelF = h;
        
        pixel = 255*(pixelF-l)/div;
        *imgData = pixel; ++imgData; //B
        *imgData = pixel; ++imgData; //G
        *imgData = pixel; ++imgData; //R
        *imgData = 255;   ++imgData; //A
        ++data;
    }
}

void array2alphamodulated_float32(PyArrayObject* array, QImage& img, float r, float g, float b, float l, float h) {
    float* data = static_cast<float*>(PyArray_DATA(array));
    unsigned char* imgData = img.bits();
    const float* dataEnd = data+PyArray_SIZE(array);
    float pixelF;
    const float div = static_cast<float>(h-l);
    while(data < dataEnd) {
        pixelF = *data;
        if(pixelF < l) pixelF = l;
        else if(pixelF > h) pixelF = h;
        pixelF = (pixelF-l)/div;
        *imgData = 255*pixelF*b; ++imgData; //B
        *imgData = 255*pixelF*g; ++imgData; //G
        *imgData = 255*pixelF*r; ++imgData; //R
        *imgData = 255*pixelF;   ++imgData; //A
        ++data;
    }
}

void array2alphamodulated_uint8(PyArrayObject* array, QImage& img, float r, float g, float b, unsigned char l, unsigned char h) {
    unsigned char* data = static_cast<unsigned char*>(PyArray_DATA(array));
    unsigned char* imgData = img.bits();
    const unsigned char* dataEnd = data+PyArray_SIZE(array);
    float pixelF;
    const float div = static_cast<float>(h-l);
    while(data < dataEnd) {
        pixelF = *data;
        if(pixelF < l) pixelF = l;
        else if(pixelF > h) pixelF = h;
        pixelF = (pixelF-l)/div;
        *imgData = 255*pixelF*b; ++imgData; //B
        *imgData = 255*pixelF*g; ++imgData; //G
        *imgData = 255*pixelF*r; ++imgData; //R
        *imgData = 255*pixelF;   ++imgData; //A
        ++data;
    }
}

void Converters::array2gray(PyObject* a, QImage& img, int normalizeLow=0, int normalizeHigh=255) {
    if(!a) { throw std::runtime_error("a = 0"); }
    if(!PyArray_Check(a)) { throw std::runtime_error("is not a pyarray"); }
    PyArrayObject* array = (PyArrayObject*) a;
    if(array == 0) { throw std::runtime_error("Could not convert to PyArrayObject"); }
    int nd = PyArray_NDIM(array);
    if(nd != 2) { throw std::runtime_error("array has dimension != 2"); }
    PyArray_Descr* desc = PyArray_DESCR(array);
    npy_intp* shape = PyArray_DIMS(array);
    if(shape[0] != img.height() || shape[1] != img.width()) {
        throw std::runtime_error("wrong shape");
    }
    if(!PyArray_ISCONTIGUOUS(array)) {
        throw std::runtime_error("array is not C-contiguous");
    }
    
    switch(desc->type_num) {
        case NPY_UINT8:
            array2gray_uint8(array, img, normalizeLow, normalizeHigh);
            break;
        case NPY_FLOAT32:
            array2gray_float32(array, img, normalizeLow, normalizeHigh);
            break;
        default:
            throw std::runtime_error("dtype is not uint8 or float32");
    }
}

void Converters::array2alphamodulated(PyObject* a, QImage& img, float r, float g, float b, int normalizeLow, int normalizeHigh) {
    if(!a) { throw std::runtime_error("a = 0"); }
    if(!PyArray_Check(a)) { throw std::runtime_error("is not a pyarray"); }
    PyArrayObject* array = (PyArrayObject*) a;
    if(array == 0) { throw std::runtime_error("Could not convert to PyArrayObject"); }
    int nd = PyArray_NDIM(array);
    if(nd != 2) { throw std::runtime_error("array has dimension != 2"); }
    PyArray_Descr* desc = PyArray_DESCR(array);
    npy_intp* shape = PyArray_DIMS(array);
    if(shape[0] != img.height() || shape[1] != img.width()) {
        throw std::runtime_error("wrong shape");
    }
    if(!PyArray_ISCONTIGUOUS(array)) {
        throw std::runtime_error("array is not C-contiguous");
    }
    
    switch(desc->type_num) {
        case NPY_UINT8:
            array2alphamodulated_uint8(array, img, r,g,b, normalizeLow, normalizeHigh);
            break;
        case NPY_FLOAT32:
            array2alphamodulated_float32(array, img, r,g,b, normalizeLow, normalizeHigh);
            break;
        default:
            std::stringstream err;
            err << "dtype is " << desc->type_num << ", was expecting float32";
            throw std::runtime_error(err.str());
    }
}
