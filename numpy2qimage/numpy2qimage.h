#ifndef NUMPY2QIMAGE_H
#define NUMPY2QIMAGE_H

#include <Python.h>
#include <numpy/arrayobject.h>

#include <QtGui/QImage>

class Converters {
    public:
    static void array2gray(PyObject* a, QImage& img, int normalizeLow, int normalizeHigh);
    static void array2alphamodulated(PyObject* a, QImage& img, float r, float g, float b, int normalizeLow, int normalizeHigh);
};

#endif /*NUMPY2QIMAGE_H*/
