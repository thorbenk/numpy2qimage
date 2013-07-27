%Module(name=_numpy2qimage, version=0)

%ModuleHeaderCode
extern "C" { void numpy2qimage_import_array(); }
%End

%PostInitialisationCode
numpy2qimage_import_array();
%End

%Import QtCore/QtCoremod.sip
%Import QtGui/QtGuimod.sip

class Converters {
    public:

    %TypeHeaderCode
    #include "numpy2qimage.h"
    %End

    static void array2gray(PyObject* a, QImage& img, int normalizeLow, int normalizeHigh);
    static void array2alphamodulated(PyObject* a, QImage& img, float r, float g, float b, int normalizeLow, int normalizeHigh);

};
