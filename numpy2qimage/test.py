import time

from _numpy2qimage import Converters
from PyQt4.QtGui import QImage
import numpy
import qimage2ndarray


def test(d):
    normalize = [17,216]

    t1 = time.time()
    img1 = QImage(d.shape[1], d.shape[0], QImage.Format_ARGB32_Premultiplied)
    Converters.array2gray(d, img1, normalize[0], normalize[1])
    t1 = time.time()-t1

    t2 = time.time()
    a = numpy.clip(d, *normalize)
    img2 = qimage2ndarray.gray2qimage(a, normalize)
    img2 = img2.convertToFormat(QImage.Format_ARGB32_Premultiplied)
    t2 = time.time()-t2

    print img2.width(), img2.height()

    print "t1: %f msec (new)" % (1000.0*t1,)
    print "t2: %f msec" % (1000.0*t2,)

    print "speedup:", t2/t1

    img1.save("/tmp/img1.png")
    img2.save("/tmp/img2.png")

    if d.dtype == numpy.float32:
        Converters.array2alphamodulated(d, img2, 1.0, 0.0, 0.0, normalize[0], normalize[1])
        img2.save("/tmp/img3.png")                             
        Converters.array2alphamodulated(d, img2, 0.0, 1.0, 0.0, normalize[0], normalize[1]) 
        img2.save("/tmp/img4.png")                             
        Converters.array2alphamodulated(d, img2, 0.0, 0.0, 1.0, normalize[0], normalize[1]) 
        img2.save("/tmp/img5.png")
        Converters.array2alphamodulated(d, img2, 1.0, 1.0, 0.0, normalize[0], normalize[1]) 
        img2.save("/tmp/img5.png")
        Converters.array2alphamodulated(d, img2, 255.0/255.0, 132.0/255.0, 241/255.0, normalize[0], normalize[1]) 
        img2.save("/tmp/img6.png")
    
d = (255*numpy.random.random((255, 255))).astype(dtype=numpy.uint8)
test(d)

d = (255*numpy.random.random((255, 255))).astype(dtype=numpy.float32)
test(d)
