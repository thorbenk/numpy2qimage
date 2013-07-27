INCLUDE_DIRECTORIES(
    ${PYTHON_INCLUDE_PATH}
    ${SIP_INCLUDE_DIR}
    ${QT_INCLUDE_DIR}
    ${QT_QT_INCLUDE_DIR}
    ${QT_QTCORE_INCLUDE_DIR}
    ${QT_QTGUI_INCLUDE_DIR}
    ${QT_QTOPENGL_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

add_library(numpy2qimage SHARED numpy2qimage.cxx)
target_link_libraries(numpy2qimage 
    ${QT_QTCORE_LIBARARY}
    ${QT_QTGUI_LIBRARY}
)

#for the following, inspired by:
#
# http://quickgit.kde.org/?p=pykde4.git&a=blob
# &h=c853e3855e976925aa56897a162830b219e8bde4
# &hb=fddbc67c8958e4661e88ea57704a27823ddfde3a
# &f=CMakeLists.txt

include(SIPMacros)
find_package(SIP REQUIRED)
find_package(PyQt4 REQUIRED)
find_package(PythonLibrary REQUIRED)

include_directories(${VIGRA_INCLUDE_DIR})

# Set a a PYTHON_3 flag.
IF (${PYTHON_SHORT_VERSION} MATCHES "^3")
    SET(PYTHON_3 TRUE)
ELSE (${PYTHON_SHORT_VERSION} MATCHES "^3")
    SET(PYTHON_3 FALSE)
ENDIF (${PYTHON_SHORT_VERSION} MATCHES "^3")

SET(SIP_INCLUDES ${PYQT4_SIP_DIR} sip)
SET(SIP_CONCAT_PARTS 1)
IF (WIN32)
    SET(SIP_TAGS ALL WS_WIN ${PYQT4_VERSION_TAG})
ELSE (WIN32)
    SET(SIP_TAGS ALL WS_X11 ${PYQT4_VERSION_TAG})
ENDIF (WIN32)
SET(SIP_DISABLE_FEATURES VendorID PyQt_NoPrintRangeBug)

# Use an extra option when compiling on Python 3.
IF (PYTHON_3)
    SET(SIP_EXTRA_OPTIONS -g -x Py_v3)
ELSE (PYTHON_3)
    SET(SIP_EXTRA_OPTIONS -g)
ENDIF (PYTHON_3)

set(SIP_INCLUDES
    ${PYQT4_SIP_DIR}
    ${PYQT4_SIP_DIR}/QtCore
    ${PYQT4_SIP_DIR}/QtGui
)
set(SIP_CONCAT_PARTS 1)

ADD_DEFINITIONS(-DQT_CORE_LIB -DQT_GUI_LIB)

ADD_SIP_PYTHON_MODULE(_numpy2qimage numpy2qimage.sip numpy2qimage ${QT_QTCORE_LIBARARY} ${QT_QTGUI_LIBRARY})
get_property(location TARGET python_module__numpy2qimage PROPERTY LOCATION)
add_custom_command(TARGET python_module__numpy2qimage
    POST_BUILD        
    COMMAND ${CMAKE_COMMAND} -E copy_if_different  
        ${location}
        ${CMAKE_CURRENT_SOURCE_DIR}
)
