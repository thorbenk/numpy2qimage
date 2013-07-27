project(numpy2qimage)

cmake_minimum_required(VERSION 2.8)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)

find_package(Boost COMPONENTS system timer)
find_package(Qt4 REQUIRED)
find_package(Vigra REQUIRED)

add_subdirectory(numpy2qimage)
