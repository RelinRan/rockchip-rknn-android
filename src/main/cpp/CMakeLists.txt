cmake_minimum_required(VERSION 3.22.1)

project(rknn_jni)

add_library(
        rknn_jni
        SHARED
        src/rknn_jni.cpp
)

target_include_directories(
        rknn_jni
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

find_library(log-lib log)

target_link_libraries(
        rknn_jni
        ${log-lib}
        dl
)
