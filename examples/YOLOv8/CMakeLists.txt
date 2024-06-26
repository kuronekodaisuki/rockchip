cmake_minimum_required(VERSION 3.4.1)

project(YOLOv8)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# skip 3rd-party lib dependencies
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined")

# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/YOLOv8_${CMAKE_SYSTEM_NAME})

set(CMAKE_SKIP_INSTALL_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

set(LIB_ARCH aarch64)

include_directories(${CMAKE_SOURCE_DIR})

# rknn api
set(RKNN_API_PATH ${CMAKE_SOURCE_DIR}/../../runtime/RK3588/${CMAKE_SYSTEM_NAME}/librknn_api)

set(RKNN_RT_LIB ${RKNN_API_PATH}/${LIB_ARCH}/librknnrt.so)

include_directories(${RKNN_API_PATH}/include)
include_directories(${CMAKE_SOURCE_DIR}/../3rdparty)

# opencv
find_package(OpenCV REQUIRED)

# rga
# rga
set(RGA_PATH ${CMAKE_SOURCE_DIR}/../3rdparty/rga)
set(RGA_LIB ${RGA_PATH}/libs/Linux//${LIB_ARCH}/librga.so)

include_directories(${RGA_PATH}/include)

# mpp
set(MPP_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/mpp)
set(MPP_LIBS ${MPP_PATH}/${CMAKE_SYSTEM_NAME}/${LIB_ARCH}/librockchip_mpp.so)

include_directories(${MPP_PATH}/include)

# zlmediakit
set(ZLMEDIAKIT_PATH ${CMAKE_SOURCE_DIR}/../3rdparty/zlmediakit)

#if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  include_directories(${ZLMEDIAKIT_PATH}/include)
  set(ZLMEDIAKIT_LIBS ${ZLMEDIAKIT_PATH}/${LIB_ARCH}/libmk_api.so)
#endif()

if(ZLMEDIAKIT_LIBS)
  add_definitions(-DBUILD_VIDEO_RTSP)
endif()

set(CMAKE_INSTALL_RPATH "lib")

# YOLOv8
include_directories(
  ${CMAKE_SOURCE_DIR}/include
  $(OpenCV_DIR)
)

add_executable(YOLOv8
  main.cpp
  Inference.cpp
  YOLOv8.cpp
)

target_link_libraries(YOLOv8
  ${RKNN_RT_LIB}
  ${RGA_LIB}
  ${OpenCV_LIBS}
)

# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/YOLOv8_${CMAKE_SYSTEM_NAME})
install(TARGETS YOLOv8 DESTINATION ./)

install(PROGRAMS ${RKNN_RT_LIB} DESTINATION lib)
install(PROGRAMS ${RGA_LIB} DESTINATION lib)
install(DIRECTORY model DESTINATION ./)

if(MPP_LIBS)
  install(TARGETS YOLOv8 DESTINATION ./)
  install(PROGRAMS ${MPP_LIBS} DESTINATION lib)
endif()

if(ZLMEDIAKIT_LIBS)
  install(PROGRAMS ${ZLMEDIAKIT_LIBS} DESTINATION lib)
endif()