cmake_minimum_required(VERSION 3.18)

project(rgbd_slam LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OpenCV_DIR
    "/home/mohit/miniconda3/envs/opencv_cuda/lib/cmake/opencv4"
)

# OpenCV was built against CUDA 12.6 (the system default /usr is CUDA 10.1).
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.6" CACHE PATH "CUDA used by OpenCV")
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED COMPONENTS
    common
    io
    filters
    visualization
)
find_package(Ceres REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)

set(CONDA_LIB "/home/mohit/miniconda3/envs/opencv_cuda/lib")
set(APP_RPATH "${CONDA_LIB};/usr/local/cuda-12.6/lib64")

# ------------------------------------------------------------
# vio_slam: RGB-D + IMU SLAM -> trajectory, 3D scene, verification
# ------------------------------------------------------------
add_executable(vio_slam vio_slam.cpp)
target_include_directories(vio_slam PRIVATE
    ${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ${PCL_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS})
target_compile_definitions(vio_slam PRIVATE ${PCL_DEFINITIONS})
target_link_directories(vio_slam PRIVATE ${CONDA_LIB})
target_link_libraries(vio_slam PRIVATE
    ${OpenCV_LIBS} Eigen3::Eigen ${PCL_LIBRARIES} ${CERES_LIBRARIES}
    OpenMP::OpenMP_CXX Threads::Threads)
target_compile_options(vio_slam PRIVATE -O3 -march=native)
set_target_properties(vio_slam PROPERTIES BUILD_RPATH "${APP_RPATH}" INSTALL_RPATH "${APP_RPATH}")

# ------------------------------------------------------------
# scene_player: 30 FPS playback of RGB + growing 3D scene
# ------------------------------------------------------------
add_executable(scene_player scene_player.cpp)
target_include_directories(scene_player PRIVATE
    ${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ${PCL_INCLUDE_DIRS})
target_compile_definitions(scene_player PRIVATE ${PCL_DEFINITIONS})
target_link_directories(scene_player PRIVATE ${CONDA_LIB})
target_link_libraries(scene_player PRIVATE
    ${OpenCV_LIBS} Eigen3::Eigen ${PCL_LIBRARIES} Threads::Threads)
target_compile_options(scene_player PRIVATE -O3 -march=native)
set_target_properties(scene_player PROPERTIES BUILD_RPATH "${APP_RPATH}" INSTALL_RPATH "${APP_RPATH}")

# ------------------------------------------------------------
# map_builder: TSDF 3D reconstruction from the vio_slam results
# ------------------------------------------------------------
add_executable(map_builder map_builder.cpp)
target_include_directories(map_builder PRIVATE
    ${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ${PCL_INCLUDE_DIRS})
target_compile_definitions(map_builder PRIVATE ${PCL_DEFINITIONS})
target_link_directories(map_builder PRIVATE ${CONDA_LIB})
target_link_libraries(map_builder PRIVATE
    ${OpenCV_LIBS} Eigen3::Eigen ${PCL_LIBRARIES} OpenMP::OpenMP_CXX Threads::Threads)
target_compile_options(map_builder PRIVATE -O3 -march=native)
set_target_properties(map_builder PROPERTIES BUILD_RPATH "${APP_RPATH}" INSTALL_RPATH "${APP_RPATH}")
