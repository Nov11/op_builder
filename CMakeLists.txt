cmake_minimum_required(VERSION 3.17)
project(custom_op VERSION 1.0.0 DESCRIPTION "custom tensorflow op")

set(CMAKE_CXX_STANDARD 14)
set(CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g -gdwarf-4")

#execute_process(COMMAND "${PROJECT_SOURCE_DIR}/.venv/bin/python" "-c" "import tensorflow as tf; import sys; sys.stdout.write(tf.sysconfig.get_lib() + '/')" OUTPUT_VARIABLE TF_LIB)
set(TF_LIB .venv/lib/python3.8/site-packages/tensorflow/)
message(STATUS "TF_LIB: ${TF_LIB}")

#execute_process(COMMAND "${PROJECT_SOURCE_DIR}/.venv/bin/python" "-c" "import tensorflow as tf; import sys; sys.stdout.write(tf.sysconfig.get_include() + '/')" OUTPUT_VARIABLE TF_INC)
set(TF_INC .venv/lib/python3.8/site-packages/tensorflow/include/)
message(STATUS "TF_INC: ${TF_INC}")

#execute_process(COMMAND "${PROJECT_SOURCE_DIR}/.venv/bin/python" "-c" "import tensorflow as tf; import sys; sys.stdout.write(tf.sysconfig.get_compile_flags() + '/')" OUTPUT_VARIABLE TF_COMPILE_FLAGS)
set(TF_COMPILE_FLAGS '.venv/lib/python3.8/site-packages/tensorflow/include', '-D_GLIBCXX_USE_CXX11_ABI=0')
message(STATUS "TF_COMPILE_FLAGS: ${TF_COMPILE_FLAGS}")
set(TF_LINK_FLAGS '.venv/lib/python3.8/site-packages/tensorflow -l:libtensorflow_framework.so.2')
message(STATUS "TF_LINK_FLAGS: ${TF_LINK_FLAGS}")


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")

include_directories(${TF_INC})
set(SOURCE
        src/kernel/zero_out_kernel.cpp
        src/op/zero_out_op.cpp)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Flags: ${CMAKE_CXX_FLAGS}")

add_library(custom_op SHARED ${SOURCE})

find_library(TF_FRAMEWORK_LIBRARY libtensorflow_framework.so.2 HINTS ${TF_LIB})
message(STATUS "TF_FRAMEWORK_LIBRARY : ${TF_FRAMEWORK_LIBRARY}")
target_link_libraries(custom_op ${TF_FRAMEWORK_LIBRARY})

set_target_properties(custom_op PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/out/library)