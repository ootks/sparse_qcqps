cmake_minimum_required(VERSION 3.4...3.18)
project(lpm_methods)

find_package(Eigen3 REQUIRED)

file(GLOB lpm_methods_src
     "src/*.h"
     "src/*.cpp"
)

if(SKBUILD)
  # Scikit-Build does not add your site-packages to the search path
  # automatically, so we need to add it _or_ the pybind11 specific directory
  # here.
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c
            "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE _tmp_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
  list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")
endif()

add_compile_options(-Wall -Wextra -pedantic -Werror)

# Now we can find pybind11
find_package(pybind11 CONFIG REQUIRED)
include_directories(${Eigen_INCLUDE_DIRS})
pybind11_add_module(lpm_methods ${lpm_methods_src})
target_link_libraries(lpm_methods PRIVATE Eigen3::Eigen)

target_compile_definitions(lpm_methods PRIVATE VERSION_INFO=${PROJECT_VERSION})

install(TARGETS lpm_methods DESTINATION .)
