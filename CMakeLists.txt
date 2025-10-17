cmake_minimum_required(VERSION 3.12)
project(ParallelTasks VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP")
else()
    message(FATAL_ERROR "OpenMP not found")
endif()

find_package(MPI REQUIRED)
if(MPI_CXX_FOUND)
    message(STATUS "Found MPI")
else()
    message(FATAL_ERROR "MPI not found")
endif()

file(GLOB taskSources "src/*.cpp")
foreach(taskSource ${taskSources})
    get_filename_component(taskName ${taskSource} NAME_WE)
    add_executable(${taskName} ${taskSource})
    target_compile_options(${taskName} PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/O2> $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-O3>)
    target_link_libraries(${taskName} PRIVATE OpenMP::OpenMP_CXX MPI::MPI_CXX)
	
    set_target_properties(${taskName} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin"
        RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin"
    )
endforeach()
