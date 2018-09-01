cmake_minimum_required(VERSION 3.5)

set (CMAKE_CXX_STANDARD 14)
project(markerdetector LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

find_package(Qt5 COMPONENTS Core Quick Multimedia REQUIRED)
find_package(OpenCV REQUIRED imgproc calib3d)
find_package(Boost REQUIRED)

include_directories(include)
include_directories(${Boost_INCLUDE_DIRS})

if(WIN32)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif(WIN32)

file(GLOB HEADERS_FILES "include/*.h*")
file(GLOB RESOURCES_FILES "resource/*.qrc")
file(GLOB SOURCE_FILES "src/*.cpp")
set(SOURCES ${HEADERS_FILES} ${SOURCE_FILES} ${RESOURCES_FILES})

add_executable(${PROJECT_NAME} "${SOURCES}")

target_link_libraries(${PROJECT_NAME}
    Qt5::Core
    Qt5::Quick
    Qt5::Multimedia
    ${OpenCV_LIBS}
    )
