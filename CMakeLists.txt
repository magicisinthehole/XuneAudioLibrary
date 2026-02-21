cmake_minimum_required(VERSION 3.14)  # 3.14 required for FetchContent
project(xune_audio_library VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "")
message(STATUS "========================================")
message(STATUS "XuneAudioLibrary Configuration")
message(STATUS "========================================")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "System: ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}")

# Chromaprint library configuration
set(CHROMAPRINT_ROOT "${CMAKE_SOURCE_DIR}/vendor/chromaprint")
set(CHROMAPRINT_BUILD_DIR "${CMAKE_BINARY_DIR}/vendor/chromaprint-build")

# Determine library name based on platform
if(APPLE)
    set(CHROMAPRINT_LIBRARY "${CHROMAPRINT_BUILD_DIR}/src/libchromaprint.dylib")
elseif(WIN32)
    set(CHROMAPRINT_LIBRARY "${CHROMAPRINT_BUILD_DIR}/src/chromaprint.dll")
else()
    set(CHROMAPRINT_LIBRARY "${CHROMAPRINT_BUILD_DIR}/src/libchromaprint.so")
endif()

# Check if Chromaprint is already built
if(NOT EXISTS "${CHROMAPRINT_LIBRARY}")
    message(STATUS "Chromaprint library not found, building from source...")

    # Create build directory
    file(MAKE_DIRECTORY ${CHROMAPRINT_BUILD_DIR})

    # Configure Chromaprint
    message(STATUS "Configuring Chromaprint...")
    # Select FFT backend and platform-specific flags
    if(APPLE)
        set(CHROMAPRINT_FFT_LIB "vdsp")       # macOS Accelerate framework (fastest, zero deps)
        set(CHROMAPRINT_PLATFORM_FLAGS -DCMAKE_OSX_DEPLOYMENT_TARGET=10.15)
    elseif(UNIX)
        set(CHROMAPRINT_FFT_LIB "fftw3")      # FFTW3 on Linux (install with: sudo apt install libfftw3-dev)
        set(CHROMAPRINT_PLATFORM_FLAGS)
    endif()

    execute_process(
        COMMAND ${CMAKE_COMMAND}
            -S ${CHROMAPRINT_ROOT}
            -B ${CHROMAPRINT_BUILD_DIR}
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DBUILD_SHARED_LIBS=ON
            -DBUILD_TOOLS=OFF
            -DBUILD_TESTS=OFF
            -DFFT_LIB=${CHROMAPRINT_FFT_LIB}
            ${CHROMAPRINT_PLATFORM_FLAGS}
            -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/install
        RESULT_VARIABLE CHROMAPRINT_CMAKE_RESULT
        OUTPUT_VARIABLE CHROMAPRINT_CMAKE_OUTPUT
        ERROR_VARIABLE CHROMAPRINT_CMAKE_ERROR
    )

    if(NOT CHROMAPRINT_CMAKE_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to configure Chromaprint:\n${CHROMAPRINT_CMAKE_ERROR}")
    endif()

    # Build Chromaprint
    message(STATUS "Building Chromaprint...")
    execute_process(
        COMMAND ${CMAKE_COMMAND} --build ${CHROMAPRINT_BUILD_DIR} --config ${CMAKE_BUILD_TYPE}
        RESULT_VARIABLE CHROMAPRINT_BUILD_RESULT
        OUTPUT_VARIABLE CHROMAPRINT_BUILD_OUTPUT
        ERROR_VARIABLE CHROMAPRINT_BUILD_ERROR
    )

    if(NOT CHROMAPRINT_BUILD_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to build Chromaprint:\n${CHROMAPRINT_BUILD_ERROR}")
    endif()

    message(STATUS "✓ Chromaprint library built successfully")
else()
    message(STATUS "✓ Using existing Chromaprint library")
endif()

# macOS specific settings
if(APPLE)
    find_library(ACCELERATE_LIBRARY Accelerate REQUIRED)
    message(STATUS "Found Accelerate framework: ${ACCELERATE_LIBRARY}")
    set(PLATFORM_LIBRARIES ${ACCELERATE_LIBRARY})
else()
    set(PLATFORM_LIBRARIES)
endif()

# Auto-copy to Xune.Audio.Native runtime directories
set(AUDIO_NATIVE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../src/Xune.Audio.Native")
if(EXISTS "${AUDIO_NATIVE_DIR}")
    message(STATUS "Xune.Audio.Native project found")

    # Determine runtime identifier based on platform and architecture
    if(APPLE)
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
            set(RUNTIME_ID "osx-arm64")
        else()
            set(RUNTIME_ID "osx-x64")
        endif()
    elseif(WIN32)
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "ARM64")
            set(RUNTIME_ID "win-arm64")
        else()
            set(RUNTIME_ID "win-x64")
        endif()
    else()
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
            set(RUNTIME_ID "linux-arm64")
        else()
            set(RUNTIME_ID "linux-x64")
        endif()
    endif()

    set(RUNTIME_DIR "${AUDIO_NATIVE_DIR}/runtimes/${RUNTIME_ID}/native")
    message(STATUS "Target runtime directory: ${RUNTIME_DIR}")

    # Custom target to copy Chromaprint library
    add_custom_target(copy_chromaprint ALL
        COMMAND ${CMAKE_COMMAND} -E make_directory "${RUNTIME_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy "${CHROMAPRINT_LIBRARY}" "${RUNTIME_DIR}/"
        COMMAND ${CMAKE_COMMAND} -E echo "Copied ${CHROMAPRINT_LIBRARY} -> ${RUNTIME_DIR}/"
        COMMENT "Copying Chromaprint library to Xune.Audio.Native runtime directory"
        VERBATIM
    )

    message(STATUS "Auto-copy enabled: library will be copied to .NET runtime folder")
else()
    message(WARNING "Xune.Audio.Native not found - library will not be auto-copied")
endif()

# ============================================================================
# SmartDJ Inference Backend (MLX on macOS, ONNX Runtime on Windows/Linux)
# ============================================================================

if(APPLE)
    # MLX backend (Metal GPU on Apple Silicon)
    message(STATUS "")
    message(STATUS "Configuring MLX backend (Metal GPU)...")

    find_package(Python 3.9 REQUIRED COMPONENTS Interpreter)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m mlx --cmake-dir
        OUTPUT_VARIABLE MLX_ROOT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE MLX_CMAKE_RESULT
    )
    if(NOT MLX_CMAKE_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to find MLX cmake dir. Install with: pip3 install mlx")
    endif()

    find_package(MLX CONFIG REQUIRED HINTS "${MLX_ROOT}")
    message(STATUS "✓ MLX: ${MLX_ROOT}")

    set(SMARTDJ_INFERENCE_SOURCE src/smartdj/model_inference_mlx.cpp)
    set(SMARTDJ_INFERENCE_LIBS mlx)
else()
    # ONNX Runtime backend (CPU on Windows/Linux)
    set(ORT_VERSION "1.22.0")

    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
            set(ORT_PLATFORM "linux-aarch64")
        else()
            set(ORT_PLATFORM "linux-x64")
        endif()
        set(ORT_LIB_NAME "libonnxruntime.so")
        set(ORT_LIB_PATTERN "libonnxruntime.so*")
    elseif(WIN32)
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "ARM64")
            set(ORT_PLATFORM "win-arm64")
        else()
            set(ORT_PLATFORM "win-x64")
        endif()
        set(ORT_LIB_NAME "onnxruntime.dll")
        set(ORT_LIB_PATTERN "onnxruntime*.dll")
    else()
        message(FATAL_ERROR "Unsupported platform for ONNX Runtime: ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}")
    endif()

    message(STATUS "")
    message(STATUS "Fetching ONNX Runtime ${ORT_VERSION} (${ORT_PLATFORM})...")

    FetchContent_Declare(
        onnxruntime
        URL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-${ORT_PLATFORM}-${ORT_VERSION}.tgz"
    )
    FetchContent_MakeAvailable(onnxruntime)

    add_library(onnxruntime::onnxruntime SHARED IMPORTED GLOBAL)
    set_target_properties(onnxruntime::onnxruntime PROPERTIES
        IMPORTED_LOCATION "${onnxruntime_SOURCE_DIR}/lib/${ORT_LIB_NAME}"
        INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_SOURCE_DIR}/include"
    )
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        set_target_properties(onnxruntime::onnxruntime PROPERTIES
            IMPORTED_SONAME "${ORT_LIB_NAME}"
        )
    endif()

    message(STATUS "✓ ONNX Runtime: ${onnxruntime_SOURCE_DIR}")

    set(SMARTDJ_INFERENCE_SOURCE src/smartdj/model_inference_ort.cpp)
    set(SMARTDJ_INFERENCE_LIBS onnxruntime::onnxruntime)
endif()

# Platform FFT library for SmartDJ mel spectrogram
# macOS: Accelerate/vDSP (already found above)
# Linux/Windows: FFTW3 double-precision (already a Chromaprint dependency)
if(NOT APPLE)
    find_library(FFTW3_LIBRARY fftw3 REQUIRED)
    message(STATUS "Found FFTW3: ${FFTW3_LIBRARY}")
endif()

# ============================================================================
# XuneAudio Library (Now Playing + SmartDJ)
# ============================================================================

message(STATUS "")
message(STATUS "Configuring XuneAudio library...")

# Platform-specific Now Playing sources
if(APPLE)
    set(NOWPLAYING_SOURCES src/nowplaying/nowplaying_macos.mm)
    find_library(MEDIAPLAYER_FRAMEWORK MediaPlayer REQUIRED)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    find_library(APPKIT_FRAMEWORK AppKit REQUIRED)
    set(NOWPLAYING_LIBS ${MEDIAPLAYER_FRAMEWORK} ${FOUNDATION_FRAMEWORK} ${APPKIT_FRAMEWORK})
    message(STATUS "  Platform: macOS (MediaPlayer.framework)")
elseif(WIN32)
    # Windows: stub for now, full SMTC implementation later
    set(NOWPLAYING_SOURCES src/nowplaying/nowplaying_stub.cpp)
    set(NOWPLAYING_LIBS)
    message(STATUS "  Platform: Windows (stub)")
elseif(UNIX)
    # Linux: stub for now, D-Bus/MPRIS implementation later
    set(NOWPLAYING_SOURCES src/nowplaying/nowplaying_stub.cpp)
    set(NOWPLAYING_LIBS)
    message(STATUS "  Platform: Linux (stub)")
else()
    set(NOWPLAYING_SOURCES src/nowplaying/nowplaying_stub.cpp)
    set(NOWPLAYING_LIBS)
    message(STATUS "  Platform: Unknown (stub)")
endif()

# SmartDJ sources (mel spectrogram + model inference + C API)
# Inference backend: MLX on macOS, ORT on Windows/Linux (set above)
set(SMARTDJ_SOURCES
    src/smartdj/mel_spectrogram.cpp
    ${SMARTDJ_INFERENCE_SOURCE}
    src/smartdj/embedding_service.cpp
)

# Create xune_audio shared library
add_library(xune_audio SHARED ${NOWPLAYING_SOURCES} ${SMARTDJ_SOURCES})

# Set library properties
set_target_properties(xune_audio PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

# Enable exports macro
target_compile_definitions(xune_audio PRIVATE XUNE_AUDIO_EXPORTS)

# Include directories
target_include_directories(xune_audio
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src  # For smartdj/ internal headers
)

# Link platform libraries (Now Playing)
if(NOWPLAYING_LIBS)
    target_link_libraries(xune_audio PRIVATE ${NOWPLAYING_LIBS})
endif()

# Link inference backend (MLX on macOS, ORT on Windows/Linux)
target_link_libraries(xune_audio PRIVATE ${SMARTDJ_INFERENCE_LIBS})

# Link platform FFT (SmartDJ mel spectrogram)
if(APPLE)
    # Accelerate is already linked via PLATFORM_LIBRARIES / NOWPLAYING_LIBS on macOS
    target_link_libraries(xune_audio PRIVATE ${ACCELERATE_LIBRARY})
else()
    # FFTW3 double-precision on Linux/Windows (always available via Chromaprint dependency)
    target_link_libraries(xune_audio PRIVATE ${FFTW3_LIBRARY})
endif()

# macOS-specific settings
if(APPLE)
    # Enable Objective-C++ for .mm files
    set_source_files_properties(${NOWPLAYING_SOURCES} PROPERTIES
        COMPILE_FLAGS "-fobjc-arc"
    )
    # Set minimum deployment target
    set_target_properties(xune_audio PROPERTIES
        OSX_DEPLOYMENT_TARGET "10.15"
    )
endif()

# Post-build: copy to .NET runtime directory
if(EXISTS "${AUDIO_NATIVE_DIR}")
    add_custom_command(TARGET xune_audio POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${RUNTIME_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:xune_audio>" "${RUNTIME_DIR}/"
        COMMAND ${CMAKE_COMMAND} -E create_symlink "$<TARGET_FILE_NAME:xune_audio>" "${RUNTIME_DIR}/libxune_audio$<IF:$<PLATFORM_ID:Darwin>,.dylib,.so>"
        COMMENT "Copying xune_audio library to ${RUNTIME_DIR}/"
    )

    # Copy ONNX Runtime shared library to .NET runtime directory (non-macOS only)
    if(NOT APPLE)
        add_custom_command(TARGET xune_audio POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy "${onnxruntime_SOURCE_DIR}/lib/${ORT_LIB_NAME}" "${RUNTIME_DIR}/"
            COMMENT "Copying ONNX Runtime library to ${RUNTIME_DIR}/"
        )
        # On Linux, also copy the versioned .so (e.g. libonnxruntime.so.1.22.0)
        if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
            file(GLOB ORT_VERSIONED_LIBS "${onnxruntime_SOURCE_DIR}/lib/libonnxruntime.so.*")
            foreach(ORT_LIB ${ORT_VERSIONED_LIBS})
                add_custom_command(TARGET xune_audio POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy "${ORT_LIB}" "${RUNTIME_DIR}/"
                )
            endforeach()
        endif()
    endif()

    message(STATUS "  Auto-copy: enabled -> ${RUNTIME_DIR}/")
endif()

message(STATUS "  Now Playing sources: ${NOWPLAYING_SOURCES}")
message(STATUS "  SmartDJ sources:    ${SMARTDJ_SOURCES}")
message(STATUS "  Now Playing libs:   ${NOWPLAYING_LIBS}")

# ============================================================================
# Tests (GoogleTest)
# ============================================================================

option(BUILD_TESTS "Build SmartDJ tests" ON)

if(BUILD_TESTS)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/releases/download/v1.15.2/googletest-1.15.2.tar.gz
    )
    # Prevent GoogleTest from overriding our compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    enable_testing()

    # Reference data paths (exported by SmartDJ/onnx_export/export_reference_data.py)
    set(REFERENCE_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../src/Xune.App/Assets/Models/reference_data")
    if(APPLE)
        set(MODEL_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../src/Xune.App/Assets/Models/myna_hybrid.safetensors")
    else()
        set(MODEL_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../src/Xune.App/Assets/Models/myna_hybrid.onnx")
    endif()

    # Compile SmartDJ sources directly into the test (internal classes have hidden
    # visibility in the shared library, so we can't link against xune_audio for them)
    add_executable(smartdj_test tests/smartdj_test.cpp ${SMARTDJ_SOURCES})
    target_link_libraries(smartdj_test PRIVATE
        ${SMARTDJ_INFERENCE_LIBS}
        GTest::gtest_main
    )
    if(APPLE)
        target_link_libraries(smartdj_test PRIVATE ${ACCELERATE_LIBRARY})
    else()
        target_link_libraries(smartdj_test PRIVATE ${FFTW3_LIBRARY})
    endif()
    target_include_directories(smartdj_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_compile_definitions(smartdj_test PRIVATE
        XUNE_AUDIO_EXPORTS
        REFERENCE_DATA_DIR="${REFERENCE_DATA_DIR}"
        MODEL_PATH="${MODEL_PATH}"
    )

    include(GoogleTest)
    gtest_discover_tests(smartdj_test)

    # Tool: generate reference embeddings using the C++ pipeline (for INT8 testing)
    add_executable(generate_reference_embeddings tools/generate_reference_embeddings.cpp ${SMARTDJ_SOURCES})
    target_link_libraries(generate_reference_embeddings PRIVATE ${SMARTDJ_INFERENCE_LIBS})
    if(APPLE)
        target_link_libraries(generate_reference_embeddings PRIVATE ${ACCELERATE_LIBRARY})
    else()
        target_link_libraries(generate_reference_embeddings PRIVATE ${FFTW3_LIBRARY})
    endif()
    target_include_directories(generate_reference_embeddings PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_compile_definitions(generate_reference_embeddings PRIVATE XUNE_AUDIO_EXPORTS)

    message(STATUS "  Tests:              enabled (reference data: ${REFERENCE_DATA_DIR})")
endif()

# ============================================================================
# Configuration Summary
# ============================================================================

# Print summary
message(STATUS "")
message(STATUS "Configuration Summary:")
message(STATUS "----------------------------------------")
message(STATUS "  Project:              ${PROJECT_NAME} v${PROJECT_VERSION}")
message(STATUS "  Build type:           ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard:         ${CMAKE_CXX_STANDARD}")
message(STATUS "  Chromaprint library:  ${CHROMAPRINT_LIBRARY}")
if(APPLE)
message(STATUS "  Inference backend:    MLX (Metal GPU)")
else()
message(STATUS "  Inference backend:    ONNX Runtime ${ORT_VERSION} (${ORT_PLATFORM})")
endif()
message(STATUS "  XuneAudio library:    $<TARGET_FILE:xune_audio>")
message(STATUS "  Runtime ID:           ${RUNTIME_ID}")
message(STATUS "  Platform libraries:   ${PLATFORM_LIBRARIES}")
message(STATUS "========================================")
message(STATUS "")
