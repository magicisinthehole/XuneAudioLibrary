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

# ============================================================================
# libebur128 (EBU R128 loudness measurement)
# ============================================================================

set(EBUR128_ROOT "${CMAKE_SOURCE_DIR}/vendor/libebur128")
set(EBUR128_BUILD_DIR "${CMAKE_BINARY_DIR}/vendor/ebur128-build")

if(APPLE)
    set(EBUR128_LIBRARY "${EBUR128_BUILD_DIR}/libebur128.dylib")
elseif(WIN32)
    set(EBUR128_LIBRARY "${EBUR128_BUILD_DIR}/ebur128.dll")
else()
    set(EBUR128_LIBRARY "${EBUR128_BUILD_DIR}/libebur128.so")
endif()

if(NOT EXISTS "${EBUR128_LIBRARY}")
    message(STATUS "libebur128 not found, building from source...")

    file(MAKE_DIRECTORY ${EBUR128_BUILD_DIR})

    set(EBUR128_PLATFORM_FLAGS)
    if(APPLE)
        list(APPEND EBUR128_PLATFORM_FLAGS -DCMAKE_OSX_DEPLOYMENT_TARGET=10.15)
    endif()

    execute_process(
        COMMAND ${CMAKE_COMMAND}
            -S ${EBUR128_ROOT}
            -B ${EBUR128_BUILD_DIR}
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DCMAKE_POLICY_VERSION_MINIMUM=3.5
            -DBUILD_SHARED_LIBS=ON
            -DENABLE_TESTS=OFF
            ${EBUR128_PLATFORM_FLAGS}
            -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/install
        RESULT_VARIABLE EBUR128_CMAKE_RESULT
        OUTPUT_VARIABLE EBUR128_CMAKE_OUTPUT
        ERROR_VARIABLE EBUR128_CMAKE_ERROR
    )

    if(NOT EBUR128_CMAKE_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to configure libebur128:\n${EBUR128_CMAKE_ERROR}")
    endif()

    message(STATUS "Building libebur128...")
    execute_process(
        COMMAND ${CMAKE_COMMAND} --build ${EBUR128_BUILD_DIR} --config ${CMAKE_BUILD_TYPE}
        RESULT_VARIABLE EBUR128_BUILD_RESULT
        OUTPUT_VARIABLE EBUR128_BUILD_OUTPUT
        ERROR_VARIABLE EBUR128_BUILD_ERROR
    )

    if(NOT EBUR128_BUILD_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to build libebur128:\n${EBUR128_BUILD_ERROR}")
    endif()

    message(STATUS "✓ libebur128 built successfully")
else()
    message(STATUS "✓ Using existing libebur128")
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

    # Custom target to copy libebur128
    add_custom_target(copy_ebur128 ALL
        COMMAND ${CMAKE_COMMAND} -E make_directory "${RUNTIME_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy "${EBUR128_LIBRARY}" "${RUNTIME_DIR}/"
        COMMAND ${CMAKE_COMMAND} -E echo "Copied ${EBUR128_LIBRARY} -> ${RUNTIME_DIR}/"
        COMMENT "Copying libebur128 to Xune.Audio.Native runtime directory"
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
    set(BEAT_INFERENCE_SOURCE src/beattracking/beat_inference_mlx.cpp)
    set(SMARTDJ_INFERENCE_LIBS mlx)
else()
    # ONNX Runtime backend (CPU on Windows/Linux)
    set(ORT_VERSION "1.24.2")

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
    set(BEAT_INFERENCE_SOURCE src/beattracking/beat_inference_ort.cpp)
    set(SMARTDJ_INFERENCE_LIBS onnxruntime::onnxruntime)
endif()

# Platform FFT library for SmartDJ mel spectrogram
# macOS: Accelerate/vDSP (already found above)
# Linux/Windows: FFTW3F single-precision (eliminates float↔double conversion overhead)
if(NOT APPLE)
    find_library(FFTW3F_LIBRARY fftw3f REQUIRED)
    message(STATUS "Found FFTW3F: ${FFTW3F_LIBRARY}")
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

# Beat tracking sources (Beat This! mel spectrogram + inference + C API)
# Uses same inference backend conditional as SmartDJ
set(BEATTRACKING_SOURCES
    src/beattracking/beat_mel_spectrogram.cpp
    ${BEAT_INFERENCE_SOURCE}
    src/beattracking/beat_postprocess.cpp
    src/beattracking/beat_service.cpp
)

# Create xune_audio shared library
add_library(xune_audio SHARED
    src/xune_version.cpp
    ${NOWPLAYING_SOURCES}
    ${SMARTDJ_SOURCES}
    ${BEATTRACKING_SOURCES}
)

# Set library properties
set_target_properties(xune_audio PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

if(APPLE)
    set_target_properties(xune_audio PROPERTIES
        MACOSX_RPATH OFF
        INSTALL_NAME_DIR "@loader_path"
    )
endif()

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
    # FFTW3F single-precision on Linux/Windows
    target_link_libraries(xune_audio PRIVATE ${FFTW3F_LIBRARY})
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

    # macOS: bundle libmlx and fix install names for hardened runtime compatibility
    if(APPLE)
        get_target_property(MLX_LIB_PATH mlx IMPORTED_LOCATION)
        if(NOT MLX_LIB_PATH)
            get_target_property(MLX_LIB_PATH mlx IMPORTED_LOCATION_RELEASE)
        endif()
        if(NOT MLX_LIB_PATH)
            get_target_property(MLX_LIB_PATH mlx IMPORTED_LOCATION_NOCONFIG)
        endif()
        if(MLX_LIB_PATH)
            get_filename_component(MLX_LIB_DIR "${MLX_LIB_PATH}" DIRECTORY)
            add_custom_command(TARGET xune_audio POST_BUILD
                # Bundle libmlx.dylib and mlx.metallib (Metal shaders)
                COMMAND ${CMAKE_COMMAND} -E copy "${MLX_LIB_PATH}" "${RUNTIME_DIR}/libmlx.dylib"
                COMMAND ${CMAKE_COMMAND} -E copy "${MLX_LIB_DIR}/mlx.metallib" "${RUNTIME_DIR}/mlx.metallib"
                # Fix libmlx install name
                COMMAND install_name_tool -id "@loader_path/libmlx.dylib" "${RUNTIME_DIR}/libmlx.dylib"
                # Fix xune_audio's reference to libmlx
                COMMAND install_name_tool -change "@rpath/libmlx.dylib" "@loader_path/libmlx.dylib"
                    "$<TARGET_FILE:xune_audio>"
                COMMAND install_name_tool -change "@rpath/libmlx.dylib" "@loader_path/libmlx.dylib"
                    "${RUNTIME_DIR}/$<TARGET_FILE_NAME:xune_audio>"
                COMMAND install_name_tool -change "@rpath/libmlx.dylib" "@loader_path/libmlx.dylib"
                    "${RUNTIME_DIR}/libxune_audio.dylib"
                # Fix xune_audio's own install name in the copied files
                COMMAND install_name_tool -id "@loader_path/libxune_audio.1.dylib"
                    "${RUNTIME_DIR}/$<TARGET_FILE_NAME:xune_audio>"
                COMMAND install_name_tool -id "@loader_path/libxune_audio.1.dylib"
                    "${RUNTIME_DIR}/libxune_audio.dylib"
                # Re-sign ad-hoc (install_name_tool invalidates signatures)
                COMMAND codesign --force --sign - "${RUNTIME_DIR}/libmlx.dylib"
                COMMAND codesign --force --sign - "${RUNTIME_DIR}/$<TARGET_FILE_NAME:xune_audio>"
                COMMAND codesign --force --sign - "${RUNTIME_DIR}/libxune_audio.dylib"
                COMMENT "Bundling libmlx.dylib, mlx.metallib, and fixing install names"
            )
        else()
            message(WARNING "Could not determine MLX library path for bundling")
        endif()
    endif()

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
message(STATUS "  BeatTracking src:   ${BEATTRACKING_SOURCES}")
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
        target_link_libraries(smartdj_test PRIVATE ${FFTW3F_LIBRARY})
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

    # Beat tracking test
    set(BEAT_REFERENCE_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../src/Xune.App/Assets/Models/beat_reference_data")
    if(APPLE)
        set(BEAT_MODEL_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../src/Xune.App/Assets/Models/beat_this_small.safetensors")
    else()
        set(BEAT_MODEL_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../src/Xune.App/Assets/Models/beat_this_small.onnx")
    endif()

    add_executable(beat_test tests/beat_test.cpp ${BEATTRACKING_SOURCES})
    target_link_libraries(beat_test PRIVATE
        ${SMARTDJ_INFERENCE_LIBS}
        GTest::gtest_main
    )
    if(APPLE)
        target_link_libraries(beat_test PRIVATE ${ACCELERATE_LIBRARY})
    else()
        target_link_libraries(beat_test PRIVATE ${FFTW3F_LIBRARY})
    endif()
    target_include_directories(beat_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_compile_definitions(beat_test PRIVATE
        XUNE_AUDIO_EXPORTS
        BEAT_REFERENCE_DATA_DIR="${BEAT_REFERENCE_DATA_DIR}"
        BEAT_MODEL_PATH="${BEAT_MODEL_PATH}"
    )
    gtest_discover_tests(beat_test)

    # Tool: generate reference embeddings using the C++ pipeline (for INT8 testing)
    add_executable(generate_reference_embeddings tools/generate_reference_embeddings.cpp ${SMARTDJ_SOURCES})
    target_link_libraries(generate_reference_embeddings PRIVATE ${SMARTDJ_INFERENCE_LIBS})
    if(APPLE)
        target_link_libraries(generate_reference_embeddings PRIVATE ${ACCELERATE_LIBRARY})
    else()
        target_link_libraries(generate_reference_embeddings PRIVATE ${FFTW3F_LIBRARY})
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
message(STATUS "  libebur128:           ${EBUR128_LIBRARY}")
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
