cmake_minimum_required(VERSION 3.14)  # 3.14 required for FetchContent
project(xune_audio_library VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

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
        if(CMAKE_OSX_ARCHITECTURES)
            list(APPEND CHROMAPRINT_PLATFORM_FLAGS -DCMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES})
        endif()
    elseif(WIN32)
        set(CHROMAPRINT_FFT_LIB "fftw3")      # FFTW3 on Windows (install with: vcpkg install fftw3:x64-windows)
        set(CHROMAPRINT_PLATFORM_FLAGS)
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

    message(STATUS "[OK] Chromaprint library built successfully")
else()
    message(STATUS "[OK] Using existing Chromaprint library")
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
        if(CMAKE_OSX_ARCHITECTURES)
            list(APPEND EBUR128_PLATFORM_FLAGS -DCMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES})
        endif()
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

    message(STATUS "[OK] libebur128 built successfully")
else()
    message(STATUS "[OK] Using existing libebur128")
endif()

# ============================================================================
# TagLib (audio metadata reading/writing)
# ============================================================================

message(STATUS "")
message(STATUS "Fetching TagLib...")

set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_BINDINGS OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS_SAVED ${BUILD_SHARED_LIBS})
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    taglib
    GIT_REPOSITORY https://github.com/taglib/taglib.git
    GIT_TAG v2.0.2
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(taglib)

set(BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS_SAVED} CACHE BOOL "" FORCE)
message(STATUS "[OK] TagLib: ${taglib_SOURCE_DIR}")

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
        if(CMAKE_OSX_ARCHITECTURES)
            set(_target_arch "${CMAKE_OSX_ARCHITECTURES}")
        else()
            set(_target_arch "${CMAKE_SYSTEM_PROCESSOR}")
        endif()
        if(_target_arch MATCHES "arm64|aarch64")
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

option(FORCE_ONNX "Force ONNX Runtime backend even on macOS (for testing)" OFF)

# MLX requires Apple Silicon (Metal GPU) — force ONNX for x86_64 cross-compilation
if(APPLE AND CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64")
    set(FORCE_ONNX ON CACHE BOOL "Forced ON: MLX requires Apple Silicon" FORCE)
endif()

if(APPLE AND NOT FORCE_ONNX)
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
    message(STATUS "[OK] MLX: ${MLX_ROOT}")

    set(SMARTDJ_INFERENCE_SOURCE src/smartdj/model_inference_mlx.cpp)
    set(BEAT_INFERENCE_SOURCE src/beattracking/beat_inference_mlx.cpp)
    set(SMARTDJ_INFERENCE_LIBS mlx)
    set(XUNE_USE_MLX TRUE)
else()
    # ONNX Runtime backend (GPU-accelerated on Windows/Linux)
    set(ORT_VERSION "1.24.2")

    # ORT_LOCAL_DIR: use a locally built ORT (e.g. aarch64 with XNNPACK from Docker)
    if(DEFINED ORT_LOCAL_DIR AND EXISTS "${ORT_LOCAL_DIR}/lib")
        message(STATUS "")
        message(STATUS "Using local ONNX Runtime from ${ORT_LOCAL_DIR}")

        set(ORT_LIB_NAME "libonnxruntime.so")
        add_library(onnxruntime::onnxruntime SHARED IMPORTED GLOBAL)
        set_target_properties(onnxruntime::onnxruntime PROPERTIES
            IMPORTED_LOCATION "${ORT_LOCAL_DIR}/lib/${ORT_LIB_NAME}"
            INTERFACE_INCLUDE_DIRECTORIES "${ORT_LOCAL_DIR}/include"
            IMPORTED_SONAME "${ORT_LIB_NAME}"
        )
        # Store source dir for post-build copy logic
        set(onnxruntime_SOURCE_DIR "${ORT_LOCAL_DIR}")
        message(STATUS "[OK] ONNX Runtime (local): ${ORT_LOCAL_DIR}")
    else()
        if(APPLE)
            if(_target_arch STREQUAL "x86_64")
                set(ORT_PLATFORM "osx-x86_64")
                # ORT dropped x86_64 macOS binaries in v1.24.0
                set(ORT_VERSION "1.23.1")
            else()
                set(ORT_PLATFORM "osx-arm64")
            endif()
            set(ORT_LIB_NAME "libonnxruntime.dylib")
        elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
            if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
                set(ORT_PLATFORM "linux-aarch64")
            else()
                # GPU tarball includes CUDA EP plugin .so
                set(ORT_PLATFORM "linux-x64-gpu")
            endif()
            set(ORT_LIB_NAME "libonnxruntime.so")
        elseif(WIN32)
            # DirectML NuGet includes DML statically linked into onnxruntime.dll
            if(CMAKE_SYSTEM_PROCESSOR MATCHES "ARM64")
                set(ORT_WIN_RID "win-arm64")
            else()
                set(ORT_WIN_RID "win-x64")
            endif()
            set(ORT_LIB_NAME "onnxruntime.dll")
        else()
            message(FATAL_ERROR "Unsupported platform for ONNX Runtime: ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}")
        endif()

        message(STATUS "")

        if(WIN32)
            # Fetch DirectML NuGet package (contains DML-enabled onnxruntime.dll for both x64 and arm64)
            message(STATUS "Fetching ONNX Runtime DirectML ${ORT_VERSION}...")
            FetchContent_Declare(
                onnxruntime
                URL "https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.DirectML/${ORT_VERSION}"
                DOWNLOAD_NAME "onnxruntime-dml-${ORT_VERSION}.zip"
            )
            FetchContent_MakeAvailable(onnxruntime)

            # NuGet layout: runtimes/{rid}/native/onnxruntime.dll, build/native/include/
            set(ORT_NATIVE_DIR "${onnxruntime_SOURCE_DIR}/runtimes/${ORT_WIN_RID}/native")
            add_library(onnxruntime::onnxruntime SHARED IMPORTED GLOBAL)
            set_target_properties(onnxruntime::onnxruntime PROPERTIES
                IMPORTED_LOCATION "${ORT_NATIVE_DIR}/${ORT_LIB_NAME}"
                IMPORTED_IMPLIB "${ORT_NATIVE_DIR}/onnxruntime.lib"
                INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_SOURCE_DIR}/build/native/include"
            )
            message(STATUS "[OK] ONNX Runtime DirectML: ${onnxruntime_SOURCE_DIR}")
        else()
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
            message(STATUS "[OK] ONNX Runtime: ${onnxruntime_SOURCE_DIR}")
        endif()
    endif()

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
    # Windows: SystemMediaTransportControls via C++/WinRT
    set(NOWPLAYING_SOURCES src/nowplaying/nowplaying_windows.cpp)
    set(NOWPLAYING_LIBS RuntimeObject)
    message(STATUS "  Platform: Windows (SMTC via C++/WinRT)")
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

# Metadata sources (TagLib-backed tag reading/writing)
set(METADATA_SOURCES
    src/metadata/MetadataFile.cpp
    src/metadata/xune_metadata.cpp
)

# Create xune_audio shared library
add_library(xune_audio SHARED
    src/xune_version.cpp
    ${NOWPLAYING_SOURCES}
    ${SMARTDJ_SOURCES}
    ${BEATTRACKING_SOURCES}
    ${METADATA_SOURCES}
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
elseif(UNIX)
    set_target_properties(xune_audio PROPERTIES
        BUILD_RPATH "$ORIGIN"
        INSTALL_RPATH "$ORIGIN"
    )
endif()

# Enable exports macro
target_compile_definitions(xune_audio PRIVATE XUNE_AUDIO_EXPORTS)
if(XUNE_USE_MLX)
    target_compile_definitions(xune_audio PRIVATE XUNE_USE_MLX)
endif()

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

# Link TagLib (static, for metadata reading/writing)
# TagLib 2.0 only exports INSTALL_INTERFACE includes, so we add BUILD_INTERFACE manually
target_link_libraries(xune_audio PRIVATE tag)
set(TAGLIB_INCLUDE_DIRS
    "${taglib_SOURCE_DIR}/taglib"
    "${taglib_SOURCE_DIR}/taglib/toolkit"
    "${taglib_SOURCE_DIR}/taglib/asf"
    "${taglib_SOURCE_DIR}/taglib/flac"
    "${taglib_SOURCE_DIR}/taglib/mp4"
    "${taglib_SOURCE_DIR}/taglib/mpeg"
    "${taglib_SOURCE_DIR}/taglib/mpeg/id3v2"
    "${taglib_SOURCE_DIR}/taglib/mpeg/id3v2/frames"
    "${taglib_SOURCE_DIR}/taglib/ogg"
    "${taglib_SOURCE_DIR}/taglib/ogg/flac"
    "${taglib_SOURCE_DIR}/taglib/ogg/vorbis"
    "${taglib_SOURCE_DIR}/taglib/riff"
    "${taglib_SOURCE_DIR}/taglib/riff/wav"
    "${taglib_BINARY_DIR}"
)
target_include_directories(xune_audio PRIVATE ${TAGLIB_INCLUDE_DIRS})

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
elseif(WIN32)
    # C++/WinRT for SMTC Now Playing integration
    target_compile_options(xune_audio PRIVATE /await)
    target_compile_definitions(xune_audio PRIVATE
        WINRT_LEAN_AND_MEAN
        WIN32_LEAN_AND_MEAN
        NOMINMAX
    )
endif()

# Post-build: copy to .NET runtime directory
if(EXISTS "${AUDIO_NATIVE_DIR}")
    add_custom_command(TARGET xune_audio POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${RUNTIME_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:xune_audio>" "${RUNTIME_DIR}/"
        COMMENT "Copying xune_audio library to ${RUNTIME_DIR}/"
    )
    # Create platform-specific symlinks (not needed on Windows — DLL name is xune_audio.dll)
    if(NOT WIN32)
        add_custom_command(TARGET xune_audio POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E create_symlink "$<TARGET_FILE_NAME:xune_audio>" "${RUNTIME_DIR}/libxune_audio$<IF:$<PLATFORM_ID:Darwin>,.dylib,.so>"
        )
    endif()

    # macOS: bundle libmlx and fix install names for hardened runtime compatibility
    if(APPLE AND NOT FORCE_ONNX)
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

    # Copy ONNX Runtime shared libraries to .NET runtime directory
    if(NOT APPLE OR FORCE_ONNX)
        if(WIN32)
            # DirectML NuGet: copy from runtimes/{rid}/native/
            set(ORT_COPY_DIR "${onnxruntime_SOURCE_DIR}/runtimes/${ORT_WIN_RID}/native")
            file(GLOB ORT_WIN_LIBS "${ORT_COPY_DIR}/*.dll")
            foreach(ORT_LIB ${ORT_WIN_LIBS})
                add_custom_command(TARGET xune_audio POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy "${ORT_LIB}" "${RUNTIME_DIR}/"
                )
            endforeach()
            add_custom_command(TARGET xune_audio POST_BUILD
                COMMENT "Copying ONNX Runtime DirectML libraries to ${RUNTIME_DIR}/"
            )
        else()
            add_custom_command(TARGET xune_audio POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy "${onnxruntime_SOURCE_DIR}/lib/${ORT_LIB_NAME}" "${RUNTIME_DIR}/"
                COMMENT "Copying ONNX Runtime library to ${RUNTIME_DIR}/"
            )
            # macOS + ONNX: fix install names so dyld finds ORT via @loader_path
            if(APPLE)
                file(GLOB _ort_versioned_dylibs "${onnxruntime_SOURCE_DIR}/lib/libonnxruntime.*.dylib")
                if(NOT _ort_versioned_dylibs)
                    message(FATAL_ERROR "No versioned ORT dylib found in ${onnxruntime_SOURCE_DIR}/lib/ — install_name_tool fixups cannot proceed")
                endif()
                list(GET _ort_versioned_dylibs 0 _ort_versioned_dylib)
                get_filename_component(_ort_versioned_name "${_ort_versioned_dylib}" NAME)
                add_custom_command(TARGET xune_audio POST_BUILD
                    COMMAND install_name_tool -id "@loader_path/${ORT_LIB_NAME}" "${RUNTIME_DIR}/${ORT_LIB_NAME}"
                    COMMAND install_name_tool -change "@rpath/${_ort_versioned_name}" "@loader_path/${ORT_LIB_NAME}"
                        "${RUNTIME_DIR}/$<TARGET_FILE_NAME:xune_audio>"
                    COMMAND install_name_tool -change "@rpath/${_ort_versioned_name}" "@loader_path/${ORT_LIB_NAME}"
                        "${RUNTIME_DIR}/libxune_audio.dylib"
                    COMMAND install_name_tool -id "@loader_path/libxune_audio.1.dylib"
                        "${RUNTIME_DIR}/$<TARGET_FILE_NAME:xune_audio>"
                    COMMAND install_name_tool -id "@loader_path/libxune_audio.1.dylib"
                        "${RUNTIME_DIR}/libxune_audio.dylib"
                    COMMAND codesign --force --sign - "${RUNTIME_DIR}/${ORT_LIB_NAME}"
                    COMMAND codesign --force --sign - "${RUNTIME_DIR}/$<TARGET_FILE_NAME:xune_audio>"
                    COMMAND codesign --force --sign - "${RUNTIME_DIR}/libxune_audio.dylib"
                    COMMENT "Fixing ONNX Runtime install names for hardened runtime"
                )
            endif()
            # Copy versioned .so and EP plugin libraries
            if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
                file(GLOB ORT_VERSIONED_LIBS "${onnxruntime_SOURCE_DIR}/lib/libonnxruntime.so.*")
                foreach(ORT_LIB ${ORT_VERSIONED_LIBS})
                    add_custom_command(TARGET xune_audio POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy "${ORT_LIB}" "${RUNTIME_DIR}/"
                    )
                endforeach()
                # Copy EP plugin .so files (CUDA, TensorRT, providers_shared)
                file(GLOB ORT_EP_LIBS "${onnxruntime_SOURCE_DIR}/lib/libonnxruntime_providers_*.so")
                foreach(ORT_LIB ${ORT_EP_LIBS})
                    add_custom_command(TARGET xune_audio POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy "${ORT_LIB}" "${RUNTIME_DIR}/"
                    )
                endforeach()
            endif()
        endif()
    endif()

    # Bundle FFTW3F DLL on Windows (not a system library)
    if(WIN32)
        get_filename_component(FFTW3F_DIR "${FFTW3F_LIBRARY}" DIRECTORY)
        find_file(FFTW3F_DLL NAMES "fftw3f.dll" "libfftw3f-3.dll" HINTS "${FFTW3F_DIR}" "${FFTW3F_DIR}/../bin")
        if(FFTW3F_DLL)
            add_custom_command(TARGET xune_audio POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy "${FFTW3F_DLL}" "${RUNTIME_DIR}/"
                COMMENT "Bundling FFTW3F DLL to ${RUNTIME_DIR}/"
            )
        endif()
    endif()

    # Bundle FFTW3F on Linux (not available in Flatpak/AppImage sandboxes)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        get_filename_component(FFTW3F_REALPATH "${FFTW3F_LIBRARY}" REALPATH)
        get_filename_component(FFTW3F_FILENAME "${FFTW3F_REALPATH}" NAME)
        add_custom_command(TARGET xune_audio POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy "${FFTW3F_REALPATH}" "${RUNTIME_DIR}/${FFTW3F_FILENAME}"
            COMMAND ${CMAKE_COMMAND} -E create_symlink "${FFTW3F_FILENAME}" "${RUNTIME_DIR}/libfftw3f.so.3"
            COMMAND ${CMAKE_COMMAND} -E create_symlink "${FFTW3F_FILENAME}" "${RUNTIME_DIR}/libfftw3f.so"
            COMMENT "Bundling FFTW3F to ${RUNTIME_DIR}/"
        )
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
    if(XUNE_USE_MLX)
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
        ${CMAKE_CURRENT_SOURCE_DIR}/tests
    )
    target_compile_definitions(smartdj_test PRIVATE
        XUNE_AUDIO_EXPORTS
        REFERENCE_DATA_DIR="${REFERENCE_DATA_DIR}"
        MODEL_PATH="${MODEL_PATH}"
    )
    if(XUNE_USE_MLX)
        target_compile_definitions(smartdj_test PRIVATE XUNE_USE_MLX)
    endif()

    include(GoogleTest)
    gtest_discover_tests(smartdj_test)

    # Beat tracking test
    set(BEAT_REFERENCE_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../src/Xune.App/Assets/Models/beat_reference_data")
    if(XUNE_USE_MLX)
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
        ${CMAKE_CURRENT_SOURCE_DIR}/tests
    )
    target_compile_definitions(beat_test PRIVATE
        XUNE_AUDIO_EXPORTS
        BEAT_REFERENCE_DATA_DIR="${BEAT_REFERENCE_DATA_DIR}"
        BEAT_MODEL_PATH="${BEAT_MODEL_PATH}"
    )
    if(XUNE_USE_MLX)
        target_compile_definitions(beat_test PRIVATE XUNE_USE_MLX)
    endif()
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

    # Metadata tests (format reading + picture writing)
    set(METADATA_TEST_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tests/testdata")

    add_executable(metadata_format_test tests/metadata_format_test.cpp)
    target_link_libraries(metadata_format_test PRIVATE xune_audio)
    target_include_directories(metadata_format_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_definitions(metadata_format_test PRIVATE TEST_DATA_DIR="${METADATA_TEST_DATA_DIR}")
    add_test(NAME metadata_format_test COMMAND metadata_format_test)

    add_executable(metadata_picture_test tests/metadata_picture_test.cpp)
    target_link_libraries(metadata_picture_test PRIVATE xune_audio)
    target_include_directories(metadata_picture_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_definitions(metadata_picture_test PRIVATE TEST_DATA_DIR="${METADATA_TEST_DATA_DIR}")
    add_test(NAME metadata_picture_test COMMAND metadata_picture_test)

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
message(STATUS "  TagLib:               ${taglib_SOURCE_DIR}")
if(APPLE AND NOT FORCE_ONNX)
message(STATUS "  Inference backend:    MLX (Metal GPU)")
elseif(WIN32)
message(STATUS "  Inference backend:    ONNX Runtime ${ORT_VERSION} DirectML (${ORT_WIN_RID})")
else()
message(STATUS "  Inference backend:    ONNX Runtime ${ORT_VERSION} (${ORT_PLATFORM})")
endif()
message(STATUS "  XuneAudio library:    $<TARGET_FILE:xune_audio>")
message(STATUS "  Runtime ID:           ${RUNTIME_ID}")
message(STATUS "  Platform libraries:   ${PLATFORM_LIBRARIES}")
message(STATUS "========================================")
message(STATUS "")
