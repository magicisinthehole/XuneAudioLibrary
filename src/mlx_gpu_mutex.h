/**
 * @file mlx_gpu_mutex.h
 * @brief Process-wide mutex for serializing MLX Metal GPU access.
 *
 * MLX's Metal backend uses a single command buffer per process.
 * Concurrent mx::eval() calls from different threads cause
 * "addCompletedHandler after commit" assertion failures.
 *
 * All code paths that call mx::eval() must hold this lock.
 */

#pragma once

#include <mutex>

namespace xune {

/// Returns the process-wide mutex that guards all mx::eval() calls.
inline std::mutex& mlx_gpu_mutex() {
    static std::mutex instance;
    return instance;
}

}  // namespace xune
