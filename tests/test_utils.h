/**
 * @file test_utils.h
 * @brief Shared test helpers for audio library tests.
 */

#pragma once

#include <cmath>
#include <cstddef>
#include <fstream>
#include <string>
#include <vector>

namespace xune {
namespace test {

inline std::vector<float> LoadFloatBin(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) return {};
    auto size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<float> data(static_cast<size_t>(size) / sizeof(float));
    file.read(reinterpret_cast<char*>(data.data()), size);
    return data;
}

inline float CosineSimilarity(const float* a, const float* b, size_t n) {
    double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
    for (size_t i = 0; i < n; ++i) {
        dot += static_cast<double>(a[i]) * b[i];
        norm_a += static_cast<double>(a[i]) * a[i];
        norm_b += static_cast<double>(b[i]) * b[i];
    }
    if (norm_a == 0.0 || norm_b == 0.0) return 0.0f;
    return static_cast<float>(dot / (std::sqrt(norm_a) * std::sqrt(norm_b)));
}

}  // namespace test
}  // namespace xune
