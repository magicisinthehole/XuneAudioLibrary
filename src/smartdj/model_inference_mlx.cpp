
/**
 * @file model_inference_mlx.cpp
 * @brief MLX Metal GPU backend for Myna model inference.
 *
 * Backend: MLX (Apple Silicon Metal GPU). Used on macOS.
 * On Windows/Linux, model_inference_ort.cpp provides the ONNX Runtime backend.
 *
 * Implements the Myna hybrid ViT forward pass using MLX C++ API:
 *   Input: (batch, 1, 128, 96) mel spectrogram
 *   Output: (batch, 768) embedding
 *
 * Architecture (vit-s-32 hybrid):
 *   1. Patch embed A (16x16 squares) + sincos positional embedding
 *   2. Patch embed B (128x2 strips) + sincos positional embedding
 *   3. Fuse along batch dim → (2B, 48, 384)
 *   4. Shared transformer (12 layers, 6 heads, dim=384, mlp=1536)
 *   5. Split → mean pool → concat → (B, 768)
 */

#include "model_inference.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>

#include <mlx/mlx.h>
#include <mlx/memory.h>

#include "../mlx_gpu_mutex.h"

namespace mx = mlx::core;

namespace xune {
namespace smartdj {

// ============================================================================
// Model Constants (must match training config: vit-s-32 + additional_patch_size=(128,2))
// ============================================================================

static constexpr int kDim = 384;
static constexpr int kDepth = 12;
static constexpr int kHeads = 6;
static constexpr int kDimHead = 64;
static constexpr int kMlpDim = 1536;
static constexpr int kInnerDim = kHeads * kDimHead;  // 384
static constexpr float kScale = 1.0f / 8.0f;         // 1/sqrt(64)
static constexpr float kLayerNormEps = 1e-5f;

// Patch A: 16x16 squares
static constexpr int kPatchHA = 16;
static constexpr int kPatchWA = 16;
static constexpr int kGridHA = 8;   // 128 / 16
static constexpr int kGridWA = 6;   // 96 / 16
static constexpr int kSeqLen = 48;  // 8 * 6 = 1 * 48

// Patch B: 128x2 vertical strips
static constexpr int kPatchHB = 128;
static constexpr int kPatchWB = 2;
static constexpr int kGridHB = 1;   // 128 / 128
static constexpr int kGridWB = 48;  // 96 / 2

// ============================================================================
// Sincos Positional Embedding (deterministic, computed at init)
// ============================================================================

static mx::array posemb_sincos_2d(int h, int w, int dim, int temperature = 10000) {
    int quarter_dim = dim / 4;

    // Frequency bands: omega[k] = 1 / (temperature ^ (k / (quarter_dim - 1)))
    std::vector<float> omega_vec(quarter_dim);
    for (int k = 0; k < quarter_dim; k++) {
        float exp = static_cast<float>(k) / static_cast<float>(quarter_dim - 1);
        omega_vec[k] = 1.0f / std::pow(static_cast<float>(temperature), exp);
    }

    // Position vectors (ij-indexed meshgrid, flattened)
    int n = h * w;
    std::vector<float> pe(n * dim);

    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            int pos = i * w + j;
            float y = static_cast<float>(i);
            float x = static_cast<float>(j);

            for (int k = 0; k < quarter_dim; k++) {
                float x_angle = x * omega_vec[k];
                float y_angle = y * omega_vec[k];

                // Layout: [sin(x), cos(x), sin(y), cos(y)]
                pe[pos * dim + k]                  = std::sin(x_angle);
                pe[pos * dim + quarter_dim + k]    = std::cos(x_angle);
                pe[pos * dim + 2 * quarter_dim + k] = std::sin(y_angle);
                pe[pos * dim + 3 * quarter_dim + k] = std::cos(y_angle);
            }
        }
    }

    return mx::array(pe.data(), {n, dim}, mx::float32);
}

// ============================================================================
// PIMPL: MLX state
// ============================================================================

struct ModelInference::Impl {
    std::unordered_map<std::string, mx::array> weights;
    mx::array pos_emb_a{mx::zeros({0})};  // (48, 384) sincos for patch A, set in LoadModel
    mx::array pos_emb_b{mx::zeros({0})};  // (48, 384) sincos for patch B, set in LoadModel
    bool ready = false;

    // Helper: get weight by key
    const mx::array& W(const std::string& key) const {
        return weights.at(key);
    }

    // Helper: linear projection y = x @ W^T + b
    mx::array linear(const mx::array& x, const std::string& w_key,
                     const std::string& b_key) const {
        auto y = mx::matmul(x, mx::transpose(W(w_key)));
        return y + W(b_key);
    }

    // Helper: linear projection without bias y = x @ W^T
    mx::array linear_no_bias(const mx::array& x, const std::string& w_key) const {
        return mx::matmul(x, mx::transpose(W(w_key)));
    }

    // GELU activation: x * (1 + erf(x / sqrt(2))) / 2
    static mx::array gelu(const mx::array& x) {
        static const float inv_sqrt2 = 1.0f / std::sqrt(2.0f);
        return x * (1.0f + mx::erf(x * inv_sqrt2)) * 0.5f;
    }

    // Patch embedding: rearrange + LayerNorm + Linear + LayerNorm
    mx::array patch_embed(const mx::array& img, int batch_size,
                          int patch_h, int patch_w,
                          int grid_h, int grid_w,
                          const std::string& prefix,
                          const mx::array& pos_emb) const {
        int patch_dim = patch_h * patch_w;  // channels=1

        // Rearrange: (B, 1, grid_h*patch_h, grid_w*patch_w) -> (B, grid_h*grid_w, patch_dim)
        // Step 1: reshape to (B, 1, grid_h, patch_h, grid_w, patch_w)
        auto x = mx::reshape(img, {batch_size, 1, grid_h, patch_h, grid_w, patch_w});
        // Step 2: permute to (B, grid_h, grid_w, patch_h, patch_w, 1)
        x = mx::transpose(x, {0, 2, 4, 3, 5, 1});
        // Step 3: reshape to (B, num_patches, patch_dim)
        int num_patches = grid_h * grid_w;
        x = mx::reshape(x, {batch_size, num_patches, patch_dim});

        // LayerNorm(patch_dim) + Linear(patch_dim -> dim) + LayerNorm(dim)
        x = mx::fast::layer_norm(x, W(prefix + ".1.weight"), W(prefix + ".1.bias"), kLayerNormEps);
        x = linear(x, prefix + ".2.weight", prefix + ".2.bias");
        x = mx::fast::layer_norm(x, W(prefix + ".3.weight"), W(prefix + ".3.bias"), kLayerNormEps);

        // Add sincos positional embedding
        x = x + pos_emb;

        return x;
    }

    // Single transformer layer: pre-norm attention + pre-norm FFN
    mx::array transformer_layer(const mx::array& x, int layer_idx) const {
        std::string attn_prefix = "transformer.layers." + std::to_string(layer_idx) + ".0";
        std::string ffn_prefix = "transformer.layers." + std::to_string(layer_idx) + ".1";

        int batch = x.shape(0);
        int seq = x.shape(1);

        // === Pre-norm Multi-Head Attention ===
        auto normed = mx::fast::layer_norm(
            x, W(attn_prefix + ".norm.weight"), W(attn_prefix + ".norm.bias"), kLayerNormEps);

        // QKV projection: (batch, seq, dim) -> (batch, seq, 3*inner_dim)
        auto qkv = linear_no_bias(normed, attn_prefix + ".to_qkv.weight");

        // Split into Q, K, V along last dim
        auto qkv_split = mx::split(qkv, 3, /*axis=*/2);
        auto& q_flat = qkv_split[0];  // (batch, seq, inner_dim)
        auto& k_flat = qkv_split[1];
        auto& v_flat = qkv_split[2];

        // Reshape for multi-head: (batch, seq, heads, dim_head) -> (batch, heads, seq, dim_head)
        auto q = mx::transpose(mx::reshape(q_flat, {batch, seq, kHeads, kDimHead}), {0, 2, 1, 3});
        auto k = mx::transpose(mx::reshape(k_flat, {batch, seq, kHeads, kDimHead}), {0, 2, 1, 3});
        auto v = mx::transpose(mx::reshape(v_flat, {batch, seq, kHeads, kDimHead}), {0, 2, 1, 3});

        // Scaled dot-product attention
        auto attn_out = mx::fast::scaled_dot_product_attention(q, k, v, kScale);

        // Merge heads: (batch, heads, seq, dim_head) -> (batch, seq, inner_dim)
        attn_out = mx::reshape(mx::transpose(attn_out, {0, 2, 1, 3}), {batch, seq, kInnerDim});

        // Output projection
        attn_out = linear_no_bias(attn_out, attn_prefix + ".to_out.weight");

        // Residual
        auto h = x + attn_out;

        // === Pre-norm Feed-Forward ===
        normed = mx::fast::layer_norm(
            h, W(ffn_prefix + ".net.0.weight"), W(ffn_prefix + ".net.0.bias"), kLayerNormEps);

        auto ffn = linear(normed, ffn_prefix + ".net.1.weight", ffn_prefix + ".net.1.bias");
        ffn = gelu(ffn);
        ffn = linear(ffn, ffn_prefix + ".net.3.weight", ffn_prefix + ".net.3.bias");

        // Residual
        return h + ffn;
    }
};

// ============================================================================
// ModelInference
// ============================================================================

ModelInference::ModelInference() : impl_(std::make_unique<Impl>()) {}
ModelInference::~ModelInference() = default;
ModelInference::ModelInference(ModelInference&&) noexcept = default;
ModelInference& ModelInference::operator=(ModelInference&&) noexcept = default;

bool ModelInference::LoadModel(const std::string& model_path,
                               const std::string& cache_dir) {
    std::lock_guard<std::mutex> lock(xune::mlx_gpu_mutex());
    try {
        // Limit the Metal buffer cache to prevent unbounded growth between
        // inference calls. Freed activation buffers are returned to the system
        // rather than held in the pool. 256 MB is enough to cache common buffer
        // sizes for reuse without hoarding memory.
        mx::set_cache_limit(256 * 1024 * 1024);

        // Load SafeTensors weights
        auto [weights, metadata] = mx::load_safetensors(model_path);
        impl_->weights = std::move(weights);

        // Compute sincos positional embeddings (deterministic)
        impl_->pos_emb_a = posemb_sincos_2d(kGridHA, kGridWA, kDim);
        impl_->pos_emb_b = posemb_sincos_2d(kGridHB, kGridWB, kDim);

        // Eagerly evaluate positional embeddings
        mx::eval(impl_->pos_emb_a);
        mx::eval(impl_->pos_emb_b);

        // Verify critical weights exist
        const char* required_keys[] = {
            "to_patch_embedding.2.weight",
            "to_patch_embedding_b.2.weight",
            "transformer.norm.weight",
            "transformer.layers.0.0.to_qkv.weight",
            "transformer.layers.11.1.net.3.weight",
        };
        for (const char* key : required_keys) {
            if (impl_->weights.find(key) == impl_->weights.end()) {
                fprintf(stderr, "[xune_embedding] Missing weight key: %s\n", key);
                return false;
            }
        }

        impl_->ready = true;
        fprintf(stderr, "[xune_embedding] MLX model loaded: %s (%zu tensors)\n",
                model_path.c_str(), impl_->weights.size());
        return true;
    } catch (const std::exception& e) {
        fprintf(stderr, "[xune_embedding] Failed to load MLX model: %s\n", e.what());
        return false;
    }
}

bool ModelInference::IsReady() const {
    return impl_ && impl_->ready;
}

bool ModelInference::RunInferenceInto(const float* input_data, int batch_size,
                                      int n_mels, int n_frames,
                                      float* output_buffer, int output_buffer_size) {
    if (!impl_->ready || !input_data || batch_size <= 0 ||
        !output_buffer || output_buffer_size < batch_size * kEmbeddingDim) {
        return false;
    }

    std::lock_guard<std::mutex> lock(xune::mlx_gpu_mutex());
    try {
        // Create input tensor: (batch_size, 1, n_mels, n_frames)
        // MLX copies the data, so input_data doesn't need to outlive this call.
        auto img = mx::array(
            input_data,
            {batch_size, 1, n_mels, n_frames},
            mx::float32);

        // === Patch Embeddings ===
        // Patch A: 16x16 squares → (B, 48, 384)
        auto x_a = impl_->patch_embed(
            img, batch_size, kPatchHA, kPatchWA, kGridHA, kGridWA,
            "to_patch_embedding", impl_->pos_emb_a);

        // Patch B: 128x2 strips → (B, 48, 384)
        auto x_b = impl_->patch_embed(
            img, batch_size, kPatchHB, kPatchWB, kGridHB, kGridWB,
            "to_patch_embedding_b", impl_->pos_emb_b);

        // === Fuse along batch dimension ===
        // (2*B, 48, 384) — transformer processes each element independently
        auto x = mx::concatenate({x_a, x_b}, /*axis=*/0);
        mx::eval(x);  // Materialize patch embeddings, free input tensor

        // === Transformer (12 layers) ===
        // Evaluate after each layer so MLX can free that layer's intermediates
        // (Q, K, V, attention scores, FFN activations). Without this, lazy
        // evaluation accumulates the full graph across all 12 layers, and peak
        // GPU memory scales with depth × batch — causing OOM at large batches.
        for (int i = 0; i < kDepth; i++) {
            x = impl_->transformer_layer(x, i);
            mx::eval(x);
        }

        // === Final LayerNorm ===
        x = mx::fast::layer_norm(
            x,
            impl_->W("transformer.norm.weight"),
            impl_->W("transformer.norm.bias"),
            kLayerNormEps);

        // === Split, Mean Pool, Concat ===
        // Split back into A and B halves
        int doubled = 2 * batch_size;
        auto x_out_a = mx::slice(x, {0, 0, 0}, {batch_size, kSeqLen, kDim});
        auto x_out_b = mx::slice(x, {batch_size, 0, 0}, {doubled, kSeqLen, kDim});

        // Mean pool over sequence dimension (axis=1)
        x_out_a = mx::mean(x_out_a, /*axis=*/1);  // (B, 384)
        x_out_b = mx::mean(x_out_b, /*axis=*/1);  // (B, 384)

        // Concatenate → (B, 768)
        auto result = mx::concatenate({x_out_a, x_out_b}, /*axis=*/1);

        // Trigger Metal GPU computation for final ops
        mx::eval(result);

        // Copy directly from MLX result to caller's buffer
        size_t output_count = static_cast<size_t>(batch_size) * kEmbeddingDim;
        std::memcpy(output_buffer, result.data<float>(), output_count * sizeof(float));
        
        // Release Metal buffer cache to prevent memory accumulation across batches
        mx::clear_cache();

        return true;
    } catch (const std::exception& e) {
        fprintf(stderr, "[xune_embedding] MLX inference failed: %s\n", e.what());
        return false;
    }
}

}  // namespace smartdj
}  // namespace xune
