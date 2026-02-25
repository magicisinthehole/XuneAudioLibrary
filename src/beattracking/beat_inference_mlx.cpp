/**
 * @file beat_inference_mlx.cpp
 * @brief MLX Metal GPU backend for Beat This! beat tracking inference.
 *
 * Implements the Beat This! small forward pass using MLX C++ API:
 *   Input: (B, T, 128) log-mel spectrogram at 50 fps
 *   Output: (B, T) beat logits + (B, T) downbeat logits
 *
 * Architecture (small variant, ~2.1M params):
 *   1. Stem: BN1d(128) → Conv2d(1→32, k=4×3, s=4×1) → BN2d → GELU
 *   2. Frontend Block ×3: PartialFTTransformer → Conv2d(double channels, k=2×3, s=2×1) → BN2d → GELU
 *      Channels: 32→64→128→256, Freq: 32→16→8→4
 *   3. Projection: reshape(B,T,1024) → Linear(1024→128)
 *   4. Transformer ×6: RMSNorm → Attention(128, 4 heads, head_dim=32, RoPE, sigmoid gating) → FFN(128→512→128)
 *   5. Final RMSNorm(128)
 *   6. SumHead: Linear(128→2) → beat = logit[0]+logit[1], downbeat = logit[1]
 */

#include "beat_inference.h"

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
namespace beattracking {

// ============================================================================
// Model Constants (Beat This! small)
// ============================================================================

static constexpr int kDim = 128;         // transformer_dim
static constexpr int kDepth = 6;         // n_layers
static constexpr int kHeads = 4;         // n_heads (transformer)
static constexpr int kDimHead = 32;      // head_dim
static constexpr int kFfMult = 4;        // ff_mult
static constexpr int kFfDim = kDim * kFfMult;  // 512
static constexpr int kStemDim = 32;
static constexpr int kNMels = 128;
static constexpr float kBnEps = 1e-5f;

// Frontend block config: {in_channels, out_channels, n_heads, freq_bins_before_conv}
struct FrontendBlockConfig {
    int in_ch, out_ch, n_heads, freq_in;
};
static constexpr FrontendBlockConfig kBlocks[3] = {
    {32, 64, 1, 32},    // Block 0
    {64, 128, 2, 16},   // Block 1
    {128, 256, 4, 8},   // Block 2
};

// ============================================================================
// Precomputed BatchNorm parameters (eval mode affine transform)
// ============================================================================

struct BnParams {
    mx::array scale{mx::zeros({0})};
    mx::array shift{mx::zeros({0})};
};

// ============================================================================
// PIMPL: MLX state
// ============================================================================

struct BeatInference::Impl {
    std::unordered_map<std::string, mx::array> weights;
    bool ready = false;

    // Precomputed BN params for stem and frontend blocks
    BnParams stem_bn1d;
    BnParams stem_bn2d;
    BnParams block_bn2d[3];

    // Conv weights transposed to OHWI for MLX conv2d
    mx::array stem_conv_w{mx::zeros({0})};
    mx::array block_conv_w[3] = {mx::zeros({0}), mx::zeros({0}), mx::zeros({0})};

    // ---------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------

    const mx::array& W(const std::string& key) const {
        return weights.at(key);
    }

    mx::array linear(const mx::array& x, const std::string& w_key,
                     const std::string& b_key) const {
        return mx::matmul(x, mx::transpose(W(w_key))) + W(b_key);
    }

    mx::array linear_no_bias(const mx::array& x, const std::string& w_key) const {
        return mx::matmul(x, mx::transpose(W(w_key)));
    }

    static mx::array gelu(const mx::array& x) {
        static const float inv_sqrt2 = 1.0f / std::sqrt(2.0f);
        return x * (1.0f + mx::erf(x * inv_sqrt2)) * 0.5f;
    }

    // BatchNorm eval mode: x * scale + shift (channel dimension varies)
    static mx::array apply_bn(const mx::array& x, const BnParams& bn, int ch_axis) {
        // Reshape scale/shift to broadcast along channel axis
        // Build shape as initializer for SmallVector
        int ndim = x.ndim();
        int ch_size = bn.scale.shape(0);
        if (ndim == 3) {
            // (B, C, T) or (B, T, C)
            mx::Shape shape = {1, 1, 1};
            shape[ch_axis] = ch_size;
            return x * mx::reshape(bn.scale, shape) + mx::reshape(bn.shift, shape);
        } else {
            // (B, C, H, W)
            mx::Shape shape = {1, 1, 1, 1};
            shape[ch_axis] = ch_size;
            return x * mx::reshape(bn.scale, shape) + mx::reshape(bn.shift, shape);
        }
    }

    // Precompute BN params: scale = gamma/sqrt(var+eps), shift = beta - mean*scale
    static BnParams make_bn(const mx::array& gamma, const mx::array& beta,
                            const mx::array& mean, const mx::array& var) {
        auto scale = gamma / mx::sqrt(var + kBnEps);
        auto shift = beta - mean * scale;
        mx::eval(scale);
        mx::eval(shift);
        return {scale, shift};
    }

    // ---------------------------------------------------------------
    // RoPE Attention (x-transformers rotation, sigmoid gating)
    // ---------------------------------------------------------------
    //
    // x-transformers rotate_half with duplicated frequencies is equivalent
    // to MLX fast::rope with traditional=true (adjacent-pair rotation).
    // The fused Metal kernel replaces CPU-side freq computation + 6
    // intermediate tensor ops per call.

    mx::array attention(const mx::array& x, int n_heads,
                        const std::string& prefix) const {
        // x: (batch, seq, dim)
        int batch = x.shape(0);
        int seq = x.shape(1);
        int dim = x.shape(2);

        // RMSNorm
        auto normed = mx::fast::rms_norm(x, W(prefix + ".norm.gamma"), kBnEps);

        // QKV: (batch, seq, 3*n_heads*dim_head)
        auto qkv = linear_no_bias(normed, prefix + ".to_qkv.weight");

        // Reshape: (batch, seq, 3, n_heads, dim_head)
        qkv = mx::reshape(qkv, {batch, seq, 3, n_heads, kDimHead});
        // Permute: (3, batch, n_heads, seq, dim_head)
        qkv = mx::transpose(qkv, {2, 0, 3, 1, 4});

        auto q = mx::slice(qkv, {0, 0, 0, 0, 0}, {1, batch, n_heads, seq, kDimHead});
        q = mx::squeeze(q, 0);
        auto k = mx::slice(qkv, {1, 0, 0, 0, 0}, {2, batch, n_heads, seq, kDimHead});
        k = mx::squeeze(k, 0);
        auto v = mx::slice(qkv, {2, 0, 0, 0, 0}, {3, batch, n_heads, seq, kDimHead});
        v = mx::squeeze(v, 0);

        // Apply RoPE (traditional=true matches x-transformers rotate_half)
        q = mx::fast::rope(q, kDimHead, /*traditional=*/true, /*base=*/10000.0f,
                           /*scale=*/1.0f, /*offset=*/0);
        k = mx::fast::rope(k, kDimHead, /*traditional=*/true, /*base=*/10000.0f,
                           /*scale=*/1.0f, /*offset=*/0);

        // Scaled dot-product attention
        float scale = 1.0f / std::sqrt(static_cast<float>(kDimHead));
        auto attn_out = mx::fast::scaled_dot_product_attention(q, k, v, scale);

        // Sigmoid gating: gates (batch, seq, n_heads) → (batch, n_heads, seq, 1)
        auto gates = linear(normed, prefix + ".to_gates.weight", prefix + ".to_gates.bias");
        gates = mx::transpose(gates, {0, 2, 1});    // (batch, n_heads, seq)
        gates = mx::expand_dims(gates, -1);          // (batch, n_heads, seq, 1)
        attn_out = attn_out * mx::sigmoid(gates);

        // Merge heads: (batch, seq, dim)
        attn_out = mx::transpose(attn_out, {0, 2, 1, 3});
        attn_out = mx::reshape(attn_out, {batch, seq, n_heads * kDimHead});

        // Output projection (to_out.0 is the first element of a Sequential)
        return linear_no_bias(attn_out, prefix + ".to_out.0.weight");
    }

    // ---------------------------------------------------------------
    // Feed-Forward (RMSNorm → Linear → GELU → Linear)
    // ---------------------------------------------------------------

    mx::array feed_forward(const mx::array& x, const std::string& prefix) const {
        auto normed = mx::fast::rms_norm(x, W(prefix + ".net.0.gamma"), kBnEps);
        auto h = linear(normed, prefix + ".net.1.weight", prefix + ".net.1.bias");
        h = gelu(h);
        return linear(h, prefix + ".net.4.weight", prefix + ".net.4.bias");
    }

    // ---------------------------------------------------------------
    // PartialFTTransformer: freq attention + time attention
    // ---------------------------------------------------------------

    mx::array partial_ft(const mx::array& x, int block_idx) const {
        // x: (B, C, F, T) in NCHW
        int B = x.shape(0);
        int C = x.shape(1);
        int F = x.shape(2);
        int T = x.shape(3);
        int n_heads = kBlocks[block_idx].n_heads;

        std::string prefix = "frontend.blocks." + std::to_string(block_idx) + ".partial";

        // Freq attention: (B, C, F, T) → (B*T, F, C)
        auto xf = mx::transpose(x, {0, 3, 2, 1});     // (B, T, F, C)
        xf = mx::reshape(xf, {B * T, F, C});

        xf = xf + attention(xf, n_heads, prefix + ".attnF");
        xf = xf + feed_forward(xf, prefix + ".ffF");
        mx::eval(xf);

        // Time attention: (B*T, F, C) → (B*F, T, C)
        xf = mx::reshape(xf, {B, T, F, C});
        auto xt = mx::transpose(xf, {0, 2, 1, 3});    // (B, F, T, C)
        xt = mx::reshape(xt, {B * F, T, C});

        xt = xt + attention(xt, n_heads, prefix + ".attnT");
        xt = xt + feed_forward(xt, prefix + ".ffT");
        mx::eval(xt);

        // Restore: (B*F, T, C) → (B, C, F, T)
        xt = mx::reshape(xt, {B, F, T, C});
        return mx::transpose(xt, {0, 3, 1, 2});        // (B, C, F, T)
    }

    // ---------------------------------------------------------------
    // Conv2d helper (NCHW ↔ NHWC for MLX)
    // ---------------------------------------------------------------

    mx::array conv2d(const mx::array& x_nchw, const mx::array& w_ohwi,
                     std::pair<int, int> stride, std::pair<int, int> padding) const {
        // x_nchw: (B, C, H, W) → NHWC: (B, H, W, C)
        auto x_nhwc = mx::transpose(x_nchw, {0, 2, 3, 1});
        auto y_nhwc = mx::conv2d(x_nhwc, w_ohwi, stride, padding);
        // NHWC → NCHW: (B, C, H, W)
        return mx::transpose(y_nhwc, {0, 3, 1, 2});
    }
};

// ============================================================================
// BeatInference
// ============================================================================

BeatInference::BeatInference() : impl_(std::make_unique<Impl>()) {}
BeatInference::~BeatInference() = default;
BeatInference::BeatInference(BeatInference&&) noexcept = default;
BeatInference& BeatInference::operator=(BeatInference&&) noexcept = default;

bool BeatInference::LoadModel(const std::string& model_path) {
    std::lock_guard<std::mutex> lock(xune::mlx_gpu_mutex());
    try {
        mx::set_cache_limit(256 * 1024 * 1024);

        auto [weights, metadata] = mx::load_safetensors(model_path);
        impl_->weights = std::move(weights);

        // Precompute stem BN1d params (channel dim = 128)
        impl_->stem_bn1d = Impl::make_bn(
            impl_->W("frontend.stem.bn1d.weight"),
            impl_->W("frontend.stem.bn1d.bias"),
            impl_->W("frontend.stem.bn1d.running_mean"),
            impl_->W("frontend.stem.bn1d.running_var"));

        // Precompute stem BN2d params (channel dim = 32)
        impl_->stem_bn2d = Impl::make_bn(
            impl_->W("frontend.stem.bn2d.weight"),
            impl_->W("frontend.stem.bn2d.bias"),
            impl_->W("frontend.stem.bn2d.running_mean"),
            impl_->W("frontend.stem.bn2d.running_var"));

        // Transpose stem conv weight: OIHW → OHWI
        impl_->stem_conv_w = mx::transpose(impl_->W("frontend.stem.conv2d.weight"), {0, 2, 3, 1});
        mx::eval(impl_->stem_conv_w);

        // Precompute frontend block BN2d params and conv weights
        for (int i = 0; i < 3; i++) {
            std::string bp = "frontend.blocks." + std::to_string(i);

            impl_->block_bn2d[i] = Impl::make_bn(
                impl_->W(bp + ".norm.weight"),
                impl_->W(bp + ".norm.bias"),
                impl_->W(bp + ".norm.running_mean"),
                impl_->W(bp + ".norm.running_var"));

            impl_->block_conv_w[i] = mx::transpose(impl_->W(bp + ".conv2d.weight"), {0, 2, 3, 1});
            mx::eval(impl_->block_conv_w[i]);
        }

        // Verify critical weights exist
        const char* required_keys[] = {
            "frontend.stem.conv2d.weight",
            "frontend.linear.weight",
            "transformer_blocks.layers.0.0.to_qkv.weight",
            "transformer_blocks.layers.5.1.net.4.weight",
            "transformer_blocks.norm.gamma",
            "task_heads.beat_downbeat_lin.weight",
        };
        for (const char* key : required_keys) {
            if (impl_->weights.find(key) == impl_->weights.end()) {
                fprintf(stderr, "[xune_beat] Missing weight key: %s\n", key);
                return false;
            }
        }

        impl_->ready = true;
        fprintf(stderr, "[xune_beat] MLX model loaded: %s (%zu tensors)\n",
                model_path.c_str(), impl_->weights.size());
        return true;
    } catch (const std::exception& e) {
        fprintf(stderr, "[xune_beat] Failed to load MLX model: %s\n", e.what());
        return false;
    }
}

bool BeatInference::IsReady() const {
    return impl_ && impl_->ready;
}

bool BeatInference::RunInference(const float* mel_data, int batch_size, int n_frames,
                                  std::vector<float>& beat_logits,
                                  std::vector<float>& downbeat_logits) {
    if (!impl_->ready || !mel_data || batch_size <= 0 || n_frames <= 0) {
        return false;
    }

    std::lock_guard<std::mutex> lock(xune::mlx_gpu_mutex());
    try {
        // Input: (B, T, 128) time-first mel spectrogram
        auto x = mx::array(
            mel_data,
            {batch_size, n_frames, kNMels},
            mx::float32);

        // ==== STEM ====
        // Permute: (B, T, 128) → (B, 128, T) for BN1d
        x = mx::transpose(x, {0, 2, 1});

        // BatchNorm1d (element-wise) → Conv2d → BN2d → GELU
        // Defer eval — the lazy graph (BN + expand + conv + BN + GELU) is shallow.
        x = Impl::apply_bn(x, impl_->stem_bn1d, /*ch_axis=*/1);
        x = mx::expand_dims(x, 1);
        x = impl_->conv2d(x, impl_->stem_conv_w, {4, 1}, {0, 1});
        x = Impl::apply_bn(x, impl_->stem_bn2d, /*ch_axis=*/1);
        x = Impl::gelu(x);
        mx::eval(x);

        // ==== FRONTEND BLOCKS ====
        for (int i = 0; i < 3; i++) {
            // PartialFTTransformer (evaluates internally after freq/time attention)
            x = impl_->partial_ft(x, i);
            // No eval here — partial_ft already evaluated; only a lazy transpose remains.

            // Conv2d (double channels, halve freq) + BN2d + GELU
            x = impl_->conv2d(x, impl_->block_conv_w[i], {2, 1}, {0, 1});
            x = Impl::apply_bn(x, impl_->block_bn2d[i], /*ch_axis=*/1);
            x = Impl::gelu(x);
            mx::eval(x);
        }

        // ==== PROJECTION ====
        // x: (B, 256, 4, T) → (B, T, 1024) → (B, T, 128)
        x = mx::transpose(x, {0, 3, 1, 2});                  // (B, T, 256, 4)
        x = mx::reshape(x, {batch_size, n_frames, 256 * 4});  // (B, T, 1024)
        x = impl_->linear(x, "frontend.linear.weight", "frontend.linear.bias");
        mx::eval(x);

        // ==== TRANSFORMER (6 layers) ====
        for (int i = 0; i < kDepth; i++) {
            std::string attn_prefix = "transformer_blocks.layers." + std::to_string(i) + ".0";
            std::string ff_prefix = "transformer_blocks.layers." + std::to_string(i) + ".1";

            x = x + impl_->attention(x, kHeads, attn_prefix);
            x = x + impl_->feed_forward(x, ff_prefix);
            mx::eval(x);
        }

        // ==== FINAL NORM ====
        x = mx::fast::rms_norm(x, impl_->W("transformer_blocks.norm.gamma"), kBnEps);
        mx::eval(x);

        // ==== SUM HEAD ====
        auto bd = impl_->linear(x, "task_heads.beat_downbeat_lin.weight",
                                   "task_heads.beat_downbeat_lin.bias");
        // bd: (B, T, 2). Avoid mx::slice on the inner dimension — it returns
        // strided views that share the underlying buffer, making memcpy unsafe.
        // Use compute ops that allocate new contiguous tensors instead.

        // beat = col0 + col1: sum along last dimension
        auto beat = mx::sum(bd, /*axis=*/-1);  // (B, T)

        // downbeat = col1: select via matmul with [0, 1]^T
        auto sel = mx::array({0.0f, 1.0f}, {2, 1});
        auto downbeat = mx::squeeze(mx::matmul(bd, sel), -1);  // (B, T)

        mx::eval({beat, downbeat});

        // Copy results
        size_t output_count = static_cast<size_t>(batch_size) * n_frames;
        beat_logits.resize(output_count);
        downbeat_logits.resize(output_count);
        std::memcpy(beat_logits.data(), beat.data<float>(), output_count * sizeof(float));
        std::memcpy(downbeat_logits.data(), downbeat.data<float>(), output_count * sizeof(float));

        return true;
    } catch (const std::exception& e) {
        fprintf(stderr, "[xune_beat] MLX inference failed: %s\n", e.what());
        return false;
    }
}

}  // namespace beattracking
}  // namespace xune
