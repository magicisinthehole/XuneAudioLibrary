/**
 * @file ort_session.h
 * @brief Shared ONNX Runtime session wrapper.
 *
 * Owns the ORT lifecycle: env, session options, EP registration, session
 * creation, and CPU memory info. Model-specific inference classes compose
 * this rather than duplicating ORT boilerplate.
 *
 * A single OrtEnv is shared process-wide (ORT recommendation) via a
 * static singleton. Each OrtSession owns its own session and options.
 */

#pragma once

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <mutex>

#ifdef _WIN32
#include <windows.h>
#endif

#include <onnxruntime_c_api.h>

namespace xune {

class OrtSession {
public:
    explicit OrtSession(const char* log_tag)
        : api_(OrtGetApiBase()->GetApi(ORT_API_VERSION)),
          log_tag_(log_tag) {}

    ~OrtSession() {
        if (memory_info_) api_->ReleaseMemoryInfo(memory_info_);
        if (session_) api_->ReleaseSession(session_);
        if (session_options_) api_->ReleaseSessionOptions(session_options_);
    }

    OrtSession(const OrtSession&) = delete;
    OrtSession& operator=(const OrtSession&) = delete;

    bool LoadModel(const std::string& model_path,
                   const std::string& cache_dir,
                   const std::string& opt_model_name) {
        auto& api = *api_;

        // Session options
        if (!CheckStatus(api.CreateSessionOptions(&session_options_)))
            return false;

        // Check for a pre-optimized model in the cache directory.
        // If found, load it directly with minimal optimization (already done).
        // Otherwise, load the original model with full optimization and write
        // the result to the cache for next time.
        std::string effective_model = model_path;
        if (!cache_dir.empty()) {
            std::string opt_path = cache_dir + "/" + opt_model_name;
            FILE* f = fopen(opt_path.c_str(), "r");
            if (f) {
                fclose(f);
                effective_model = opt_path;
                if (!CheckStatus(api.SetSessionGraphOptimizationLevel(session_options_, ORT_ENABLE_BASIC)))
                    return false;
                fprintf(stderr, "[%s] Loading pre-optimized model from cache\n", log_tag_);
            } else {
                if (!CheckStatus(api.SetSessionGraphOptimizationLevel(session_options_, ORT_ENABLE_ALL)))
                    return false;
                if (!CreateSessionPath(api, opt_path, /*is_model=*/false))
                    return false;
            }
        } else {
            if (!CheckStatus(api.SetSessionGraphOptimizationLevel(session_options_, ORT_ENABLE_ALL)))
                return false;
        }

        // Hardware-accelerated EPs (best-effort, falls back to CPU)
        RegisterExecutionProviders(api);

        // Thread pool config based on active EP
        bool xnnpack_active = (std::strcmp(execution_provider_, "XNNPACK") == 0);
        int num_threads = xnnpack_active ? 1 : 4;
        if (!CheckStatus(api.SetIntraOpNumThreads(session_options_, num_threads)))
            return false;
        if (xnnpack_active) {
            api.AddSessionConfigEntry(session_options_,
                "session.intra_op.allow_spinning", "0");
        }

        // Create session
        if (!CreateSessionPath(api, effective_model, /*is_model=*/true))
            return false;

        ready_ = true;

        // CPU memory info — reused across all inference calls.
        // Uses OrtDeviceAllocator (malloc/free) instead of OrtArenaAllocator
        // to avoid steady RSS growth across 1500+ tracks.
        if (!CheckStatus(api.CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &memory_info_))) {
            ready_ = false;
            return false;
        }

        fprintf(stderr, "[%s] ORT model loaded (EP: %s)\n", log_tag_, execution_provider_);
        return true;
    }

    bool IsReady() const { return ready_; }
    const char* GetExecutionProvider() const { return execution_provider_; }
    const OrtApi* Api() const { return api_; }
    OrtApi const& ApiRef() const { return *api_; }
    ::OrtSession* Session() const { return session_; }
    OrtMemoryInfo* MemoryInfo() const { return memory_info_; }

    bool CheckStatus(OrtStatus* status) {
        if (status != nullptr) {
            const char* msg = api_->GetErrorMessage(status);
            fprintf(stderr, "[%s] ORT error: %s\n", log_tag_, msg);
            api_->ReleaseStatus(status);
            return false;
        }
        return true;
    }

private:
    static OrtEnv* GetSharedEnv(const OrtApi* api) {
        static OrtEnv* env = nullptr;
        static std::once_flag flag;
        std::call_once(flag, [api]() {
            auto status = api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "xune", &env);
            if (status) {
                const char* msg = api->GetErrorMessage(status);
                fprintf(stderr, "[xune] ORT env creation failed: %s\n", msg);
                api->ReleaseStatus(status);
                env = nullptr;
            }
        });
        return env;
    }

    void RegisterExecutionProviders(const OrtApi& api) {
#if ORT_API_VERSION >= 17
#ifdef _WIN32
        {
            const char* keys[] = {"device_id"};
            const char* vals[] = {"0"};
            auto status = api.SessionOptionsAppendExecutionProvider(
                session_options_, "DML", keys, vals, 1);
            if (status) {
                api.ReleaseStatus(status);
            } else {
                execution_provider_ = "DirectML";
            }
        }
#elif defined(__linux__)
    #if defined(__aarch64__)
        {
            const char* keys[] = {"intra_op_num_threads"};
            const char* vals[] = {"4"};
            auto status = api.SessionOptionsAppendExecutionProvider(
                session_options_, "XNNPACK", keys, vals, 1);
            if (status) {
                api.ReleaseStatus(status);
            } else {
                execution_provider_ = "XNNPACK";
            }
        }
    #endif
        {
            const char* keys[] = {"device_id"};
            const char* vals[] = {"0"};
            auto status = api.SessionOptionsAppendExecutionProvider(
                session_options_, "CUDA", keys, vals, 1);
            if (status) {
                api.ReleaseStatus(status);
            } else {
                execution_provider_ = "CUDA";
            }
        }
#endif
#endif
    }

    // Handles both SetOptimizedModelFilePath and CreateSession, with wchar_t on Windows
    bool CreateSessionPath(const OrtApi& api, const std::string& path, bool is_model) {
#ifdef _WIN32
        int wlen = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
        std::vector<wchar_t> wpath(wlen);
        MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wpath.data(), wlen);
        if (is_model) {
            return CheckStatus(api.CreateSession(
                GetSharedEnv(api_), wpath.data(), session_options_, &session_));
        } else {
            api.SetOptimizedModelFilePath(session_options_, wpath.data());
            return true;
        }
#else
        if (is_model) {
            return CheckStatus(api.CreateSession(
                GetSharedEnv(api_), path.c_str(), session_options_, &session_));
        } else {
            api.SetOptimizedModelFilePath(session_options_, path.c_str());
            return true;
        }
#endif
    }

    const OrtApi* api_;
    const char* log_tag_;
    ::OrtSession* session_ = nullptr;
    OrtSessionOptions* session_options_ = nullptr;
    OrtMemoryInfo* memory_info_ = nullptr;
    const char* execution_provider_ = "CPU";
    bool ready_ = false;
};

}  // namespace xune
