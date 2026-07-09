#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define GGML_VK_NAME "Vulkan"
#define GGML_VK_MAX_DEVICES 16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_vk_init(size_t dev_num);

enum ggml_vk_ssd_expert_kind {
    GGML_VK_SSD_EXPERT_GATE = 0,
    GGML_VK_SSD_EXPERT_UP   = 1,
    GGML_VK_SSD_EXPERT_DOWN = 2,
};

typedef bool (*ggml_vk_ssd_expert_read_t)(
        void * user_data,
        int    layer,
        int    expert,
        int    kind,
        void * dst,
        size_t n_bytes);

struct ggml_vk_ssd_expert_provider {
    void * user_data;
    ggml_vk_ssd_expert_read_t read;
};

GGML_BACKEND_API void ggml_backend_vk_set_ssd_expert_provider(
        ggml_backend_t backend,
        const struct ggml_vk_ssd_expert_provider * provider);

GGML_BACKEND_API bool ggml_backend_is_vk(ggml_backend_t backend);
GGML_BACKEND_API int  ggml_backend_vk_get_device_count(void);
GGML_BACKEND_API void ggml_backend_vk_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_vk_get_device_memory(int device, size_t * free, size_t * total);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_vk_buffer_type(size_t dev_num);
// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_vk_host_buffer_type(void);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_vk_reg(void);

#ifdef  __cplusplus
}
#endif
