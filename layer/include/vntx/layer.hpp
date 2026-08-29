#pragma once

#include <vulkan/vk_layer.h>
#include <vulkan/vulkan.h>

#include <atomic>
#include <memory>
#include <shared_mutex>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

#include "vntx/logging.hpp"

namespace vntx {

/// @brief Metadata and VRAM sizing state for a candidate Neural Texture.
struct CandidateTextureInfo {
    VkExtent3D extent{0, 0, 0};
    VkFormat format{VK_FORMAT_UNDEFINED};
    uint32_t mip_levels{1};
    uint32_t array_layers{1};
    VkImageUsageFlags usage{0};
    uint64_t native_size_bytes{0};  ///< Exact native BC uncompressed size across all mips & layers
    uint64_t ntc_size_bytes{0};     ///< Compact NTC size (64-byte header + MLP weights)
    uint64_t downsized_memory_size{0};  ///< Driver-aligned memory requirement size in VRAM
    VkDeviceMemory bound_memory{VK_NULL_HANDLE};  ///< Bound VkDeviceMemory handle
    VkDeviceSize bound_offset{0};                 ///< Bound memory offset
    uint64_t texture_hash{0};                     ///< xxHash3 64-bit identifier
    VkDeviceSize alignment{0};                    ///< Driver required memory alignment
    uint32_t memory_type_bits{0};                 ///< Driver supported memory type bits
    uint32_t scale_factor{
        1};  ///< Physical dimension compression scale factor (e.g. 2 = 75% VRAM saved)
    VkExtent3D created_extent{0, 0, 0};  ///< Actual physical image extent created on GPU
    bool is_bound{false};                ///< True after vkBindImageMemory[2]
    bool fallback_triggered{false};      ///< True if runtime validation failed and image reverted to pass-through
};

/// @brief Thread-safe instance-wide and device-wide telemetry counters for VRAM reduction
/// statistics.
struct SessionTelemetry {
    std::atomic<uint64_t> total_candidate_textures{0};
    std::atomic<uint64_t> total_native_vram_bytes{0};
    std::atomic<uint64_t> total_compressed_vram_bytes{0};
    std::atomic<uint64_t> total_vram_saved_bytes{0};

    SessionTelemetry() = default;
    SessionTelemetry(const SessionTelemetry& other) noexcept
        : total_candidate_textures(other.total_candidate_textures.load(std::memory_order_relaxed)),
          total_native_vram_bytes(other.total_native_vram_bytes.load(std::memory_order_relaxed)),
          total_compressed_vram_bytes(
              other.total_compressed_vram_bytes.load(std::memory_order_relaxed)),
          total_vram_saved_bytes(other.total_vram_saved_bytes.load(std::memory_order_relaxed)) {}

    SessionTelemetry& operator=(const SessionTelemetry& other) noexcept {
        if (this != &other) {
            total_candidate_textures.store(
                other.total_candidate_textures.load(std::memory_order_relaxed),
                std::memory_order_relaxed);
            total_native_vram_bytes.store(
                other.total_native_vram_bytes.load(std::memory_order_relaxed),
                std::memory_order_relaxed);
            total_compressed_vram_bytes.store(
                other.total_compressed_vram_bytes.load(std::memory_order_relaxed),
                std::memory_order_relaxed);
            total_vram_saved_bytes.store(
                other.total_vram_saved_bytes.load(std::memory_order_relaxed),
                std::memory_order_relaxed);
        }
        return *this;
    }

    void record_candidate(const uint64_t native_bytes, const uint64_t compressed_bytes) noexcept {
        total_candidate_textures.fetch_add(1, std::memory_order_relaxed);
        total_native_vram_bytes.fetch_add(native_bytes, std::memory_order_relaxed);
        total_compressed_vram_bytes.fetch_add(compressed_bytes, std::memory_order_relaxed);
        if (native_bytes > compressed_bytes) {
            total_vram_saved_bytes.fetch_add(native_bytes - compressed_bytes,
                                             std::memory_order_relaxed);
        }
    }

    [[nodiscard]] double get_compression_ratio() const noexcept {
        const uint64_t comp = total_compressed_vram_bytes.load(std::memory_order_relaxed);
        if (comp == 0) return 1.0;
        return static_cast<double>(total_native_vram_bytes.load(std::memory_order_relaxed)) /
               static_cast<double>(comp);
    }

    [[nodiscard]] double get_savings_percentage() const noexcept {
        const uint64_t native = total_native_vram_bytes.load(std::memory_order_relaxed);
        if (native == 0) return 0.0;
        const uint64_t saved = total_vram_saved_bytes.load(std::memory_order_relaxed);
        return (static_cast<double>(saved) / static_cast<double>(native)) * 100.0;
    }

    void log_summary(const std::string_view context_name = "Session") const noexcept {
        const uint64_t count = total_candidate_textures.load(std::memory_order_relaxed);
        if (count == 0) {
            VNTX_LOG_INFO("{} VRAM telemetry: 0 candidate textures processed", context_name);
            return;
        }
        const uint64_t native_bytes = total_native_vram_bytes.load(std::memory_order_relaxed);
        const uint64_t comp_bytes = total_compressed_vram_bytes.load(std::memory_order_relaxed);
        const uint64_t saved_bytes = total_vram_saved_bytes.load(std::memory_order_relaxed);

        const double native_mb = static_cast<double>(native_bytes) / (1024.0 * 1024.0);
        const double comp_mb = static_cast<double>(comp_bytes) / (1024.0 * 1024.0);
        const double saved_mb = static_cast<double>(saved_bytes) / (1024.0 * 1024.0);
        const double ratio =
            (comp_bytes > 0) ? (static_cast<double>(native_bytes) / static_cast<double>(comp_bytes))
                             : 1.0;
        const double saved_pct =
            (native_bytes > 0)
                ? ((static_cast<double>(saved_bytes) / static_cast<double>(native_bytes)) * 100.0)
                : 0.0;

        VNTX_LOG_INFO(
            "{} VRAM telemetry: {} candidate textures, native VRAM={:.2f}MB, compressed "
            "VRAM={:.2f}MB, saved VRAM={:.2f}MB ({:.2f}x ratio, {:.1f}% saved)",
            context_name, count, native_mb, comp_mb, saved_mb, ratio, saved_pct);
    }
};

/// @brief Instance-level state and downstream dispatch functions.
struct InstanceData {
    VkInstance instance{VK_NULL_HANDLE};
    PFN_vkGetInstanceProcAddr next_get_instance_proc_addr{nullptr};
    PFN_vkCreateDevice next_create_device{nullptr};
    PFN_vkDestroyInstance next_destroy_instance{nullptr};
};

/// @brief Device-level state, candidate tracking, and downstream dispatch functions.
struct DeviceData {
    VkDevice device{VK_NULL_HANDLE};
    VkPhysicalDevice physical_device{VK_NULL_HANDLE};
    PFN_vkGetDeviceProcAddr next_get_device_proc_addr{nullptr};
    PFN_vkDestroyDevice next_destroy_device{nullptr};
    PFN_vkCreateImage next_create_image{nullptr};
    PFN_vkDestroyImage next_destroy_image{nullptr};
    PFN_vkGetImageMemoryRequirements next_get_image_memory_requirements{nullptr};
    PFN_vkGetImageMemoryRequirements2 next_get_image_memory_requirements2{nullptr};
    PFN_vkBindImageMemory next_bind_image_memory{nullptr};
    PFN_vkBindImageMemory2 next_bind_image_memory2{nullptr};
    PFN_vkCmdCopyBufferToImage next_cmd_copy_buffer_to_image{nullptr};
    PFN_vkCmdCopyBufferToImage2 next_cmd_copy_buffer_to_image2{nullptr};
    PFN_vkCmdPipelineBarrier next_cmd_pipeline_barrier{nullptr};
    PFN_vkCmdPipelineBarrier2 next_cmd_pipeline_barrier2{nullptr};
    PFN_vkCreateImageView next_create_image_view{nullptr};
    PFN_vkDestroyImageView next_destroy_image_view{nullptr};
    PFN_vkCreateShaderModule next_create_shader_module{nullptr};
    PFN_vkDestroyShaderModule next_destroy_shader_module{nullptr};

    std::shared_mutex image_mutex;
    std::unordered_map<VkImage, CandidateTextureInfo> candidate_textures;
    std::unordered_set<VkImage> candidate_images;
    std::unordered_set<VkImage> active_ntc_images;
    SessionTelemetry session_telemetry;
};

/// @brief Registry for instance and device contexts with graceful fallback state.
class LayerContext {
public:
    static LayerContext& get() noexcept;

    [[nodiscard]] bool is_disabled() const noexcept;
    void disable() noexcept;

    void register_instance(VkInstance instance, std::unique_ptr<InstanceData> data);
    void unregister_instance(VkInstance instance);
    [[nodiscard]] InstanceData* get_instance_data(VkInstance instance) const;

    void register_device(VkDevice device, std::unique_ptr<DeviceData> data);
    void unregister_device(VkDevice device);
    [[nodiscard]] DeviceData* get_device_data(VkDevice device) const;
    [[nodiscard]] DeviceData* get_device_data_from_command_buffer(
        VkCommandBuffer commandBuffer) const;

    [[nodiscard]] SessionTelemetry& get_telemetry() noexcept { return telemetry_; }
    [[nodiscard]] const SessionTelemetry& get_telemetry() const noexcept { return telemetry_; }

private:
    LayerContext() = default;
    ~LayerContext() = default;
    LayerContext(const LayerContext&) = delete;
    LayerContext& operator=(const LayerContext&) = delete;

    std::atomic<bool> disabled_{false};

    mutable std::shared_mutex instance_map_mutex_;
    std::unordered_map<void*, std::unique_ptr<InstanceData>> instance_map_;

    mutable std::shared_mutex device_map_mutex_;
    std::unordered_map<void*, std::unique_ptr<DeviceData>> device_map_;

    SessionTelemetry telemetry_;
};

// Dispatch key helpers (Khronos Layer Dispatch Architecture)
inline void* get_dispatch_key(const void* handle) noexcept {
    if (!handle) return nullptr;
    return const_cast<void*>(*reinterpret_cast<void* const*>(handle));
}

}  // namespace vntx

// C-ABI layer entrypoints
extern "C" {

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vntx_GetInstanceProcAddr(VkInstance instance,
                                                                  const char* pName);

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vntx_GetDeviceProcAddr(VkDevice device, const char* pName);

VKAPI_ATTR VkResult VKAPI_CALL
vntx_NegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface* pVersionStruct);

VKAPI_ATTR VkResult VKAPI_CALL
vntx_EnumerateInstanceLayerProperties(uint32_t* pPropertyCount, VkLayerProperties* pProperties);

VKAPI_ATTR VkResult VKAPI_CALL vntx_EnumerateInstanceExtensionProperties(
    const char* pLayerName, uint32_t* pPropertyCount, VkExtensionProperties* pProperties);

// Core Vulkan function hooks
VKAPI_ATTR VkResult VKAPI_CALL vntx_CreateInstance(const VkInstanceCreateInfo* pCreateInfo,
                                                   const VkAllocationCallbacks* pAllocator,
                                                   VkInstance* pInstance);

VKAPI_ATTR void VKAPI_CALL vntx_DestroyInstance(VkInstance instance,
                                                const VkAllocationCallbacks* pAllocator);

VKAPI_ATTR VkResult VKAPI_CALL vntx_CreateDevice(VkPhysicalDevice physicalDevice,
                                                 const VkDeviceCreateInfo* pCreateInfo,
                                                 const VkAllocationCallbacks* pAllocator,
                                                 VkDevice* pDevice);

VKAPI_ATTR void VKAPI_CALL vntx_DestroyDevice(VkDevice device,
                                              const VkAllocationCallbacks* pAllocator);

VKAPI_ATTR VkResult VKAPI_CALL vntx_CreateImage(VkDevice device,
                                                const VkImageCreateInfo* pCreateInfo,
                                                const VkAllocationCallbacks* pAllocator,
                                                VkImage* pImage);

VKAPI_ATTR void VKAPI_CALL vntx_DestroyImage(VkDevice device, VkImage image,
                                             const VkAllocationCallbacks* pAllocator);

VKAPI_ATTR void VKAPI_CALL vntx_GetImageMemoryRequirements(
    VkDevice device, VkImage image, VkMemoryRequirements* pMemoryRequirements);

VKAPI_ATTR void VKAPI_CALL
vntx_GetImageMemoryRequirements2(VkDevice device, const VkImageMemoryRequirementsInfo2* pInfo,
                                 VkMemoryRequirements2* pMemoryRequirements);

VKAPI_ATTR VkResult VKAPI_CALL vntx_BindImageMemory(VkDevice device, VkImage image,
                                                    VkDeviceMemory memory,
                                                    VkDeviceSize memoryOffset);

VKAPI_ATTR VkResult VKAPI_CALL vntx_BindImageMemory2(VkDevice device, uint32_t bindInfoCount,
                                                     const VkBindImageMemoryInfo* pBindInfos);

VKAPI_ATTR void VKAPI_CALL vntx_CmdCopyBufferToImage(VkCommandBuffer commandBuffer,
                                                     VkBuffer srcBuffer, VkImage dstImage,
                                                     VkImageLayout dstImageLayout,
                                                     uint32_t regionCount,
                                                     const VkBufferImageCopy* pRegions);

VKAPI_ATTR void VKAPI_CALL vntx_CmdCopyBufferToImage2(
    VkCommandBuffer commandBuffer, const VkCopyBufferToImageInfo2* pCopyBufferToImageInfo);

VKAPI_ATTR void VKAPI_CALL vntx_CmdPipelineBarrier(
    VkCommandBuffer commandBuffer, VkPipelineStageFlags srcStageMask,
    VkPipelineStageFlags dstStageMask, VkDependencyFlags dependencyFlags,
    uint32_t memoryBarrierCount, const VkMemoryBarrier* pMemoryBarriers,
    uint32_t bufferMemoryBarrierCount, const VkBufferMemoryBarrier* pBufferMemoryBarriers,
    uint32_t imageMemoryBarrierCount, const VkImageMemoryBarrier* pImageMemoryBarriers);

VKAPI_ATTR void VKAPI_CALL vntx_CmdPipelineBarrier2(VkCommandBuffer commandBuffer,
                                                    const VkDependencyInfo* pDependencyInfo);

VKAPI_ATTR VkResult VKAPI_CALL vntx_CreateImageView(VkDevice device,
                                                    const VkImageViewCreateInfo* pCreateInfo,
                                                    const VkAllocationCallbacks* pAllocator,
                                                    VkImageView* pView);

VKAPI_ATTR void VKAPI_CALL vntx_DestroyImageView(VkDevice device, VkImageView imageView,
                                                 const VkAllocationCallbacks* pAllocator);

VKAPI_ATTR VkResult VKAPI_CALL vntx_CreateShaderModule(VkDevice device,
                                                       const VkShaderModuleCreateInfo* pCreateInfo,
                                                       const VkAllocationCallbacks* pAllocator,
                                                       VkShaderModule* pShaderModule);

VKAPI_ATTR void VKAPI_CALL vntx_DestroyShaderModule(VkDevice device, VkShaderModule shaderModule,
                                                    const VkAllocationCallbacks* pAllocator);

}  // extern "C"
