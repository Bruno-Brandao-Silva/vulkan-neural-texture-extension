// End-to-end reproduction of the VNTX GPU hang on real hardware.
//
// Mirrors what VKD3D-Proton emits for The Witcher 3: a full-size BC7 source image copied into a
// candidate texture that the layer created at half size. The copy carries the application's
// original 2048x2048 extents, so the transfer engine writes ~5.6MB into a ~1.4MB allocation --
// the ACCESS_TYPE_VIRT_WRITE MMU fault (Xid 31, ENGINE CE0) seen in the kernel log.
//
// Run against the pre-fix and post-fix layer and compare.

#include <vulkan/vulkan.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CHECK(expr)                                                              \
    do {                                                                         \
        const VkResult r_ = (expr);                                              \
        if (r_ != VK_SUCCESS) {                                                  \
            std::printf("  FATAL %s -> VkResult %d\n", #expr, static_cast<int>(r_)); \
            return 2;                                                            \
        }                                                                        \
    } while (0)

namespace {

VkInstance g_instance = VK_NULL_HANDLE;
VkPhysicalDevice g_phys = VK_NULL_HANDLE;
VkDevice g_device = VK_NULL_HANDLE;
VkQueue g_queue = VK_NULL_HANDLE;
uint32_t g_queue_family = 0;
VkPhysicalDeviceMemoryProperties g_mem_props{};

struct Image {
    VkImage handle{VK_NULL_HANDLE};
    VkDeviceMemory memory{VK_NULL_HANDLE};
    VkDeviceSize allocated{0};
};

int find_device_local_type(uint32_t bits) {
    for (uint32_t i = 0; i < g_mem_props.memoryTypeCount; ++i) {
        if ((bits & (1u << i)) &&
            (g_mem_props.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

// Creates an image exactly as the application asks. Whatever the layer does to it underneath is
// invisible here -- which is the whole point.
bool make_image(uint32_t dim, uint32_t mips, VkImageUsageFlags usage, Image& out) {
    VkImageCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = VK_FORMAT_BC7_UNORM_BLOCK;
    info.extent = {dim, dim, 1};
    info.mipLevels = mips;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    if (vkCreateImage(g_device, &info, nullptr, &out.handle) != VK_SUCCESS) {
        return false;
    }

    // The application sizes its allocation from what the layer reports, exactly as VKD3D does.
    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(g_device, out.handle, &req);
    const int type = find_device_local_type(req.memoryTypeBits);
    if (type < 0) {
        return false;
    }

    VkMemoryAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex = static_cast<uint32_t>(type);
    if (vkAllocateMemory(g_device, &alloc, nullptr, &out.memory) != VK_SUCCESS) {
        return false;
    }
    out.allocated = req.size;
    return vkBindImageMemory(g_device, out.handle, out.memory, 0) == VK_SUCCESS;
}

void barrier(VkCommandBuffer cmd, VkImage image, VkImageLayout to, VkAccessFlags dst_access) {
    VkImageMemoryBarrier b{};
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    b.newLayout = to;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, 1};
    b.dstAccessMask = dst_access;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                         0, nullptr, 0, nullptr, 1, &b);
}

}  // namespace

int main(int argc, char** argv) {
    const std::string label = (argc > 1) ? argv[1] : "layer";

    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "vntx-hang-repro";
    app.apiVersion = VK_API_VERSION_1_3;

    const std::string layer_name =
        "VK_LAYER_VNTX_repro_" + std::string((argc > 2) ? argv[2] : "new");
    const char* const layers[] = {layer_name.c_str()};
    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    ici.enabledLayerCount = 1;
    ici.ppEnabledLayerNames = layers;
    CHECK(vkCreateInstance(&ici, nullptr, &g_instance));

    uint32_t count = 0;
    vkEnumeratePhysicalDevices(g_instance, &count, nullptr);
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(g_instance, &count, devices.data());

    // The NVIDIA dGPU is the one that faulted, and it does not drive the display.
    std::string picked;
    for (const auto d : devices) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(d, &props);
        if (std::strstr(props.deviceName, "NVIDIA") || std::strstr(props.deviceName, "RTX")) {
            g_phys = d;
            picked = props.deviceName;
            break;
        }
    }
    if (g_phys == VK_NULL_HANDLE) {
        std::printf("  SKIP: no NVIDIA device present\n");
        return 3;
    }
    std::printf("[%s] device: %s\n", label.c_str(), picked.c_str());

    vkGetPhysicalDeviceMemoryProperties(g_phys, &g_mem_props);

    uint32_t qf_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(g_phys, &qf_count, nullptr);
    std::vector<VkQueueFamilyProperties> qfs(qf_count);
    vkGetPhysicalDeviceQueueFamilyProperties(g_phys, &qf_count, qfs.data());
    for (uint32_t i = 0; i < qf_count; ++i) {
        if (qfs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            g_queue_family = i;
            break;
        }
    }

    const float priority = 1.0f;
    VkDeviceQueueCreateInfo dqci{};
    dqci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    dqci.queueFamilyIndex = g_queue_family;
    dqci.queueCount = 1;
    dqci.pQueuePriorities = &priority;

    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &dqci;
    CHECK(vkCreateDevice(g_phys, &dci, nullptr, &g_device));
    vkGetDeviceQueue(g_device, g_queue_family, 0, &g_queue);

    // Source: single-mip, so no version of the layer downscales it. This is the full-size peer.
    Image src{};
    if (!make_image(2048, 1, VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, src)) {
        std::printf("  FATAL: source image setup failed\n");
        return 2;
    }

    // Each shape addresses the destination with geometry the application believes in, which the
    // pre-fix layer forwarded untouched. Selected one at a time so a GPU fault in one case cannot
    // mask the others.
    struct Case {
        const char* name;
        VkImageUsageFlags usage;
        uint32_t request_mips;   // what the application asks vkCreateImage for
        uint32_t copy_mip;       // which mip the copy addresses
        uint32_t copy_dim;       // extent at that mip, from the application's point of view
        bool whole_chain;        // copy every mip in one command
    };
    const Case cases[] = {
        {"mip0 full extent", VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, 12, 0,
         2048, false},
        {"mip0, TRANSFER_SRC usage",
         VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
             VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
         12, 0, 2048, false},
        {"tail mip beyond physical chain",
         VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, 12, 11, 4, false},
        {"mip 1 of a 2-mip texture",
         VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, 2, 1, 1024, false},
        {"whole mip chain in one command",
         VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, 12, 0, 2048, true},
    };
    constexpr int CASE_COUNT = static_cast<int>(sizeof(cases) / sizeof(cases[0]));

    const int selected = (argc > 3) ? std::atoi(argv[3]) : -1;
    int failures = 0;
    for (int ci = 0; ci < CASE_COUNT; ++ci) {
        if (selected >= 0 && ci != selected) {
            continue;
        }
        const auto& c = cases[ci];

        Image dst{};
        if (!make_image(2048, c.request_mips, c.usage, dst)) {
            std::printf("  FATAL: destination setup failed for %s\n", c.name);
            return 2;
        }
        std::printf("[%s] case %d %-34s dst alloc = %8llu bytes", label.c_str(), ci, c.name,
                    static_cast<unsigned long long>(dst.allocated));
        std::fflush(stdout);

        VkCommandPoolCreateInfo pci{};
        pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pci.queueFamilyIndex = g_queue_family;
        VkCommandPool pool = VK_NULL_HANDLE;
        CHECK(vkCreateCommandPool(g_device, &pci, nullptr, &pool));

        VkCommandBufferAllocateInfo cbai{};
        cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool = pool;
        cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        CHECK(vkAllocateCommandBuffers(g_device, &cbai, &cmd));

        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        CHECK(vkBeginCommandBuffer(cmd, &bi));
        barrier(cmd, src.handle, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_TRANSFER_READ_BIT);
        barrier(cmd, dst.handle, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_ACCESS_TRANSFER_WRITE_BIT);

        std::vector<VkImageCopy> regions;
        if (c.whole_chain) {
            for (uint32_t m = 0; m < c.request_mips; ++m) {
                const uint32_t d = (2048u >> m) ? (2048u >> m) : 1u;
                VkImageCopy r{};
                r.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                r.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, m, 0, 1};
                r.extent = {d, d, 1};
                regions.push_back(r);
            }
        } else {
            VkImageCopy r{};
            r.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            r.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, c.copy_mip, 0, 1};
            r.extent = {c.copy_dim, c.copy_dim, 1};
            regions.push_back(r);
        }
        vkCmdCopyImage(cmd, src.handle, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst.handle,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       static_cast<uint32_t>(regions.size()), regions.data());
        CHECK(vkEndCommandBuffer(cmd));

        VkFenceCreateInfo fci{};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VkFence fence = VK_NULL_HANDLE;
        CHECK(vkCreateFence(g_device, &fci, nullptr, &fence));

        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        const VkResult submit = vkQueueSubmit(g_queue, 1, &si, fence);
        const VkResult waited = (submit == VK_SUCCESS)
                                    ? vkWaitForFences(g_device, 1, &fence, VK_TRUE, 5000000000ull)
                                    : submit;

        const bool ok = (submit == VK_SUCCESS && waited == VK_SUCCESS);
        std::printf("  -> submit=%d wait=%d  %s\n", static_cast<int>(submit),
                    static_cast<int>(waited), ok ? "OK" : "*** GPU FAULT / DEVICE LOST ***");
        std::fflush(stdout);
        if (!ok) {
            ++failures;
            return 1;  // the device is gone
        }

        vkDestroyFence(g_device, fence, nullptr);
        vkDestroyCommandPool(g_device, pool, nullptr);
        vkDestroyImage(g_device, dst.handle, nullptr);
        vkFreeMemory(g_device, dst.memory, nullptr);
    }

    vkDestroyImage(g_device, src.handle, nullptr);
    vkFreeMemory(g_device, src.memory, nullptr);
    vkDestroyDevice(g_device, nullptr);
    vkDestroyInstance(g_instance, nullptr);
    return failures == 0 ? 0 : 1;
}
