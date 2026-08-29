// Replays every distinct texture shape The Witcher 3 Next-Gen created in ~/vntx_witcher.log
// through the layer, copying the whole mip chain with the application's original geometry.
// Non-square and non-power-of-two shapes are where per-axis clamping arithmetic goes wrong.
#include <vulkan/vulkan.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {
VkInstance g_instance = VK_NULL_HANDLE;
VkPhysicalDevice g_phys = VK_NULL_HANDLE;
VkDevice g_device = VK_NULL_HANDLE;
VkQueue g_queue = VK_NULL_HANDLE;
uint32_t g_qf = 0;
VkPhysicalDeviceMemoryProperties g_mem{};

struct Image { VkImage handle{}; VkDeviceMemory memory{}; VkDeviceSize allocated{}; };

int device_local_type(uint32_t bits) {
    for (uint32_t i = 0; i < g_mem.memoryTypeCount; ++i)
        if ((bits & (1u << i)) &&
            (g_mem.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
            return static_cast<int>(i);
    return -1;
}

bool make_image(uint32_t w, uint32_t h, uint32_t mips, VkImageUsageFlags usage, Image& out) {
    VkImageCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = VK_FORMAT_BC7_UNORM_BLOCK;
    info.extent = {w, h, 1};
    info.mipLevels = mips;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vkCreateImage(g_device, &info, nullptr, &out.handle) != VK_SUCCESS) return false;
    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(g_device, out.handle, &req);
    const int t = device_local_type(req.memoryTypeBits);
    if (t < 0) return false;
    VkMemoryAllocateInfo a{};
    a.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    a.allocationSize = req.size;
    a.memoryTypeIndex = static_cast<uint32_t>(t);
    if (vkAllocateMemory(g_device, &a, nullptr, &out.memory) != VK_SUCCESS) return false;
    out.allocated = req.size;
    return vkBindImageMemory(g_device, out.handle, out.memory, 0) == VK_SUCCESS;
}

void barrier(VkCommandBuffer cmd, VkImage img, VkImageLayout to, VkAccessFlags access) {
    VkImageMemoryBarrier b{};
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    b.newLayout = to;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = img;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, 1};
    b.dstAccessMask = access;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                         0, nullptr, 0, nullptr, 1, &b);
}

uint32_t full_mips(uint32_t w, uint32_t h) {
    uint32_t n = 1, d = (w > h) ? w : h;
    while (d > 1) { d >>= 1; ++n; }
    return n;
}
}  // namespace

int main(int argc, char** argv) {
    const std::string layer_name =
        "VK_LAYER_VNTX_repro_" + std::string((argc > 1) ? argv[1] : "new");
    const int start = (argc > 2) ? std::atoi(argv[2]) : 0;

    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.apiVersion = VK_API_VERSION_1_3;
    const char* const layers[] = {layer_name.c_str()};
    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    ici.enabledLayerCount = 1;
    ici.ppEnabledLayerNames = layers;
    if (vkCreateInstance(&ici, nullptr, &g_instance) != VK_SUCCESS) return 2;

    uint32_t n = 0;
    vkEnumeratePhysicalDevices(g_instance, &n, nullptr);
    std::vector<VkPhysicalDevice> devs(n);
    vkEnumeratePhysicalDevices(g_instance, &n, devs.data());
    for (auto d : devs) {
        VkPhysicalDeviceProperties p{};
        vkGetPhysicalDeviceProperties(d, &p);
        if (std::strstr(p.deviceName, "NVIDIA")) { g_phys = d; break; }
    }
    if (!g_phys) return 3;
    vkGetPhysicalDeviceMemoryProperties(g_phys, &g_mem);

    uint32_t qn = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(g_phys, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qfs(qn);
    vkGetPhysicalDeviceQueueFamilyProperties(g_phys, &qn, qfs.data());
    for (uint32_t i = 0; i < qn; ++i)
        if (qfs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) { g_qf = i; break; }

    const float prio = 1.0f;
    VkDeviceQueueCreateInfo dq{};
    dq.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    dq.queueFamilyIndex = g_qf; dq.queueCount = 1; dq.pQueuePriorities = &prio;
    VkDeviceCreateInfo dc{};
    dc.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dc.queueCreateInfoCount = 1; dc.pQueueCreateInfos = &dq;
    if (vkCreateDevice(g_phys, &dc, nullptr, &g_device) != VK_SUCCESS) return 2;
    vkGetDeviceQueue(g_device, g_qf, 0, &g_queue);

    struct Shape { uint32_t w, h; };
    const Shape shapes[] = {
        {128,128},{128,256},{128,512},{188,136},{256,128},{256,256},{256,512},{256,1024},
        {256,2048},{364,132},{400,1080},{512,128},{512,256},{512,512},{512,1024},{512,2048},
        {896,128},{968,704},{976,1012},{984,960},{988,396},{988,444},{988,448},{988,460},
        {992,780},{1008,508},{1020,1016},{1024,184},{1024,256},{1024,432},{1024,460},{1024,468},
        {1024,512},{1024,624},{1024,928},{1024,996},{1024,1004},{1024,1012},{1024,1024},
        {1024,2048},{1024,4096},{1268,560},{1376,316},{1404,688},{1420,620},{1548,752},
        {2048,1024},{2048,2048},{2048,4096},{2828,1024},
    };
    const int count = static_cast<int>(sizeof(shapes) / sizeof(shapes[0]));

    for (int i = start; i < count; ++i) {
        const auto s = shapes[i];
        const uint32_t mips = full_mips(s.w, s.h);

        Image src{}, dst{};
        if (!make_image(s.w, s.h, 1,
                        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, src) ||
            !make_image(s.w, s.h, mips,
                        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, dst)) {
            std::printf("  %4ux%-4u  SETUP FAILED\n", s.w, s.h);
            return 2;
        }

        VkCommandPoolCreateInfo pci{};
        pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pci.queueFamilyIndex = g_qf;
        VkCommandPool pool{};
        vkCreateCommandPool(g_device, &pci, nullptr, &pool);
        VkCommandBufferAllocateInfo cbai{};
        cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool = pool; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        VkCommandBuffer cmd{};
        vkAllocateCommandBuffers(g_device, &cbai, &cmd);
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &bi);
        barrier(cmd, src.handle, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_TRANSFER_READ_BIT);
        barrier(cmd, dst.handle, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT);

        std::vector<VkImageCopy> regions;
        for (uint32_t m = 0; m < mips; ++m) {
            VkImageCopy r{};
            r.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            r.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, m, 0, 1};
            r.extent = {(s.w >> m) ? (s.w >> m) : 1u, (s.h >> m) ? (s.h >> m) : 1u, 1};
            regions.push_back(r);
        }
        vkCmdCopyImage(cmd, src.handle, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst.handle,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       static_cast<uint32_t>(regions.size()), regions.data());
        vkEndCommandBuffer(cmd);

        VkFenceCreateInfo fci{};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VkFence fence{};
        vkCreateFence(g_device, &fci, nullptr, &fence);
        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
        const VkResult sub = vkQueueSubmit(g_queue, 1, &si, fence);
        const VkResult wait = (sub == VK_SUCCESS)
                                  ? vkWaitForFences(g_device, 1, &fence, VK_TRUE, 5000000000ull)
                                  : sub;
        const bool ok = (sub == VK_SUCCESS && wait == VK_SUCCESS);
        std::printf("  [%2d] %4ux%-4u mips=%2u alloc=%9llu  %s\n", i, s.w, s.h, mips,
                    static_cast<unsigned long long>(dst.allocated),
                    ok ? "ok" : "*** DEVICE LOST ***");
        std::fflush(stdout);
        if (!ok) return 1;

        vkDestroyFence(g_device, fence, nullptr);
        vkDestroyCommandPool(g_device, pool, nullptr);
        vkDestroyImage(g_device, dst.handle, nullptr); vkFreeMemory(g_device, dst.memory, nullptr);
        vkDestroyImage(g_device, src.handle, nullptr); vkFreeMemory(g_device, src.memory, nullptr);
    }
    std::printf("  all %d shapes completed\n", count - start);
    return 0;
}
