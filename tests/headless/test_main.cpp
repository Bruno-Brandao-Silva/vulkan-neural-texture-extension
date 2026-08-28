#include <gtest/gtest.h>

#include "vntx/config.hpp"

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    vntx::LayerConfig cfg{};
    cfg.downsize_vram_allocations = false;
    vntx::set_layer_config(cfg);
    return RUN_ALL_TESTS();
}
