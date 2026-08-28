#include <gtest/gtest.h>

#include "vntx/config.hpp"

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    vntx::set_layer_config(vntx::LayerConfig{});
    return RUN_ALL_TESTS();
}
