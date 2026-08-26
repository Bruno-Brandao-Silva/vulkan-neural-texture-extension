#version 450

layout(set = 0, binding = 0) readonly buffer NtcWeights {
    float raw_weights[];
} weights_data;

layout(location = 0) in vec2 in_uv;
layout(location = 0) out vec4 out_color;

vec4 evaluate_neural_texture(vec2 uv) {
    float h1[64];
    for (int i = 0; i < 64; ++i) {
        float w0 = weights_data.raw_weights[i * 2 + 0];
        float w1 = weights_data.raw_weights[i * 2 + 1];
        float b  = weights_data.raw_weights[128 + i];
        h1[i] = max(0.0, uv.x * w0 + uv.y * w1 + b);
    }

    float h2[64];
    int l2_weight_offset = 128 + 64;
    int l2_bias_offset   = l2_weight_offset + (64 * 64);
    for (int i = 0; i < 64; ++i) {
        float sum = weights_data.raw_weights[l2_bias_offset + i];
        for (int j = 0; j < 64; ++j) {
            sum += h1[j] * weights_data.raw_weights[l2_weight_offset + (j * 64 + i)];
        }
        h2[i] = max(0.0, sum);
    }

    int l3_weight_offset = l2_bias_offset + 64;
    int l3_bias_offset   = l3_weight_offset + (64 * 4);
    vec4 result = vec4(0.0);
    for (int c = 0; c < 4; ++c) {
        float sum = weights_data.raw_weights[l3_bias_offset + c];
        for (int j = 0; j < 64; ++j) {
            sum += h2[j] * weights_data.raw_weights[l3_weight_offset + (j * 4 + c)];
        }
        result[c] = 1.0 / (1.0 + exp(-sum));
    }

    return result;
}

void main() {
    out_color = evaluate_neural_texture(in_uv);
}
