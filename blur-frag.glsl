#version 330

#define BLUR_SIZE_1D 4

noperspective in vec2 f_texcoord;

out vec4 out_color;

uniform sampler2D blur_tex;
uniform sampler2D blur_ssao_tex;

void main() {
    vec2 texel_size = 1.0f / vec2(textureSize(blur_tex, 0));
    float result = 0.0f;
    vec2 hlim = vec2(float(-BLUR_SIZE_1D) * 0.5 + 0.5);

    for (int x = 0; x < BLUR_SIZE_1D; x++) {
        for (int y = 0; y < BLUR_SIZE_1D; y++) {
            vec2 offset = (hlim + vec2(float(x), float(y))) * texel_size;
            result += texture(blur_ssao_tex, f_texcoord + offset).r;
        }
    }

    float value = result / float(BLUR_SIZE_1D * BLUR_SIZE_1D);
    out_color = vec4(texture(blur_tex, f_texcoord).rgb * value, 1.0f);
}
