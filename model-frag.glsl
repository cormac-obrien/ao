#version 330

in vec3 f_normal;
in vec2 f_texcoord;

layout (location = 0) out vec4 out_color;
layout (location = 1) out vec4 out_normal;

uniform bool diff;
uniform bool mask;
uniform sampler2D diff_tex;
uniform sampler2D mask_tex;

void main() {
    out_normal = vec4(normalize(f_normal * 0.5 + 0.5), 1.0f);

    if (mask && texture(mask_tex, f_texcoord).r < 0.5) {
        discard;
    } else if (diff) {
        out_color = texture(diff_tex, f_texcoord);
    } else {
        out_color = vec4(0.0f, 0.0f, 0.0f, 1.0f);
    }
};
