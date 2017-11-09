#version 330

layout (location = 0) in vec2 v_position;
layout (location = 1) in vec2 v_texcoord;

out vec2 f_texcoord;

void main() {
    f_texcoord = v_texcoord;
    gl_Position = vec4(v_position.xy, 1.0f, 1.0f);
}
