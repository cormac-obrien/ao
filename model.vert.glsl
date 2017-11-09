#version 330

layout (location = 0) in vec3 v_position;
layout (location = 1) in vec3 v_normal;
layout (location = 2) in vec2 v_texcoord;

out vec3 f_normal;
out vec2 f_texcoord;

uniform mat4 world_matrix;
uniform mat4 projection_matrix;
uniform mat3 normal_matrix;

void main() {
    f_normal = normal_matrix * v_normal;
    f_texcoord = v_texcoord;
    vec4 model_pos = vec4(v_position, 1.0f);
    vec4 world_pos = world_matrix * model_pos;
    gl_Position = projection_matrix * world_pos;
};
