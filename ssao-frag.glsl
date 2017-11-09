#version 330

#define KERNEL_SIZE 32
#define Z_DELTA_MIN 0.0001f
#define Z_DELTA_MAX 0.005f
#define SAMPLE_RADIUS 50.0f

in vec2 f_texcoord;

out vec4 color;

uniform sampler2D ssao_tex;
uniform sampler2D ssao_normal_tex;
uniform sampler2D ssao_depth_tex;
uniform vec3[KERNEL_SIZE] ssao_kernel;
uniform sampler2D ssao_noise_tex;
uniform vec2 ssao_noise_scale;
uniform mat4 ssao_projection_matrix;
uniform mat4 ssao_inverse_projection_matrix;

vec3 get_view_position(vec2 texcoord) {
    // scale-bias texcoords from [0, 1] to [-1, 1] to retrieve NDC x and y
    float x = texcoord.s * 2.0f - 1.0f;
    float y = texcoord.t * 2.0f - 1.0f;

    // pull NDC z out of the depth buffer
    float z = texture(ssao_depth_tex, texcoord).r;

    vec4 ndc_position = vec4(x, y, z, 1.0f);

    // calculate this fragment's position in view space
    vec4 view_position = ssao_inverse_projection_matrix * ndc_position;
    return view_position.xyz / view_position.w;
}

void main() {
    vec3 view_position = get_view_position(f_texcoord);
    vec3 view_normal = normalize(texture(ssao_normal_tex, f_texcoord).xyz * 2.0 - 1.0);
    vec3 rotation = normalize(texture(ssao_noise_tex, f_texcoord * ssao_noise_scale).xyz * 2.0 - 1.0);
    vec3 view_tangent = normalize(rotation - view_normal * dot(rotation, view_normal));
    vec3 view_bitangent = cross(view_normal, view_tangent);
    mat3 kernel_matrix = mat3(view_tangent, view_bitangent, view_normal);
    float occlusion = 0.0f;

    for (int i = 0; i < KERNEL_SIZE; i++) {
        vec3 view_sample = view_position + SAMPLE_RADIUS * (kernel_matrix * ssao_kernel[i]);
        vec4 ndc_sample = ssao_projection_matrix * vec4(view_sample, 1.0f);
        ndc_sample.xy /= ndc_sample.w;

        // scale-bias back from [-1, 1] to [0, 1]
        vec2 sample_texcoord = ndc_sample.xy * 0.5 + 0.5;
        float ndc_depth = texture(ssao_depth_tex, sample_texcoord).r;

        float delta_z = (ndc_sample.z / ndc_sample.w - ndc_depth) / ndc_depth;
        if (delta_z > 0.00001f && delta_z < 0.00025f) {
            occlusion += 1.0f;
        }
    }

    occlusion /= float(KERNEL_SIZE) - 1.0f;
    occlusion = 1.0f - occlusion;
    color = vec4(occlusion, occlusion, occlusion, 1.0f); // occlusion only
    // color = texture(ssao_tex, f_texcoord) * vec4(occlusion, occlusion, occlusion, 1.0f); // occlusion + color
}
