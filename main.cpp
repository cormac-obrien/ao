/* Copyright © 2017 Cormac O'Brien
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <algorithm>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <map>
#include <random>
#include <vector>

#include <GL/gl3w.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <SOIL/SOIL.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.hpp"

#define KERNEL_SIZE 32
#define NOISE_SIZE_1D 4
#define NOISE_SIZE (NOISE_SIZE_1D * NOISE_SIZE_1D)

#define LERP(a, b, f) ((a) * (1.0f - (f)) + (b) * (f))

#define INFO(M, ...)                                                           \
    fprintf(stderr, "[INFO] (%s:%d) " M "\n", __FILE__, __LINE__, ##__VA_ARGS__)

#define ERROR(M, ...)                                                          \
    fprintf(stderr, "[ERROR] (%s:%d) " M "\n", __FILE__, __LINE__,             \
            ##__VA_ARGS__)

#define ERROR_OPENGL(M, ...)                                                   \
    do {                                                                       \
        GLenum err;                                                            \
        size_t ecnt = 0;                                                       \
        while ((err = glGetError()) != GL_NO_ERROR) {                          \
            fprintf(stderr, "[OPENGL] (%s:%d) " M ": (%x)\n", __FILE__,        \
                    __LINE__, ##__VA_ARGS__, err);                             \
            ecnt += 1;                                                         \
        }                                                                      \
        if (ecnt > 0) {                                                        \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0);

typedef struct __attribute__((packed)) Vertex {
    float position[3];
    float normal[3];
    float texcoord[2];
} Vertex;

typedef struct DrawObject {
    GLuint vbo;
    size_t tri_count;
    size_t material_id;
} DrawObject;

typedef struct DiffuseTexture {
    GLuint width;
    GLuint height;
    GLubyte *data;
} DiffuseTexture;

typedef struct Texture {
    GLuint diff;
    GLuint mask;
    GLuint bump;
    GLuint spec;
} Texture;

const char *const vert_src = R"glsl(
#version 330

layout (location = 0) in vec3 v_position;
layout (location = 1) in vec3 v_normal;
layout (location = 2) in vec2 v_texcoord;

out vec3 f_normal;
out vec2 f_texcoord;

uniform mat4 world;
uniform mat4 persp;

void main() {
    f_texcoord = v_texcoord;
    vec4 model_pos = vec4(v_position, 1.0f);
    vec4 world_pos = world * model_pos;
    gl_Position = persp * world_pos;
};
)glsl";

const char *const frag_src = R"glsl(
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
    if (mask && texture(mask_tex, f_texcoord).r < 0.5) {
        discard;
    } else if (diff) {
        out_color = texture(diff_tex, f_texcoord);
    } else {
        out_color = vec4(0.0f, 0.0f, 0.0f, 1.0f);
    }

    out_normal = vec4(f_normal, 1.0f);
};
)glsl";

const char *const fullscreen_vert_src = R"glsl(
#version 330

layout (location = 0) in vec2 v_position;
layout (location = 1) in vec2 v_texcoord;

out vec2 f_texcoord;

void main() {
    f_texcoord = v_texcoord;
    gl_Position = vec4(v_position.xy, 0.0f, 1.0f);
}
)glsl";

const char *const fullscreen_frag_src = R"glsl(
#version 330

in vec2 f_texcoord;

out vec4 color;

uniform sampler2D tex;

void main() {
    color = texture(tex, f_texcoord);
}
)glsl";

/*
 * Vertex data for rendering a texture to fullscreen, stored as [x y | s t]
 */
const GLfloat fullscreen_vertex[6][4] = {
    {-1.0f, -1.0f, 0.0f, 0.0f}, // bottom left
    {1.0f, 1.0f, 1.0f, 1.0f},   // top right
    {-1.0f, 1.0f, 0.0f, 1.0f},  // top left
    {-1.0f, -1.0f, 0.0f, 0.0f}, // bottom left
    {1.0f, -1.0f, 1.0f, 0.0f},  // bottom right
    {1.0f, 1.0f, 1.0f, 1.0f}    // top right
};

void APIENTRY opengl_error_callback(GLenum source, GLenum type, GLuint id,
                                    GLenum severity, GLsizei length,
                                    const GLchar *msg, const void *user_param) {
    if (severity != GL_DEBUG_SEVERITY_NOTIFICATION) {
        printf("%s\n", msg);
    }
}

GLuint new_shader(const GLenum type, const char *src) {
    GLuint shader = glCreateShader(type);
    const GLchar **srcaddr = (const GLchar **)&src;
    glShaderSource(shader, 1, srcaddr, NULL);
    glCompileShader(shader);

    GLint shader_stat;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &shader_stat);
    if (shader_stat == GL_FALSE) {
        GLint loglen;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &loglen);

        GLchar *log = (GLchar *)calloc(loglen + 1, sizeof(*log));
        glGetShaderInfoLog(shader, loglen, NULL, log);
        fprintf(stderr, "Failed to compile shader:\n%s\n", log);
        free(log);
        shader = 0;
    }

    return shader;
}

GLuint new_program(size_t shader_count, const GLuint *shaders) {
    GLuint program = glCreateProgram();
    ERROR_OPENGL("Couldn't create a new shader program.");

    for (size_t i = 0; i < shader_count; i++) {
        glAttachShader(program, shaders[i]);
    }

    glLinkProgram(program);

    INFO("Linked shaders.");

    GLint prog_stat;
    glGetProgramiv(program, GL_LINK_STATUS, &prog_stat);
    if (prog_stat == GL_FALSE) {
        GLint loglen;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &loglen);

        GLchar *log = (GLchar *)calloc(loglen + 1, sizeof(*log));
        glGetProgramInfoLog(program, loglen, NULL, log);
        fprintf(stderr, "Failed to link program:\n%s\n", log);
        free(log);
    }

    return program;
}

/*
 * Load all textures specified by the MTL data given in `materials` and store
 * them in `textures`
 */
bool create_textures(std::vector<Texture> &textures,
                     std::vector<tinyobj::material_t> &materials) {
    /*
     * Query anisotropic filtering
     */
    float aniso = 1.0f;
    glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY, &aniso);

    for (size_t i = 0; i < materials.size(); i++) {
        Texture tex = Texture{.diff = 0, .mask = 0, .bump = 0, .spec = 0};

        if (!materials[i].diffuse_texname.empty()) {
            std::string path = "crytek-sponza/" + materials[i].diffuse_texname;
            std::replace(path.begin(), path.end(), '\\', '/');

            GLuint tex_id = SOIL_load_OGL_texture(
                path.c_str(), 0, 0,
                SOIL_FLAG_MIPMAPS | SOIL_FLAG_TEXTURE_REPEATS |
                    SOIL_FLAG_INVERT_Y);

            if (tex_id == 0) {
                ERROR("Failed to load diffuse texture from %s", path.c_str());
                return false;
            }

            /*
             * Enable anisotropic filtering
             */
            glBindTexture(GL_TEXTURE_2D, tex_id);
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY, aniso);

            tex.diff = tex_id;
        }

        if (!materials[i].alpha_texname.empty()) {
            std::string path = "crytek-sponza/" + materials[i].alpha_texname;
            std::replace(path.begin(), path.end(), '\\', '/');

            GLuint tex_id = SOIL_load_OGL_texture(
                path.c_str(), 0, 0,
                SOIL_FLAG_MIPMAPS | SOIL_FLAG_TEXTURE_REPEATS |
                    SOIL_FLAG_INVERT_Y);

            if (tex_id == 0) {
                ERROR("Failed to load mask texture from %s", path.c_str());
                return false;
            }

            tex.mask = tex_id;
        }

        if (!materials[i].bump_texname.empty()) {
            std::string path = "crytek-sponza/" + materials[i].bump_texname;
            std::replace(path.begin(), path.end(), '\\', '/');

            GLuint tex_id = SOIL_load_OGL_texture(
                path.c_str(), 0, 0,
                SOIL_FLAG_MIPMAPS | SOIL_FLAG_TEXTURE_REPEATS |
                    SOIL_FLAG_INVERT_Y);

            if (tex_id == 0) {
                ERROR("Failed to load bump texture from %s", path.c_str());
                return false;
            }

            tex.bump = tex_id;
        }

        if (!materials[i].specular_texname.empty()) {
            std::string path = "crytek-sponza/" + materials[i].specular_texname;
            std::replace(path.begin(), path.end(), '\\', '/');

            GLuint tex_id = SOIL_load_OGL_texture(
                path.c_str(), 0, 0,
                SOIL_FLAG_MIPMAPS | SOIL_FLAG_TEXTURE_REPEATS |
                    SOIL_FLAG_INVERT_Y);

            if (tex_id == 0) {
                ERROR("Failed to load specular texture from %s", path.c_str());
                return false;
            }

            tex.spec = tex_id;
        }

        textures.push_back(tex);
    }

    return true;
}

bool create_draw_objects(std::vector<DrawObject> &draw_objects,
                         tinyobj::attrib_t &attrib,
                         std::vector<tinyobj::shape_t> &shapes,
                         std::vector<tinyobj::material_t> &materials) {
    for (size_t s = 0; s < shapes.size(); s++) {
        /*
         * Shapes can have multiple materials, so generate one DrawObject and
         * vertex buffer per material in each object
         */
        std::vector<DrawObject> face_objects;
        std::vector<std::vector<Vertex>> vertex_data;

        for (size_t face_id = 0; face_id < shapes[s].mesh.indices.size() / 3;
             face_id++) {
            Vertex verts[3];
            tinyobj::index_t idx[3];
            for (size_t i = 0; i < 3; i++) {
                idx[i] = shapes[s].mesh.indices[3 * face_id + i];
            }

            int mat_id = shapes[s].mesh.material_ids[face_id];
            if (mat_id < 0 || mat_id > static_cast<int>(materials.size())) {
                ERROR("Bad material ID for shape \"%s\"",
                      shapes[s].name.c_str());
                return false;
            }

            /*
             * Sequence position data
             */
            for (size_t pos_id = 0; pos_id < 3; pos_id++) {
                for (size_t component = 0; component < 3; component++) {
                    verts[pos_id].position[component] =
                        attrib
                            .vertices[3 * idx[pos_id].vertex_index + component];
                }
            }

            /*
             * Sequence surface normals
             */
            if (attrib.normals.size() > 0) {
                for (size_t normal_id = 0; normal_id < 3; normal_id++) {
                    for (size_t component = 0; component < 3; component++) {
                        verts[normal_id].normal[component] =
                            attrib.vertices[3 * idx[normal_id].vertex_index +
                                            component];
                    }
                }
            } else {
                ERROR("OBJ file does not provide normals and normal "
                      "calculation is not implemented.");
                return false;
            }

            /*
             * Sequence texture coordinates
             */
            if (attrib.texcoords.size() > 0) {
                for (size_t i = 0; i < 3; i++) {
                    verts[i].texcoord[0] =
                        attrib.texcoords[2 * idx[i].texcoord_index];
                    verts[i].texcoord[1] =
                        attrib.texcoords[2 * idx[i].texcoord_index + 1];
                }
            }

            /*
             * Append vertex data
             */
            std::vector<DrawObject>::iterator it =
                std::find_if(face_objects.begin(), face_objects.end(),
                             [mat_id](DrawObject o) -> bool {
                                 return mat_id == (int)o.material_id;
                             });

            if (it == face_objects.end()) {
                // INFO("No object for this material, appending");

                DrawObject o = DrawObject();
                o.material_id = mat_id;
                face_objects.push_back(o);

                std::vector<Vertex> vdata;
                for (size_t i = 0; i < 3; i++) {
                    vdata.push_back(verts[i]);
                }
                vertex_data.push_back(vdata);
            } else {
                size_t obj_index = it - face_objects.begin();
                // INFO("Object index = %zu", obj_index);

                for (size_t i = 0; i < 3; i++) {
                    vertex_data[obj_index].push_back(verts[i]);
                }
            }
        }

        for (size_t o = 0; o < face_objects.size(); o++) {
            if (vertex_data.size() > 0) {
                glGenBuffers(1, &face_objects[o].vbo);
                glBindBuffer(GL_ARRAY_BUFFER, face_objects[o].vbo);
                glBufferData(GL_ARRAY_BUFFER,
                             vertex_data[o].size() * sizeof(Vertex),
                             &vertex_data[o].at(0), GL_STATIC_DRAW);
                face_objects[o].tri_count = vertex_data[o].size() / 3;
            }
        }

        draw_objects.insert(draw_objects.end(), face_objects.begin(),
                            face_objects.end());

        /*
         * Data has been pushed to the GPU, make sure it's deallocated
         */
        face_objects.clear();
        face_objects.shrink_to_fit();
        vertex_data.clear();
        vertex_data.shrink_to_fit();
    }

    return true;
}

int main(int argc, char *argv[]) {
    /*
     * GLFW setup
     */
    glfwInit();
    glfwWindowHint(GLFW_SAMPLES, 16);
    GLFWwindow *window =
        glfwCreateWindow(1920, 1080, "objview", glfwGetPrimaryMonitor(), NULL);
    glfwMakeContextCurrent(window);

    /*
     * OpenGL context setup
     */
    if (gl3wInit() != 0) {
        ERROR("Failed to initialize OpenGL context");
    }

    if (!gl3wIsSupported(4, 3)) {
        INFO("OpenGL debug output unavailable (OpenGL 4.3 not supported).");
    } else {
        INFO("Enabling OpenGL debug output.");
        glEnable(GL_DEBUG_OUTPUT);
        glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
        glDebugMessageCallback(opengl_error_callback, NULL);
    }

    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glFrontFace(GL_CCW);

    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glDepthFunc(GL_LEQUAL);
    glDepthRange(0.0f, 1.0f);

    glEnable(GL_MULTISAMPLE);

    /*
     * Load OBJ file
     */
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string err_string;

    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &err_string,
                                "crytek-sponza/sponza.obj", "crytek-sponza/");
    if (!err_string.empty()) {
        INFO("OBJ loader log:\n%s", err_string.c_str());
    }

    if (!ret) {
        ERROR("Error while loading OBJ file.");
        exit(EXIT_FAILURE);
    }

    /*
     * Add default material
     */
    materials.push_back(tinyobj::material_t());

    std::vector<Texture> textures;
    if (create_textures(textures, materials) == false) {
        ERROR("Failed to create textures");
        exit(EXIT_FAILURE);
    }

    /*
     * Reorganize geometry data for uploading to GPU
     */
    std::vector<DrawObject> draw_objects;
    if (create_draw_objects(draw_objects, attrib, shapes, materials) == false) {
        ERROR("Failed to create draw objects");
        exit(EXIT_FAILURE);
    }

    GLuint vert = new_shader(GL_VERTEX_SHADER, vert_src);
    GLuint frag = new_shader(GL_FRAGMENT_SHADER, frag_src);
    GLuint shaders[2] = {vert, frag};
    GLuint prog = new_program(2, shaders);
    GLuint fullscreen_vert = new_shader(GL_VERTEX_SHADER, fullscreen_vert_src);
    GLuint fullscreen_frag =
        new_shader(GL_FRAGMENT_SHADER, fullscreen_frag_src);
    GLuint fullscreen_shaders[2] = {fullscreen_vert, fullscreen_frag};
    GLuint fullscreen_prog = new_program(2, fullscreen_shaders);
    ERROR_OPENGL("Shader compilation and linking failed.");

    glUseProgram(prog);
    GLuint world_unif = glGetUniformLocation(prog, "world");
    GLuint persp_unif = glGetUniformLocation(prog, "persp");
    GLuint diff_unif = glGetUniformLocation(prog, "diff");
    GLuint mask_unif = glGetUniformLocation(prog, "mask");
    GLuint diff_tex_unif = glGetUniformLocation(prog, "diff_tex");
    GLuint mask_tex_unif = glGetUniformLocation(prog, "mask_tex");

    glUniform1i(diff_tex_unif, 0);
    glUniform1i(mask_tex_unif, 1);

    GLuint geometry_vao;
    glGenVertexArrays(1, &geometry_vao);

    /*
     *
     */
    GLuint fullscreen_vao, fullscreen_vbo;
    glGenVertexArrays(1, &fullscreen_vao);
    glGenBuffers(1, &fullscreen_vbo);
    ERROR_OPENGL("Error generating fullscreen vertex array");
    glBindVertexArray(fullscreen_vao);
    glBindBuffer(GL_ARRAY_BUFFER, fullscreen_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(fullscreen_vertex), fullscreen_vertex,
                 GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat),
                          reinterpret_cast<void *>(0));
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat),
                          reinterpret_cast<void *>(2 * sizeof(GLfloat)));
    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);

    /*
     * Set up for SSAO rendering pass. We generate three images: one with
     * diffuse textures, one with surface normals and one with depth values.
     */

    GLuint ssao_tex;
    glActiveTexture(GL_TEXTURE0);
    glGenTextures(1, &ssao_tex);
    glBindTexture(GL_TEXTURE_2D, ssao_tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1920, 1080, 0, GL_RGB,
                 GL_UNSIGNED_BYTE, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);
    ERROR_OPENGL("Error generating SSAO texture");

    GLuint ssao_normal_tex;
    glActiveTexture(GL_TEXTURE1);
    glGenTextures(1, &ssao_normal_tex);
    glBindTexture(GL_TEXTURE_2D, ssao_normal_tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1920, 1080, 0, GL_RGB,
                 GL_UNSIGNED_BYTE, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);
    ERROR_OPENGL("Error generating SSAO normal texture");

    GLuint ssao_depth_tex;
    glActiveTexture(GL_TEXTURE2);
    glGenTextures(1, &ssao_depth_tex);
    glBindTexture(GL_TEXTURE_2D, ssao_depth_tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, 1920, 1080, 0,
                 GL_DEPTH_COMPONENT, GL_FLOAT, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);
    ERROR_OPENGL("Error generating SSAO depth texture");

    GLuint ssao_fbo;
    glGenFramebuffers(1, &ssao_fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, ssao_fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                           ssao_tex, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D,
                           ssao_normal_tex, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D,
                           ssao_depth_tex, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        ERROR_OPENGL("SSAO framebuffer object not complete");
        exit(EXIT_FAILURE);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    std::mt19937 gen;
    std::uniform_real_distribution<GLfloat> zero_one_dist(0.0f, 1.0f);
    std::uniform_real_distribution<GLfloat> minus_one_one_dist(-1.0f, 1.0f);

    GLfloat kernel[3 * KERNEL_SIZE];
    for (size_t i = 0; i < KERNEL_SIZE; i++) {
        /*
         * Generate points at random across the surface of a hemisphere on the
         * z-axis with radius r = 1. Initially generate x and y on the range
         * [-1, 1] and z on the range [0, 1], then normalize the vector
         */
        float x = minus_one_one_dist(gen);
        float y = minus_one_one_dist(gen);
        float z = zero_one_dist(gen);
        float mag = sqrtf(x * x + y * y + z * z);

        kernel[3 * i] = x / mag;
        kernel[3 * i + 1] = y / mag;
        kernel[3 * i + 2] = z / mag;

        /*
         * Scatter the points inside the hemisphere using an attenuation
         * function to place more samples closer to the origin
         */
        float scale = (float)i / (float)KERNEL_SIZE;
        scale = LERP(0.1f, 1.0f, scale);
        for (size_t c = 0; c < 3; c++) {
            kernel[3 * i + c] *= scale;
        }
    }

    GLfloat noise[3 * NOISE_SIZE];
    for (size_t i = 0; i < NOISE_SIZE; i++) {
        /*
         * Generate a small noise texture used for randomly rotating the sample
         * points about the z-axis
         */
        float x = minus_one_one_dist(gen);
        float y = minus_one_one_dist(gen);
        float z = 0.0;
        float mag = sqrtf(x * x + y * y + z * z);

        noise[3 * i] = x / mag;
        noise[3 * i + 1] = y / mag;
        noise[3 * i + 2] = z / mag;
    }

    float pos[3] = {0.0f};
    float angle[3] = {0.0f};

    while (!glfwWindowShouldClose(window)) {
        /*
         * Input handling
         */
        glfwPollEvents();

        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
            pos[0] -= cosf(angle[1]);
            pos[2] -= sinf(angle[1]);
        }
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
            pos[0] += cosf(angle[1]);
            pos[2] += sinf(angle[1]);
        }
        if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
            pos[1] -= 1.0f;
        }
        if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS) {
            pos[1] += 1.0f;
        }
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
            pos[0] -= sinf(angle[1]);
            pos[2] += cosf(angle[1]);
        }
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
            pos[0] += sinf(angle[1]);
            pos[2] -= cosf(angle[1]);
        }

        if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
            angle[0] -= 0.01f;
        }
        if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
            angle[0] += 0.01f;
        }
        if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) {
            angle[1] -= 0.01f;
        }
        if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
            angle[1] += 0.01f;
        }

        /*
         *
         */
        glUseProgram(prog);

        /*
         * Calculate world transform matrix based on current position and angle
         */
        glm::mat4 world;
        world = glm::scale(world, glm::vec3(0.1, 0.1, 0.1));
        world = glm::rotate(world, angle[0], glm::vec3(1.0f, 0.0, 0.0f));
        world = glm::rotate(world, angle[1], glm::vec3(0.0f, 1.0, 0.0f));
        world = glm::translate(world, glm::vec3(pos[0], pos[1], pos[2]));
        glUniformMatrix4fv(world_unif, 1, GL_FALSE, &world[0][0]);

        /*
         * Calculate perspective matrix based on current aspect ratio
         */
        int framebuffer_width, framebuffer_height;
        glfwGetFramebufferSize(window, &framebuffer_width, &framebuffer_height);

        float fovy = 0.5f * (float)M_PI;
        float aspect = (float)framebuffer_width / (float)framebuffer_height;
        float z_near = 1.0f;
        float z_far = 1024.0f;
        glm::mat4 perspective = glm::perspective(fovy, aspect, z_near, z_far);
        glUniformMatrix4fv(persp_unif, 1, GL_FALSE, &perspective[0][0]);

        /*
         * Initial rendering pass:
         * - diffuse textures with masking
         * - surface normals
         * - depth buffer
         */
        GLenum draw_buffers[] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};

        glBindFramebuffer(GL_FRAMEBUFFER, ssao_fbo);
        glDrawBuffers(2, draw_buffers);

        /*
         * Clear framebuffer
         */
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClearDepth(1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glBindVertexArray(geometry_vao);
        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);
        glEnableVertexAttribArray(2);
        for (size_t i = 0; i < draw_objects.size(); i++) {
            glBindBuffer(GL_ARRAY_BUFFER, draw_objects[i].vbo);

            /*
             * TODO: only call these once per VBO
             */
            glVertexAttribPointer(
                0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                reinterpret_cast<void *>(offsetof(Vertex, position)));
            glVertexAttribPointer(
                1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                reinterpret_cast<void *>(offsetof(Vertex, normal)));
            glVertexAttribPointer(
                2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                reinterpret_cast<void *>(offsetof(Vertex, texcoord)));

            Texture tex = textures[draw_objects[i].material_id];

            glActiveTexture(GL_TEXTURE0);
            glUniform1i(diff_unif, tex.diff == 0 ? GL_FALSE : GL_TRUE);
            glBindTexture(GL_TEXTURE_2D, tex.diff);

            glActiveTexture(GL_TEXTURE1);
            glUniform1i(mask_unif, tex.mask == 0 ? GL_FALSE : GL_TRUE);
            glBindTexture(GL_TEXTURE_2D, tex.mask);

            glDrawArrays(GL_TRIANGLES, 0, draw_objects[i].tri_count * 3);
            ERROR_OPENGL("Error in initial pass (draw object %zu)", i);
        }
        glDisableVertexAttribArray(0);
        glDisableVertexAttribArray(1);
        glDisableVertexAttribArray(2);

        /*
         * Final pass: render texture to screen
         */
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        ERROR_OPENGL("error binding default framebuffer");
        glUseProgram(fullscreen_prog);

        glBindVertexArray(fullscreen_vao);
        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);
        glBindBuffer(GL_ARRAY_BUFFER, fullscreen_vbo);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, ssao_tex);

        glDrawArrays(GL_TRIANGLES, 0, 6);
        glDisableVertexAttribArray(0);
        glDisableVertexAttribArray(1);

        ERROR_OPENGL("checking for opengl error");
        glfwSwapBuffers(window);
    }

    glfwDestroyWindow(window);
    glfwTerminate();

    exit(EXIT_SUCCESS);
}
