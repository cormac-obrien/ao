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
#include <vector>

#include <GL/gl3w.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <SOIL/SOIL.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.hpp"

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
                    __LINE__, err, ##__VA_ARGS__);                             \
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

out vec4 color;

uniform bool textured;
uniform sampler2D tex;

void main() {
    if (textured) {
        color = texture(tex, f_texcoord);
    } else {
        color = vec4(0.0f, 0.0f, 0.0f, 1.0f);
    }
};
)glsl";

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

int main(int argc, char *argv[]) {
    glfwInit();
    glfwWindowHint(GLFW_SAMPLES, 16);
    GLFWwindow *window =
        glfwCreateWindow(1920, 1080, "objview", glfwGetPrimaryMonitor(), NULL);
    glfwMakeContextCurrent(window);

    if (gl3wInit() != 0) {
        ERROR("Failed to initialize OpenGL context");
    }

    if (!gl3wIsSupported(4, 3)) {
        INFO("OpenGL debug output unavailable (OpenGL 4.3 not supported).");
    } else {
        INFO("Enabling OpenGL debug output.");
        glEnable(GL_DEBUG_OUTPUT);
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
    std::vector<GLuint> texture_ids;
    std::string err_string;

    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &err_string,
                                "crytek-sponza/sponza.obj", "crytek-sponza/");
    if (!err_string.empty()) {
        INFO("Possible error while loading OBJ file:\n%s", err_string.c_str());
    }

    if (!ret) {
        ERROR("Error while loading OBJ file.");
        exit(EXIT_FAILURE);
    }

    /*
     * Add default material
     */
    materials.push_back(tinyobj::material_t());

    /*
     * Load diffuse texture files
     */
    float aniso = 1.0f;
    glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY, &aniso);

    std::vector<Texture> textures;
    for (size_t i = 0; i < materials.size(); i++) {
        Texture tex;

        if (!materials[i].diffuse_texname.empty()) {
            std::string path = "crytek-sponza/" + materials[i].diffuse_texname;
            std::replace(path.begin(), path.end(), '\\', '/');

            GLuint tex_id = SOIL_load_OGL_texture(
                path.c_str(), 0, 0,
                SOIL_FLAG_MIPMAPS | SOIL_FLAG_TEXTURE_REPEATS |
                    SOIL_FLAG_INVERT_Y);

            if (tex_id == 0) {
                ERROR("Failed to load diffuse texture from %s", path.c_str());
                exit(EXIT_FAILURE);
            }

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
            }

            tex.spec = tex_id;
        }

        textures.push_back(tex);
    }

    /*
     * Reorganize geometry data for uploading to GPU
     */
    std::vector<DrawObject> draw_objects;
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

    GLuint vert = new_shader(GL_VERTEX_SHADER, vert_src);
    GLuint frag = new_shader(GL_FRAGMENT_SHADER, frag_src);
    GLuint shaders[2] = {vert, frag};
    GLuint prog = new_program(2, shaders);
    ERROR_OPENGL("Shader compilation and linking failed.");
    glUseProgram(prog);

    GLuint world_unif = glGetUniformLocation(prog, "world");
    GLuint persp_unif = glGetUniformLocation(prog, "persp");
    GLuint textured_unif = glGetUniformLocation(prog, "textured");

    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);

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

        glm::mat4 perspective = glm::perspective(
            90.0f, (float)framebuffer_width / (float)framebuffer_height, 1.0f,
            1024.0f);
        glUniformMatrix4fv(persp_unif, 1, GL_FALSE, &perspective[0][0]);

        /*
         * Clear framebuffer
         */
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClearDepth(1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        for (size_t i = 0; i < draw_objects.size(); i++) {
            glBindBuffer(GL_ARRAY_BUFFER, draw_objects[i].vbo);

            glVertexAttribPointer(
                0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                reinterpret_cast<void *>(offsetof(Vertex, position)));
            glVertexAttribPointer(
                1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                reinterpret_cast<void *>(offsetof(Vertex, normal)));
            glVertexAttribPointer(
                2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                reinterpret_cast<void *>(offsetof(Vertex, texcoord)));

            GLuint tex_id = textures[draw_objects[i].material_id].diff;
            if (tex_id != 0) {
                glUniform1i(textured_unif, GL_TRUE);
            } else {
                glUniform1i(textured_unif, GL_FALSE);
            }
            glBindTexture(GL_TEXTURE_2D, tex_id);

            glDrawArrays(GL_TRIANGLES, 0, draw_objects[i].tri_count * 3);
        }

        ERROR_OPENGL("checking for opengl error");
        glfwSwapBuffers(window);
    }

    glfwDestroyWindow(window);
    glfwTerminate();

    exit(EXIT_SUCCESS);
}
