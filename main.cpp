/* Copyright © 2016 Cormac O'Brien
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

/*
 * TODO:
 * - per-face texturing
 */

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <map>
#include <vector>

#include <GL/gl3w.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.hpp"

#define INFO(M, ...) fprintf(stderr, "[INFO] (%s:%d) " M "\n", __FILE__, \
                             __LINE__, ##__VA_ARGS__)

#define ERROR(M, ...) fprintf(stderr, "[ERROR] (%s:%d) " M "\n", __FILE__, \
                              __LINE__, ##__VA_ARGS__)

#define ERROR_OPENGL(M, ...) do {                                       \
        GLenum err;                                                     \
        size_t ecnt = 0;                                                \
        while ((err = glGetError()) != GL_NO_ERROR) {                   \
            fprintf(stderr, "[OPENGL] (%s:%d) " M ": (%x)\n", __FILE__, \
                    __LINE__, err, ##__VA_ARGS__);                      \
            ecnt += 1;                                                  \
        }                                                               \
        if (ecnt > 0) {                                                 \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0);

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

const char * const vert_src =
    "#version 330\n"

    "layout (location = 0) in vec3 v_position;\n"
    "layout (location = 1) in vec3 v_normal;\n"
    "layout (location = 2) in vec3 v_color;\n"
    "layout (location = 3) in vec2 v_texcoord;\n"

    "out vec3 f_color;\n"
    "out vec2 f_texcoord;\n"

    "uniform mat4 world;\n"
    "uniform mat4 persp;\n"

    "void main()\n"
    "{\n"
    "    f_color = v_color;\n"
    "    f_texcoord = v_texcoord;\n"
    "    vec4 model_pos = vec4(v_position, 1.0f);\n"
    "    vec4 world_pos = world * model_pos;\n"
    "    gl_Position = persp * world_pos;\n"
    "}\n";

const char * const frag_src =
    "#version 330\n"

    "in vec3 f_color;\n"
    "in vec2 f_texcoord;\n"

    "out vec4 color;\n"

    "uniform bool textured;\n"
    "uniform sampler2D tex;\n"

    "void main()\n"
    "{\n"
    "    if (textured) {\n"
    "        color = texture(tex, f_texcoord);\n"
    "    } else {\n"
    "        color = vec4(f_color, 1.0f);\n"
    "    }\n"
    "}\n";

bool load_tga(DiffuseTexture *tex, std::string filename) {
    std::ifstream tgafile(filename.c_str(), std::ios::in | std::ios::binary);
    if (!tgafile.is_open()) {
        ERROR("Couldn't open \"%s\"", filename.c_str());
        return false;
    }

    struct __attribute__((packed)) {
        uint8_t id_len;
        uint8_t colormap_type;
        uint8_t image_type;

        struct __attribute__((packed)) {
            uint16_t first;
            uint16_t len;
            uint8_t bpp;
        } colormap_spec;

        struct __attribute__((packed)) {
            uint16_t x_orig;
            uint16_t y_orig;
            uint16_t width;
            uint16_t height;
            uint8_t bpp;
            uint8_t desc;
        } image_spec;
    } header;

    tgafile.read((char *)&header, sizeof header);


    if (header.image_type != 2) {
        ERROR("Only truecolor images supported");
        return false;
    }

    if (header.colormap_type != 0) {
        ERROR("Colormap support not implemented");
        return false;
    }

    uint8_t *image_id = nullptr;
    if (header.id_len > 0) {
        image_id = new uint8_t[header.id_len];
        tgafile.read((char *)&image_id, sizeof image_id);
    }

    if (header.image_spec.bpp != 24 && header.image_spec.bpp != 32) {
        ERROR("Only 24bpp and 32bpp images supported (%" PRIu8 "bpp specified)", header.image_spec.bpp);
        return false;
    }

    uint16_t w = header.image_spec.width;
    uint16_t h = header.image_spec.height;
    uint8_t bpp = header.image_spec.bpp;
    INFO("%" PRIu16 "x%" PRIu16 "x%" PRIu8 "", w, h, bpp);

    uint8_t *tga_data = new uint8_t[w * h * bpp / 8];
    if (tga_data == nullptr) {
        ERROR("TGA data allocation failed");
        return false;
    }

    tgafile.read((char *)tga_data, w * h * bpp / 8);

    GLubyte *rgba = new GLubyte[w * h * 4];
    if (rgba == nullptr) {
        ERROR("RGBA data allocation failed");
        return false;
    }

    for (size_t i = 0; i < w * h; i++) {
        if (bpp == 24) {
            rgba[4 * i]     = tga_data[3 * i + 2];
            rgba[4 * i + 1] = tga_data[3 * i + 1];
            rgba[4 * i + 2] = tga_data[3 * i];
            rgba[4 * i + 3] = 255;
        } else if (bpp == 32) {
            rgba[4 * i]     = tga_data[4 * i + 2];
            rgba[4 * i + 1] = tga_data[4 * i + 1];
            rgba[4 * i + 2] = tga_data[4 * i];
            rgba[4 * i + 3] = tga_data[4 * i + 3];
        } else {
            ERROR("Bad bpp value");
            exit(EXIT_FAILURE);
        }
    }

    delete[] tga_data;
    delete[] image_id;
    tgafile.close();

    tex->width = w;
    tex->height = h;
    tex->data = rgba;

    return true;
}

GLuint new_shader(const GLenum type, const char *src)
{
    GLuint shader = glCreateShader(type);
    const GLchar **srcaddr = (const GLchar **)&src;
    glShaderSource(shader, 1, srcaddr, NULL);
    glCompileShader(shader);

    GLint shader_stat;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &shader_stat);
    if (shader_stat == GL_FALSE) {
        GLint loglen;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &loglen);

        GLchar *log = (GLchar *)calloc(loglen + 1, sizeof (*log));
        glGetShaderInfoLog(shader, loglen, NULL, log);
        fprintf(stderr, "Failed to compile shader:\n%s\n", log);
        free(log);
        shader = 0;
    }

    return shader;
}

GLuint new_program(size_t shader_count, const GLuint *shaders)
{
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

        GLchar *log = (GLchar *)calloc(loglen + 1, sizeof (*log));
        glGetProgramInfoLog(program, loglen, NULL, log);
        fprintf(stderr, "Failed to link program:\n%s\n", log);
        free(log);
    }

    return program;
}

int main(int argc, char *argv[]) {
    glfwInit();
    glfwWindowHint(GLFW_SAMPLES, 16);
    GLFWwindow *window = glfwCreateWindow(1920, 1080, "objview", glfwGetPrimaryMonitor(), NULL);
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

    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &err_string, "crytek-sponza/sponza.obj", "crytek-sponza/");
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
    for (size_t i = 0; i < materials.size(); i++) {
        if (!materials[i].diffuse_texname.empty()) {
            std::replace(materials[i].diffuse_texname.begin(), materials[i].diffuse_texname.end(), '\\', '/');
            printf("materials[%zu].diffuse_texname = %s\n", i, materials[i].diffuse_texname.c_str());
            DiffuseTexture tex;
            bool status = load_tga(&tex, "crytek-sponza/" + materials[i].diffuse_texname);
            if (!status) {
                ERROR("Failed to load texture");
                exit(EXIT_FAILURE);
            }

            GLuint tex_id;
            glGenTextures(1, &tex_id);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, tex_id);
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, tex.width, tex.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, tex.data);
            texture_ids.push_back(tex_id);
            INFO("texture %u: %s", tex_id, materials[i].diffuse_texname.c_str());
            delete[] tex.data;
        } else {
            printf("materials[%zu].diffuse_texname: no texture\n", i);
            texture_ids.push_back(0);
        }
    }

    /*
     * Reorganize geometry data for uploading to GPU
     */
    std::vector<DrawObject> draw_objects;
    for (size_t s = 0; s < shapes.size(); s++) {
        DrawObject o;

        /*
         * Vertex data is organized as [x y z | r θ ϕ | r g b | u v]
         */
        std::vector<float> vertex_data;

        for (size_t face_id = 0; face_id < shapes[s].mesh.indices.size() / 3; face_id++) {
            tinyobj::index_t idx[3];
            for (size_t i = 0; i < 3; i++) {
                idx[i] = shapes[s].mesh.indices[3 * face_id + i];
            }

            int mat_id = shapes[s].mesh.material_ids[face_id];
            if (mat_id < 0 || mat_id > static_cast<int>(materials.size())) {
                ERROR("Bad material ID for shape \"%s\"", shapes[s].name.c_str());
            }

            /*
             * Sequence position data
             */
            float positions[3][3] = {{0.0}};
            for (size_t pos_id = 0; pos_id < 3; pos_id++) {
                for (size_t component = 0; component < 3; component++) {
                    positions[pos_id][component] = attrib.vertices[3 * idx[pos_id].vertex_index + component];
                }
            }

            /*
             * Sequence surface normals
             */
            float normals[3][3];
            if (attrib.normals.size() > 0) {
                for (size_t normal_id = 0; normal_id < 3; normal_id++) {
                    for(size_t component = 0; component < 3; component++) {
                        normals[normal_id][component] = attrib.normals[3 * idx[normal_id].normal_index + component];
                    }
                }
            } else {
                ERROR("OBJ file does not provide normals and normal calculation is not implemented.");
            }

            /*
             * Sequence texture coordinates
             */
            float texcoord[3][2] = {{0.0}};
            if (attrib.texcoords.size() > 0) {
                for (size_t i = 0; i < 3; i++) {
                    texcoord[i][0] = attrib.texcoords[2 * idx[i].texcoord_index];
                    texcoord[i][1] = 1.0 - attrib.texcoords[2 * idx[i].texcoord_index + 1];
                }
            }

            float diffuse[3] = {0.0};
            for (size_t i = 0; i < 3; i++) {
                diffuse[i] = materials[mat_id].diffuse[i];
            }

            float normal_factor = 0.2;
            float diffuse_factor = 0.8;

            float color[3][3];
            for (size_t color_id = 0; color_id < 3; color_id++) {
                float len_squared = 0.0;
                for (size_t component = 0; component < 3; component++) {
                    color[color_id][component] = normals[color_id][component] * normal_factor
                                               + diffuse[component] * diffuse_factor;
                    len_squared += color[color_id][component] * color[color_id][component];
                }

                /*
                 * Normalize vertex colors
                 */
                if (len_squared > 0.0) {
                    float len = sqrtf(len_squared);

                    for (size_t component = 0; component < 3; component++) {
                        color[color_id][component] /= len;
                    }
                }
            }

            /*
             * Append vertex data
             */
            for (size_t i = 0; i < 3; i++) {
                for (size_t component = 0; component < 3; component++) {
                    vertex_data.push_back(positions[i][component]);
                }

                for (size_t component = 0; component < 3; component++) {
                    vertex_data.push_back(normals[i][component]);
                }

                for (size_t component = 0; component < 3; component++) {
                    vertex_data.push_back(color[i][component]);
                }

                for (size_t component = 0; component < 2; component++) {
                    vertex_data.push_back(texcoord[i][component]);
                }
            }

            if (shapes[s].mesh.material_ids.size() > 0) {
                o.material_id = shapes[s].mesh.material_ids[0];
            } else {
                o.material_id = materials.size() - 1;
            }
        }

        if (vertex_data.size() > 0) {
            glGenBuffers(1, &o.vbo);
            glBindBuffer(GL_ARRAY_BUFFER, o.vbo);
            glBufferData(GL_ARRAY_BUFFER, vertex_data.size() * sizeof (float), &vertex_data.at(0), GL_STATIC_DRAW);
        }

        o.tri_count = vertex_data.size() / (3 + 3 + 3 + 2) / 3;
        draw_objects.push_back(o);

        /*
         * Data has been pushed to the GPU, make sure it's deallocated
         */
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

    /*
     * 0 position (3 floats, 12 bytes)
     * 1 normal   (3 floats, 12 bytes)
     * 2 color    (3 floats, 12 bytes)
     * 3 texcoord (2 floats,  8 bytes)
     *
     * normal   + color    + texcoord = 32 bytes
     * color    + texcoord + position = 32 bytes
     * texcoord + position + normal   = 32 bytes
     * position + normal   + color    = 36 bytes
     */

    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);
    glEnableVertexAttribArray(3);

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
            90.0f,
            (float)framebuffer_width / (float)framebuffer_height,
            1.0f,
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

            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 44, reinterpret_cast<void *>(0));
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 44, reinterpret_cast<void *>(12));
            glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 44, reinterpret_cast<void *>(24));
            glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, 44, reinterpret_cast<void *>(36));

            GLuint tex_id = texture_ids[draw_objects[i].material_id];
            if (tex_id != 0) {
                glUniform1i(textured_unif, GL_TRUE);
            } else {
                glUniform1i(textured_unif, GL_FALSE);
            }
            glBindTexture(GL_TEXTURE_2D, texture_ids[draw_objects[i].material_id]);

            glDrawArrays(GL_TRIANGLES, 0, draw_objects[i].tri_count * 3);
        }

        ERROR_OPENGL("checking for opengl error");
        glfwSwapBuffers(window);
    }

    glfwDestroyWindow(window);
    glfwTerminate();

    exit(EXIT_SUCCESS);
}
