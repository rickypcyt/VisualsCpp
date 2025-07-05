#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "audio_capture.h"
#include "waveform.h"
#include <imgui.h>
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <vector>
#include <thread>
#include <atomic>
#include <string>
#include <memory>
#include <algorithm>
#include <cstdio>
#include <cmath>

// OpenGL helpers
GLuint create_vbo(size_t max_points) {
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, max_points * 2 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    return vbo;
}
void update_vbo(GLuint vbo, const std::vector<float>& samples, int width, int height, float y_offset) {
    std::vector<float> verts(samples.size() * 2);
    for (size_t i = 0; i < samples.size(); ++i) {
        float x = (float)i / (samples.size() - 1) * width;
        float y = (samples[i] * 0.5f + 0.5f) * (height - 2 * y_offset) + y_offset;
        verts[2 * i] = x;
        verts[2 * i + 1] = y;
    }
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0, verts.size() * sizeof(float), verts.data());
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}
void draw_waveform(GLuint vbo, size_t n_points) {
    glLineWidth(3.0f); // Línea más gruesa
    glEnableClientState(GL_VERTEX_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(2, GL_FLOAT, 0, 0);
    glColor3f(1.0f, 0.8f, 0.2f); // Amarillo brillante
    glDrawArrays(GL_LINE_STRIP, 0, n_points);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDisableClientState(GL_VERTEX_ARRAY);
    glLineWidth(1.0f); // Restaurar grosor
}

// Dibuja una esfera wireframe deformada por el audio
void draw_wire_sphere_audio(float base_radius, int stacks, int slices, const std::vector<float>& samples, float intensity) {
    size_t N = samples.size();
    for (int i = 0; i <= stacks; ++i) {
        float lat0 = M_PI * (-0.5f + (float)(i - 1) / stacks);
        float z0  = sin(lat0);
        float zr0 =  cos(lat0);

        float lat1 = M_PI * (-0.5f + (float)i / stacks);
        float z1 = sin(lat1);
        float zr1 = cos(lat1);

        glBegin(GL_LINE_LOOP);
        for (int j = 0; j <= slices; ++j) {
            float lng = 2 * M_PI * (float)(j - 1) / slices;
            float x = cos(lng);
            float y = sin(lng);
            // Mapeo: un sample por vértice (wrap si hay más vértices que samples)
            size_t idx = ((i * slices + j) * N) / ((stacks+1) * (slices+1));
            float deform = (N > 0) ? samples[idx] * intensity : 0.0f;
            float r0 = base_radius + deform;
            float r1 = base_radius + deform;
            glVertex3f(r0 * x * zr0, r0 * y * zr0, r0 * z0);
            glVertex3f(r1 * x * zr1, r1 * y * zr1, r1 * z1);
        }
        glEnd();
    }
}

void myPerspective(float fovy, float aspect, float zNear, float zFar) {
    float f = 1.0f / tanf(fovy * 0.5f * M_PI / 180.0f);
    float m[16] = {0};
    m[0] = f / aspect;
    m[5] = f;
    m[10] = (zFar + zNear) / (zNear - zFar);
    m[11] = -1.0f;
    m[14] = (2.0f * zFar * zNear) / (zNear - zFar);
    glMultMatrixf(m);
}

void myLookAt(float eyeX, float eyeY, float eyeZ,
              float centerX, float centerY, float centerZ,
              float upX, float upY, float upZ) {
    float f[3] = {centerX - eyeX, centerY - eyeY, centerZ - eyeZ};
    float fn = sqrtf(f[0]*f[0] + f[1]*f[1] + f[2]*f[2]);
    f[0] /= fn; f[1] /= fn; f[2] /= fn;
    float up[3] = {upX, upY, upZ};
    float upn = sqrtf(up[0]*up[0] + up[1]*up[1] + up[2]*up[2]);
    up[0] /= upn; up[1] /= upn; up[2] /= upn;
    float s[3] = {f[1]*up[2] - f[2]*up[1], f[2]*up[0] - f[0]*up[2], f[0]*up[1] - f[1]*up[0]};
    float sn = sqrtf(s[0]*s[0] + s[1]*s[1] + s[2]*s[2]);
    s[0] /= sn; s[1] /= sn; s[2] /= sn;
    float u[3] = {s[1]*f[2] - s[2]*f[1], s[2]*f[0] - s[0]*f[2], s[0]*f[1] - s[1]*f[0]};
    float m[16] = {
        s[0], u[0], -f[0], 0.0f,
        s[1], u[1], -f[1], 0.0f,
        s[2], u[2], -f[2], 0.0f,
        0.0f, 0.0f,  0.0f, 1.0f
    };
    glMultMatrixf(m);
    glTranslatef(-eyeX, -eyeY, -eyeZ);
}

int main() {
    // Inicializar GLFW
    if (!glfwInit()) return -1;
    GLFWwindow* window = glfwCreateWindow(800, 400, "Music Visualizer", NULL, NULL);
    if (!window) { glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        fprintf(stderr, "Failed to initialize GLEW\n");
        return -1;
    }
    glfwSwapInterval(1);

    // Inicializar ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    // Listar monitores
    auto monitors = get_monitor_sources();
    int current_monitor = 0;
    if (monitors.empty()) {
        printf("No monitor sources found!\n");
        return 1;
    }

    // Inicializar captura de audio
    auto waveform = std::make_unique<WaveformBuffer>(2048);
    std::atomic<bool> running{true};
    std::thread audio_thread([&](){
        capture_audio_to_waveform(*waveform, running, monitors[current_monitor].first);
    });

    // OpenGL VBO para la onda
    const size_t max_points = 2048;
    GLuint vbo = create_vbo(max_points);

    // Bucle principal
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin("Audio Waveform", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);
        std::vector<const char*> items;
        for (auto& m : monitors) items.push_back(m.second.c_str());
        int prev_monitor = current_monitor;
        ImGui::Combo("Monitor", &current_monitor, items.data(), items.size());
        // Debug Plot antes de Render
        std::vector<float> samples = waveform->get_samples();
        printf("samples: %zu\n", samples.size());
        if (!samples.empty()) {
            float maxval = *std::max_element(samples.begin(), samples.end());
            float minval = *std::min_element(samples.begin(), samples.end());
            printf("min: %f, max: %f\n", minval, maxval);
            ImGui::Begin("Debug Plot");
            ImGui::PlotLines("Wave (ImGui)", samples.data(), samples.size(), 0, nullptr, -1.0f, 1.0f, ImVec2(760, 200));
            ImGui::End();
        } else {
            ImGui::Begin("Debug Plot");
            ImGui::Text("No audio data");
            ImGui::End();
        }
        ImGui::End();
        if (current_monitor != prev_monitor) {
            running = false;
            audio_thread.join();
            waveform = std::make_unique<WaveformBuffer>(2048);
            running = true;
            audio_thread = std::thread([&](){
                capture_audio_to_waveform(*waveform, running, monitors[current_monitor].first);
            });
        }

        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Habilitar profundidad y proyección 3D
        glEnable(GL_DEPTH_TEST);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        float aspect = (float)display_w / (float)display_h;
        myPerspective(45.0f, aspect, 0.1f, 100.0f);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        myLookAt(0, 0, 3, 0, 0, 0, 0, 1, 0);
        static float angle = 0.0f;
        angle += 0.5f;
        glRotatef(angle, 0.3f, 1.0f, 0.2f);
        glColor3f(0.2f, 1.0f, 0.7f);
        // Esfera deformada por el audio
        draw_wire_sphere_audio(1.0f, 24, 32, samples, 0.3f);
        glDisable(GL_DEPTH_TEST);

        // Dibuja la onda con OpenGL (2D) y debug
        if (!samples.empty()) {
            float y_offset = 60.0f;
            float x_margin = 40.0f;
            std::vector<float> verts(samples.size() * 2);
            for (size_t i = 0; i < samples.size(); ++i) {
                float x = x_margin + (float)i / (samples.size() - 1) * (display_w - 2 * x_margin);
                float y = (samples[i] * 0.5f + 0.5f) * (display_h - 2 * y_offset) + y_offset;
                verts[2 * i] = x;
                verts[2 * i + 1] = y;
            }
            glBindBuffer(GL_ARRAY_BUFFER, vbo);
            glBufferSubData(GL_ARRAY_BUFFER, 0, verts.size() * sizeof(float), verts.data());
            glBindBuffer(GL_ARRAY_BUFFER, 0);
            draw_waveform(vbo, samples.size());
        }

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    running = false;
    audio_thread.join();
    glDeleteBuffers(1, &vbo);
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
} 