#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <cmath>
#include <thread>
#include <chrono>
#include "src/window_utils.h"
#include "src/shader_utils.h"
#include "src/triangle_utils.h"

// Dear ImGui
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

const char* vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;
out vec3 ourColor;
uniform float uAngle;
uniform float uAspect;
void main() {
    float s = sin(uAngle);
    float c = cos(uAngle);
    mat2 rot = mat2(c, -s, s, c);
    vec2 rotated = rot * aPos.xy;
    rotated.x /= uAspect;
    gl_Position = vec4(rotated, aPos.z, 1.0);
    ourColor = aColor;
}
)";

const char* fragmentShaderSource = R"(
#version 330 core
in vec3 ourColor;
out vec4 FragColor;
void main() {
    FragColor = vec4(ourColor, 1.0);
}
)";

int main() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }
    int width, height;
    GLFWwindow* window = createFullscreenWindow(width, height);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(0); // FPS ilimitados
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return -1;
    }
    glViewport(0, 0, width, height);
    glEnable(GL_MULTISAMPLE); // Habilitar MSAA

    // Dear ImGui: setup
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    // Triángulo
    float triSize = 0.8f;
    float prevSize = triSize;
    float angle = 0.0f;
    bool autoRotate = false;
    float rotationSpeed = 90.0f; // grados por segundo
    float lastTime = glfwGetTime();
    // FPS control
    enum FPSMode { FPS_VSYNC = 0, FPS_UNLIMITED, FPS_CUSTOM };
    int fpsMode = FPS_VSYNC;
    int prevFpsMode = FPS_VSYNC;
    int customFps = 60;

    GLuint shaderProgram = createShaderProgram(vertexShaderSource, fragmentShaderSource);
    GLuint VAO, VBO;
    createTriangle(VAO, VBO, triSize, triSize);

    while (!glfwWindowShouldClose(window)) {
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, true);
        }
        float currentTime = glfwGetTime();
        float deltaTime = currentTime - lastTime;
        lastTime = currentTime;

        // Start ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin("Triángulo");
        ImGui::SliderFloat("Tamaño", &triSize, 0.1f, 2.0f, "%.2f");
        ImGui::SliderAngle("Rotación", &angle, 0.0f, 360.0f);
        ImGui::Checkbox("Rotación automática", &autoRotate);
        ImGui::SliderFloat("Velocidad de rotación (°/s)", &rotationSpeed, 10.0f, 720.0f, "%.1f");
        const char* fpsModes[] = { "VSync", "Ilimitado", "Custom" };
        ImGui::Combo("FPS Mode", &fpsMode, fpsModes, IM_ARRAYSIZE(fpsModes));
        if (fpsMode == FPS_CUSTOM) {
            ImGui::SliderInt("Custom FPS", &customFps, 10, 1000);
        }
        ImGui::Text("ESC para salir");
        ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
        ImGui::End();

        if (autoRotate) {
            angle += rotationSpeed * deltaTime * (3.14159265f / 180.0f); // radianes
            if (angle > 2.0f * 3.14159265f) angle -= 2.0f * 3.14159265f;
            if (angle < 0.0f) angle += 2.0f * 3.14159265f;
        }

        // Si cambió el tamaño, recrear el triángulo
        if (triSize != prevSize) {
            glDeleteVertexArrays(1, &VAO);
            glDeleteBuffers(1, &VBO);
            createTriangle(VAO, VBO, triSize, triSize);
            prevSize = triSize;
        }

        // Cambiar swap interval si cambia el modo
        if (fpsMode != prevFpsMode) {
            if (fpsMode == FPS_VSYNC) {
                glfwSwapInterval(1);
            } else if (fpsMode == FPS_UNLIMITED) {
                glfwSwapInterval(0);
            } else if (fpsMode == FPS_CUSTOM) {
                glfwSwapInterval(0);
            }
            prevFpsMode = fpsMode;
        }

        glClearColor(0.0f, 0.0f, 0.0f, 1.0f); // Fondo negro
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glUseProgram(shaderProgram);
        GLint angleLoc = glGetUniformLocation(shaderProgram, "uAngle");
        glUniform1f(angleLoc, angle); // ImGui::SliderAngle ya da radianes
        float aspect = (float)width / (float)height;
        GLint aspectLoc = glGetUniformLocation(shaderProgram, "uAspect");
        glUniform1f(aspectLoc, aspect);
        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glBindVertexArray(0);

        // Render ImGui
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
        glfwPollEvents();

        // FPS custom: sleep si es necesario
        if (fpsMode == FPS_CUSTOM && customFps > 0) {
            float frameTime = 1.0f / (float)customFps;
            float elapsed = glfwGetTime() - currentTime;
            if (elapsed < frameTime) {
                // sleep en milisegundos
                int ms = (int)((frameTime - elapsed) * 1000.0f);
                if (ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(ms));
            }
        }
    }

    // Cleanup ImGui
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteProgram(shaderProgram);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
} 