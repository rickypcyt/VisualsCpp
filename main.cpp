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
uniform vec2 uTranslate;
uniform vec2 uScale;
void main() {
    float s = sin(uAngle);
    float c = cos(uAngle);
    mat2 rot = mat2(c, -s, s, c);
    vec2 pos = aPos.xy * uScale + uTranslate;
    vec2 rotated = rot * pos;
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

// Estructura para presets
struct Preset {
    float triSize, angle, rotationSpeed, translateX, translateY, scaleX, scaleY;
    ImVec4 colorTop, colorLeft, colorRight;
    bool autoRotate, animateColor;
};
Preset preset;
bool hasPreset = false;

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
    // Advanced controls
    ImVec4 colorTop = ImVec4(1.0f, 0.0f, 0.0f, 1.0f);
    ImVec4 colorLeft = ImVec4(0.0f, 1.0f, 0.0f, 1.0f);
    ImVec4 colorRight = ImVec4(0.0f, 0.0f, 1.0f, 1.0f);
    float translateX = 0.0f, translateY = 0.0f;
    float scaleX = 1.0f, scaleY = 1.0f;
    bool animateColor = false;

    float prevColorTop[3] = {colorTop.x, colorTop.y, colorTop.z};
    float prevColorLeft[3] = {colorLeft.x, colorLeft.y, colorLeft.z};
    float prevColorRight[3] = {colorRight.x, colorRight.y, colorRight.z};
    GLuint shaderProgram = createShaderProgram(vertexShaderSource, fragmentShaderSource);
    GLuint VAO, VBO;
    createTriangle(VAO, VBO, triSize, triSize, prevColorTop, prevColorLeft, prevColorRight);

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

        // Ventana izquierda: controles principales
        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Once);
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

        // Ventana derecha: opciones avanzadas
        ImGui::SetNextWindowPos(ImVec2(width - 350, 10), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(340, 0), ImGuiCond_Always);
        ImGui::Begin("Opciones Avanzadas");
        ImGui::ColorEdit3("Color vértice superior", (float*)&colorTop);
        ImGui::ColorEdit3("Color vértice izquierdo", (float*)&colorLeft);
        ImGui::ColorEdit3("Color vértice derecho", (float*)&colorRight);
        ImGui::Separator();
        ImGui::SliderFloat("Mover X", &translateX, -1.0f, 1.0f, "%.2f");
        ImGui::SliderFloat("Mover Y", &translateY, -1.0f, 1.0f, "%.2f");
        ImGui::SliderFloat("Escala X", &scaleX, 0.1f, 2.0f, "%.2f");
        ImGui::SliderFloat("Escala Y", &scaleY, 0.1f, 2.0f, "%.2f");
        ImGui::Checkbox("Animar color", &animateColor);
        ImGui::Separator();
        ImGui::Text("OpenGL: %s", (const char*)glGetString(GL_VERSION));
        ImGui::Text("GPU: %s", (const char*)glGetString(GL_RENDERER));
        ImGui::Text("Resolución: %dx%d", width, height);
        ImGui::Separator();
        if (ImGui::Button("Reset")) {
            triSize = 0.8f;
            angle = 0.0f;
            autoRotate = false;
            rotationSpeed = 90.0f;
            colorTop = ImVec4(1.0f, 0.0f, 0.0f, 1.0f);
            colorLeft = ImVec4(0.0f, 1.0f, 0.0f, 1.0f);
            colorRight = ImVec4(0.0f, 0.0f, 1.0f, 1.0f);
            translateX = 0.0f; translateY = 0.0f;
            scaleX = 1.0f; scaleY = 1.0f;
            animateColor = false;
        }
        ImGui::SameLine();
        if (ImGui::Button("Guardar preset")) {
            preset = {triSize, angle, rotationSpeed, translateX, translateY, scaleX, scaleY, colorTop, colorLeft, colorRight, autoRotate, animateColor};
            hasPreset = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Cargar preset") && hasPreset) {
            triSize = preset.triSize;
            angle = preset.angle;
            rotationSpeed = preset.rotationSpeed;
            translateX = preset.translateX;
            translateY = preset.translateY;
            scaleX = preset.scaleX;
            scaleY = preset.scaleY;
            colorTop = preset.colorTop;
            colorLeft = preset.colorLeft;
            colorRight = preset.colorRight;
            autoRotate = preset.autoRotate;
            animateColor = preset.animateColor;
        }
        if (ImGui::Button("Captura de pantalla")) {
            // TODO: screenshot
        }
        ImGui::End();

        if (autoRotate) {
            angle += rotationSpeed * deltaTime * (3.14159265f / 180.0f); // radianes
            if (angle > 2.0f * 3.14159265f) angle -= 2.0f * 3.14159265f;
            if (angle < 0.0f) angle += 2.0f * 3.14159265f;
        }

        // Animación de color
        if (animateColor) {
            float t = glfwGetTime();
            colorTop.x = 0.5f + 0.5f * sin(t);
            colorTop.y = 0.5f + 0.5f * sin(t + 2.0f);
            colorTop.z = 0.5f + 0.5f * sin(t + 4.0f);
            colorLeft.x = 0.5f + 0.5f * sin(t + 1.0f);
            colorLeft.y = 0.5f + 0.5f * sin(t + 3.0f);
            colorLeft.z = 0.5f + 0.5f * sin(t + 5.0f);
            colorRight.x = 0.5f + 0.5f * sin(t + 2.0f);
            colorRight.y = 0.5f + 0.5f * sin(t + 4.0f);
            colorRight.z = 0.5f + 0.5f * sin(t + 6.0f);
        }

        // Si cambió el tamaño o los colores, recrear el triángulo
        float curColorTop[3] = {colorTop.x, colorTop.y, colorTop.z};
        float curColorLeft[3] = {colorLeft.x, colorLeft.y, colorLeft.z};
        float curColorRight[3] = {colorRight.x, colorRight.y, colorRight.z};
        bool colorChanged = false;
        for (int i = 0; i < 3; ++i) {
            if (curColorTop[i] != prevColorTop[i] || curColorLeft[i] != prevColorLeft[i] || curColorRight[i] != prevColorRight[i]) {
                colorChanged = true;
                break;
            }
        }
        if (triSize != prevSize || colorChanged) {
            glDeleteVertexArrays(1, &VAO);
            glDeleteBuffers(1, &VBO);
            createTriangle(VAO, VBO, triSize, triSize, curColorTop, curColorLeft, curColorRight);
            prevSize = triSize;
            for (int i = 0; i < 3; ++i) {
                prevColorTop[i] = curColorTop[i];
                prevColorLeft[i] = curColorLeft[i];
                prevColorRight[i] = curColorRight[i];
            }
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
        GLint translateLoc = glGetUniformLocation(shaderProgram, "uTranslate");
        glUniform2f(translateLoc, translateX, translateY);
        GLint scaleLoc = glGetUniformLocation(shaderProgram, "uScale");
        glUniform2f(scaleLoc, scaleX, scaleY);
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