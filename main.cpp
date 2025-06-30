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

#include <ctime>
#include <fstream>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "json.hpp"
using json = nlohmann::json;

#include <sstream>
#include <filesystem>

// Add shader sources
const char* vertexShaderSource = R"(
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;
out vec3 vColor;
uniform float uAngle;
uniform float uAspect;
uniform vec2 uTranslate;
uniform vec2 uScale;
void main() {
    float s = sin(uAngle);
    float c = cos(uAngle);
    mat2 rot = mat2(c, -s, s, c);
    vec2 pos = rot * (aPos.xy * uScale) + uTranslate;
    pos.x /= uAspect;
    gl_Position = vec4(pos, aPos.z, 1.0);
    vColor = aColor;
}
)";

const char* fragmentShaderSource = R"(
#version 330 core
in vec3 vColor;
out vec4 FragColor;
void main() {
    FragColor = vec4(vColor, 1.0);
}
)";

// Límites para random
struct RandomLimits {
    float sizeMin = 0.1f, sizeMax = 2.0f;
    float speedMin = 10.0f, speedMax = 5000.0f;
    float txMin = -1.0f, txMax = 1.0f;
    float tyMin = -1.0f, tyMax = 1.0f;
    float sxMin = 0.1f, sxMax = 2.0f;
    float syMin = 0.1f, syMax = 2.0f;
    float colorMin = 0.0f, colorMax = 1.0f;
    int numCenterMin = 0, numCenterMax = 30;
    int numRightMin = 0, numRightMax = 30;
    int numLeftMin = 0, numLeftMax = 30;
    int shapeMin = 0, shapeMax = 2;
    int segMin = 8, segMax = 128;
};

// Estructura para presets
struct Preset {
    float triSize, rotationSpeed, translateX, translateY, scaleX, scaleY;
    ImVec4 colorTop, colorLeft, colorRight;
    int numCenter, numRight, numLeft, shapeType;
    float groupAngleCenter, groupAngleRight, groupAngleLeft;
    bool randomize;
    RandomLimits randomLimits;
};

// Funciones para info del sistema
float getCPUUsage() {
    static long long lastTotalUser = 0, lastTotalUserLow = 0, lastTotalSys = 0, lastTotalIdle = 0;
    FILE* file = fopen("/proc/stat", "r");
    if (!file) return -1.0f;
    char buffer[1024];
    if (!fgets(buffer, sizeof(buffer), file)) { fclose(file); return -1.0f; }
    fclose(file);
    long long user = 0, nice = 0, sys = 0, idle = 0, iowait = 0, irq = 0, softirq = 0, steal = 0;
    int n = sscanf(buffer, "cpu %lld %lld %lld %lld %lld %lld %lld %lld", &user, &nice, &sys, &idle, &iowait, &irq, &softirq, &steal);
    if (n < 4) return -1.0f;
    long long totalUser = user, totalUserLow = nice, totalSys = sys, totalIdle = idle;
    long long total = (totalUser - lastTotalUser) + (totalUserLow - lastTotalUserLow) + (totalSys - lastTotalSys);
    long long totalAll = total + (totalIdle - lastTotalIdle);
    float percent = 0.0f;
    if (totalAll > 0) percent = 100.0f * total / totalAll;
    else percent = 0.0f;
    lastTotalUser = totalUser;
    lastTotalUserLow = totalUserLow;
    lastTotalSys = totalSys;
    lastTotalIdle = totalIdle;
    return percent;
}

float getCPUTemp() {
    namespace fs = std::filesystem;
    for (const auto& entry : fs::directory_iterator("/sys/class/thermal/")) {
        std::string path = entry.path();
        if (path.find("/temp") != std::string::npos) {
            std::ifstream in(path);
            int temp = 0;
            if (in >> temp && temp > 0) {
                return temp / 1000.0f;
            }
        }
    }
    return -1.0f;
}

float getGPUTemp() {
    // NVIDIA: intenta leer de nvidia-smi
    FILE* pipe = popen("nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null", "r");
    if (pipe) {
        char buffer[128];
        if (fgets(buffer, sizeof(buffer), pipe)) {
            float val = atof(buffer);
            pclose(pipe);
            if (val > 0.0f) return val;
        } else {
            pclose(pipe);
        }
    }
    // AMD/Intel: intenta leer de /sys/class/drm
    namespace fs = std::filesystem;
    for (const auto& entry : fs::directory_iterator("/sys/class/drm/")) {
        std::string path = entry.path();
        if (path.find("/hwmon") != std::string::npos) {
            std::string tempPath = path + "/temp1_input";
            std::ifstream in(tempPath);
            int temp = 0;
            if (in >> temp && temp > 0) {
                return temp / 1000.0f;
            }
        }
    }
    return -1.0f;
}

void savePreset(const char* filename,
                float triSize, float rotationSpeed, float translateX, float translateY, float scaleX, float scaleY,
                ImVec4 colorTop, ImVec4 colorLeft, ImVec4 colorRight,
                int numCenter, int numRight, int numLeft, int shapeType,
                float groupAngleCenter, float groupAngleRight, float groupAngleLeft,
                bool randomize, const RandomLimits& randomLimits) {
    json j;
    j["triSize"] = triSize;
    j["rotationSpeed"] = rotationSpeed;
    j["translateX"] = translateX;
    j["translateY"] = translateY;
    j["scaleX"] = scaleX;
    j["scaleY"] = scaleY;
    j["colorTop"] = {colorTop.x, colorTop.y, colorTop.z};
    j["colorLeft"] = {colorLeft.x, colorLeft.y, colorLeft.z};
    j["colorRight"] = {colorRight.x, colorRight.y, colorRight.z};
    j["numCenter"] = numCenter;
    j["numRight"] = numRight;
    j["numLeft"] = numLeft;
    j["shapeType"] = shapeType;
    j["groupAngleCenter"] = groupAngleCenter;
    j["groupAngleRight"] = groupAngleRight;
    j["groupAngleLeft"] = groupAngleLeft;
    j["randomize"] = randomize;
    j["randomLimits"] = {
        {"sizeMin", randomLimits.sizeMin}, {"sizeMax", randomLimits.sizeMax},
        {"speedMin", randomLimits.speedMin}, {"speedMax", randomLimits.speedMax},
        {"txMin", randomLimits.txMin}, {"txMax", randomLimits.txMax},
        {"tyMin", randomLimits.tyMin}, {"tyMax", randomLimits.tyMax},
        {"sxMin", randomLimits.sxMin}, {"sxMax", randomLimits.sxMax},
        {"syMin", randomLimits.syMin}, {"syMax", randomLimits.syMax},
        {"colorMin", randomLimits.colorMin}, {"colorMax", randomLimits.colorMax},
        {"numCenterMin", randomLimits.numCenterMin}, {"numCenterMax", randomLimits.numCenterMax},
        {"numRightMin", randomLimits.numRightMin}, {"numRightMax", randomLimits.numRightMax},
        {"numLeftMin", randomLimits.numLeftMin}, {"numLeftMax", randomLimits.numLeftMax},
        {"shapeMin", randomLimits.shapeMin}, {"shapeMax", randomLimits.shapeMax},
        {"segMin", randomLimits.segMin}, {"segMax", randomLimits.segMax}
    };
    std::ofstream out(filename);
    out << j.dump(4);
}

void loadPreset(const char* filename,
                float& triSize, float& rotationSpeed, float& translateX, float& translateY, float& scaleX, float& scaleY,
                ImVec4& colorTop, ImVec4& colorLeft, ImVec4& colorRight,
                int& numCenter, int& numRight, int& numLeft, int& shapeType,
                float& groupAngleCenter, float& groupAngleRight, float& groupAngleLeft,
                bool& randomize, RandomLimits& randomLimits) {
    std::ifstream in(filename);
    if (!in) return;
    json j;
    in >> j;
    triSize = j.value("triSize", triSize);
    rotationSpeed = j.value("rotationSpeed", rotationSpeed);
    translateX = j.value("translateX", translateX);
    translateY = j.value("translateY", translateY);
    scaleX = j.value("scaleX", scaleX);
    scaleY = j.value("scaleY", scaleY);
    auto ct = j.value("colorTop", std::vector<float>{colorTop.x, colorTop.y, colorTop.z});
    auto cl = j.value("colorLeft", std::vector<float>{colorLeft.x, colorLeft.y, colorLeft.z});
    auto cr = j.value("colorRight", std::vector<float>{colorRight.x, colorRight.y, colorRight.z});
    if (ct.size() == 3) { colorTop.x = ct[0]; colorTop.y = ct[1]; colorTop.z = ct[2]; }
    if (cl.size() == 3) { colorLeft.x = cl[0]; colorLeft.y = cl[1]; colorLeft.z = cl[2]; }
    if (cr.size() == 3) { colorRight.x = cr[0]; colorRight.y = cr[1]; colorRight.z = cr[2]; }
    numCenter = j.value("numCenter", numCenter);
    numRight = j.value("numRight", numRight);
    numLeft = j.value("numLeft", numLeft);
    shapeType = j.value("shapeType", shapeType);
    groupAngleCenter = j.value("groupAngleCenter", groupAngleCenter);
    groupAngleRight = j.value("groupAngleRight", groupAngleRight);
    groupAngleLeft = j.value("groupAngleLeft", groupAngleLeft);
    randomize = j.value("randomize", randomize);
    if (j.contains("randomLimits")) {
        auto rl = j["randomLimits"];
        randomLimits.sizeMin = rl.value("sizeMin", randomLimits.sizeMin);
        randomLimits.sizeMax = rl.value("sizeMax", randomLimits.sizeMax);
        randomLimits.speedMin = rl.value("speedMin", randomLimits.speedMin);
        randomLimits.speedMax = rl.value("speedMax", randomLimits.speedMax);
        randomLimits.txMin = rl.value("txMin", randomLimits.txMin);
        randomLimits.txMax = rl.value("txMax", randomLimits.txMax);
        randomLimits.tyMin = rl.value("tyMin", randomLimits.tyMin);
        randomLimits.tyMax = rl.value("tyMax", randomLimits.tyMax);
        randomLimits.sxMin = rl.value("sxMin", randomLimits.sxMin);
        randomLimits.sxMax = rl.value("sxMax", randomLimits.sxMax);
        randomLimits.syMin = rl.value("syMin", randomLimits.syMin);
        randomLimits.syMax = rl.value("syMax", randomLimits.syMax);
        randomLimits.colorMin = rl.value("colorMin", randomLimits.colorMin);
        randomLimits.colorMax = rl.value("colorMax", randomLimits.colorMax);
        randomLimits.numCenterMin = rl.value("numCenterMin", randomLimits.numCenterMin);
        randomLimits.numCenterMax = rl.value("numCenterMax", randomLimits.numCenterMax);
        randomLimits.numRightMin = rl.value("numRightMin", randomLimits.numRightMin);
        randomLimits.numRightMax = rl.value("numRightMax", randomLimits.numRightMax);
        randomLimits.numLeftMin = rl.value("numLeftMin", randomLimits.numLeftMin);
        randomLimits.numLeftMax = rl.value("numLeftMax", randomLimits.numLeftMax);
        randomLimits.shapeMin = rl.value("shapeMin", randomLimits.shapeMin);
        randomLimits.shapeMax = rl.value("shapeMax", randomLimits.shapeMax);
        randomLimits.segMin = rl.value("segMin", randomLimits.segMin);
        randomLimits.segMax = rl.value("segMax", randomLimits.segMax);
    }
}

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
    // BPM global
    float bpm = 120.0f;
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
    bool randomize = false;
    // Objetivos de randomización
    float randomLerpSpeed = 0.01f;
    float groupAngleCenter = 0.0f, groupAngleRight = 0.0f, groupAngleLeft = 0.0f;
    int numCenter = 1, numRight = 0, numLeft = 0;
    const char* shapeNames[] = {"Triángulo", "Cuadrado", "Círculo"};
    int shapeType = 0;
    int nSegments = 3;
    // Declare randomLimits
    RandomLimits randomLimits;
    GLuint shaderProgram = createShaderProgram(vertexShaderSource, fragmentShaderSource);
    GLuint VAO, VBO;
    float colorTopArr[3] = {colorTop.x, colorTop.y, colorTop.z};
    float colorLeftArr[3] = {colorLeft.x, colorLeft.y, colorLeft.z};
    float colorRightArr[3] = {colorRight.x, colorRight.y, colorRight.z};
    int actualSegments = (shapeType == 0) ? 3 : (shapeType == 1) ? 4 : 128;
    createShape(VAO, VBO, shapeType, triSize, colorTopArr, colorLeftArr, colorRightArr, actualSegments);

    int numTriangles = 1;

    // Al iniciar el programa, intenta cargar preset.json
    loadPreset("preset.json", triSize, rotationSpeed, translateX, translateY, scaleX, scaleY, colorTop, colorLeft, colorRight, numCenter, numRight, numLeft, shapeType, groupAngleCenter, groupAngleRight, groupAngleLeft, randomize, randomLimits);

    while (!glfwWindowShouldClose(window)) {
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, true);
        }
        float currentTime = glfwGetTime();
        float deltaTime = currentTime - lastTime;
        lastTime = currentTime;

        // --- BPM y fase de beat ---
        float beatPhase = fmod(currentTime * bpm / 60.0f, 1.0f); // 0..1

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
        ImGui::SliderFloat("BPM", &bpm, 30.0f, 300.0f, "%.1f");
        ImGui::Text("Beat phase: %.2f", beatPhase);
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
        ImGui::SliderInt("Cantidad de triángulos", &numTriangles, 1, 10);
        ImGui::Combo("Figura", &shapeType, shapeNames, IM_ARRAYSIZE(shapeNames));
        ImGui::SliderAngle("Rotación centro", &groupAngleCenter, 0.0f, 360.0f);
        ImGui::SliderAngle("Rotación derecha", &groupAngleRight, 0.0f, 360.0f);
        ImGui::SliderAngle("Rotación izquierda", &groupAngleLeft, 0.0f, 360.0f);
        ImGui::Checkbox("Randomizar parámetros", &randomize);
        ImGui::SliderFloat("Suavidad randomización", &randomLerpSpeed, 0.001f, 0.2f, "%.3f");
        ImGui::SliderInt("Centro", &numCenter, 0, 30);
        ImGui::SliderInt("Derecha", &numRight, 0, 30);
        ImGui::SliderInt("Izquierda", &numLeft, 0, 30);
        if (ImGui::Button("Reset")) {
            triSize = 0.8f;
            rotationSpeed = 90.0f;
            colorTop = ImVec4(1.0f, 0.0f, 0.0f, 1.0f);
            colorLeft = ImVec4(0.0f, 1.0f, 0.0f, 1.0f);
            colorRight = ImVec4(0.0f, 0.0f, 1.0f, 1.0f);
            translateX = 0.0f; translateY = 0.0f;
            scaleX = 1.0f; scaleY = 1.0f;
            numCenter = 1; numRight = 0; numLeft = 0;
            shapeType = 0;
            groupAngleCenter = 0.0f; groupAngleRight = 0.0f; groupAngleLeft = 0.0f;
            randomize = false;
            randomLimits = RandomLimits();
        }
        ImGui::SameLine();
        if (ImGui::Button("Guardar preset")) {
            savePreset("preset.json", triSize, rotationSpeed, translateX, translateY, scaleX, scaleY, colorTop, colorLeft, colorRight, numCenter, numRight, numLeft, shapeType, groupAngleCenter, groupAngleRight, groupAngleLeft, randomize, randomLimits);
        }
        ImGui::SameLine();
        if (ImGui::Button("Cargar preset")) {
            loadPreset("preset.json", triSize, rotationSpeed, translateX, translateY, scaleX, scaleY, colorTop, colorLeft, colorRight, numCenter, numRight, numLeft, shapeType, groupAngleCenter, groupAngleRight, groupAngleLeft, randomize, randomLimits);
        }
        if (ImGui::Button("Captura de pantalla")) {
            int w, h;
            glfwGetFramebufferSize(window, &w, &h);
            std::vector<unsigned char> pixels(4 * w * h);
            glPixelStorei(GL_PACK_ALIGNMENT, 1);
            glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
            // Flip vertical
            for (int y = 0; y < h / 2; ++y) {
                for (int x = 0; x < w * 4; ++x) {
                    std::swap(pixels[y * w * 4 + x], pixels[(h - 1 - y) * w * 4 + x]);
                }
            }
            char filename[128];
            std::time_t t = std::time(nullptr);
            std::strftime(filename, sizeof(filename), "screenshot_%Y%m%d_%H%M%S.png", std::localtime(&t));
            stbi_write_png(filename, w, h, 4, pixels.data(), w * 4);
        }
        ImGui::End();

        // Ventana randomización
        ImGui::SetNextWindowPos(ImVec2(width - 350, height - 400), ImGuiCond_Once);
        ImGui::SetNextWindowSize(ImVec2(340, 390), ImGuiCond_Once);
        ImGui::Begin("Randomización");
        ImGui::Checkbox("Activar random", &randomize);
        ImGui::SliderFloat("Tamaño min", &randomLimits.sizeMin, 0.01f, 2.0f, "%.2f");
        ImGui::SliderFloat("Tamaño max", &randomLimits.sizeMax, 0.01f, 2.0f, "%.2f");
        ImGui::SliderFloat("Velocidad min", &randomLimits.speedMin, 1.0f, 5000.0f, "%.1f");
        ImGui::SliderFloat("Velocidad max", &randomLimits.speedMax, 1.0f, 5000.0f, "%.1f");
        ImGui::SliderFloat("Translación X min", &randomLimits.txMin, -2.0f, 2.0f, "%.2f");
        ImGui::SliderFloat("Translación X max", &randomLimits.txMax, -2.0f, 2.0f, "%.2f");
        ImGui::SliderFloat("Translación Y min", &randomLimits.tyMin, -2.0f, 2.0f, "%.2f");
        ImGui::SliderFloat("Translación Y max", &randomLimits.tyMax, -2.0f, 2.0f, "%.2f");
        ImGui::SliderFloat("Escala X min", &randomLimits.sxMin, 0.01f, 2.0f, "%.2f");
        ImGui::SliderFloat("Escala X max", &randomLimits.sxMax, 0.01f, 2.0f, "%.2f");
        ImGui::SliderFloat("Escala Y min", &randomLimits.syMin, 0.01f, 2.0f, "%.2f");
        ImGui::SliderFloat("Escala Y max", &randomLimits.syMax, 0.01f, 2.0f, "%.2f");
        ImGui::SliderFloat("Color min", &randomLimits.colorMin, 0.0f, 1.0f, "%.2f");
        ImGui::SliderFloat("Color max", &randomLimits.colorMax, 0.0f, 1.0f, "%.2f");
        ImGui::SliderInt("Centro min", &randomLimits.numCenterMin, 0, 30);
        ImGui::SliderInt("Centro max", &randomLimits.numCenterMax, 0, 30);
        ImGui::SliderInt("Derecha min", &randomLimits.numRightMin, 0, 30);
        ImGui::SliderInt("Derecha max", &randomLimits.numRightMax, 0, 30);
        ImGui::SliderInt("Izquierda min", &randomLimits.numLeftMin, 0, 30);
        ImGui::SliderInt("Izquierda max", &randomLimits.numLeftMax, 0, 30);
        ImGui::SliderInt("Figura min", &randomLimits.shapeMin, 0, 2);
        ImGui::SliderInt("Figura max", &randomLimits.shapeMax, 0, 2);
        ImGui::SliderInt("Segmentos min", &randomLimits.segMin, 8, 128);
        ImGui::SliderInt("Segmentos max", &randomLimits.segMax, 8, 128);
        ImGui::End();

        // Ventana monitor del sistema
        ImGui::SetNextWindowPos(ImVec2(10, height - 200), ImGuiCond_Once);
        ImGui::SetNextWindowSize(ImVec2(340, 120), ImGuiCond_Once);
        ImGui::Begin("Monitor del sistema");
        float cpuUsage = getCPUUsage();
        float cpuTemp = getCPUTemp();
        float gpuTemp = getGPUTemp();
        ImGui::Text("CPU uso: %s", cpuUsage >= 0.0f ? (std::to_string(cpuUsage) + "%").c_str() : "No disponible");
        ImGui::Text("CPU temp: %s", cpuTemp >= 0.0f ? (std::to_string(cpuTemp) + " °C").c_str() : "No disponible");
        ImGui::Text("GPU temp: %s", gpuTemp >= 0.0f ? (std::to_string(gpuTemp) + " °C").c_str() : "No disponible");
        ImGui::End();

        if (autoRotate) {
            angle += rotationSpeed * deltaTime * (3.14159265f / 180.0f); // radianes
            if (angle > 2.0f * 3.14159265f) angle -= 2.0f * 3.14159265f;
            if (angle < 0.0f) angle += 2.0f * 3.14159265f;
        }

        // Animación de color
        if (animateColor) {
            float t = glfwGetTime();
            // Ejemplo: animar colorTop con el beat
            colorTop.x = 0.5f + 0.5f * sin(2.0f * 3.14159265f * beatPhase);
            colorTop.y = 0.5f + 0.5f * sin(2.0f * 3.14159265f * beatPhase + 2.0f);
            colorTop.z = 0.5f + 0.5f * sin(2.0f * 3.14159265f * beatPhase + 4.0f);
            // Los otros colores pueden seguir animándose con t
            colorLeft.x = 0.5f + 0.5f * sin(t + 1.0f);
            colorLeft.y = 0.5f + 0.5f * sin(t + 3.0f);
            colorLeft.z = 0.5f + 0.5f * sin(t + 5.0f);
            colorRight.x = 0.5f + 0.5f * sin(t + 2.0f);
            colorRight.y = 0.5f + 0.5f * sin(t + 4.0f);
            colorRight.z = 0.5f + 0.5f * sin(t + 6.0f);
        }

        // Modo random loco
        if (randomize) {
            unsigned int t = static_cast<unsigned int>(glfwGetTime() * 1000.0);
            srand(t);
            triSize = randomLimits.sizeMin + static_cast<float>(rand())/RAND_MAX * (randomLimits.sizeMax - randomLimits.sizeMin);
            rotationSpeed = randomLimits.speedMin + static_cast<float>(rand())/RAND_MAX * (randomLimits.speedMax - randomLimits.speedMin);
            translateX = randomLimits.txMin + static_cast<float>(rand())/RAND_MAX * (randomLimits.txMax - randomLimits.txMin);
            translateY = randomLimits.tyMin + static_cast<float>(rand())/RAND_MAX * (randomLimits.tyMax - randomLimits.tyMin);
            scaleX = randomLimits.sxMin + static_cast<float>(rand())/RAND_MAX * (randomLimits.sxMax - randomLimits.sxMin);
            scaleY = randomLimits.syMin + static_cast<float>(rand())/RAND_MAX * (randomLimits.syMax - randomLimits.syMin);
            colorTop.x = randomLimits.colorMin + static_cast<float>(rand())/RAND_MAX * (randomLimits.colorMax - randomLimits.colorMin);
            colorTop.y = randomLimits.colorMin + static_cast<float>(rand())/RAND_MAX * (randomLimits.colorMax - randomLimits.colorMin);
            colorTop.z = randomLimits.colorMin + static_cast<float>(rand())/RAND_MAX * (randomLimits.colorMax - randomLimits.colorMin);
            colorLeft.x = randomLimits.colorMin + static_cast<float>(rand())/RAND_MAX * (randomLimits.colorMax - randomLimits.colorMin);
            colorLeft.y = randomLimits.colorMin + static_cast<float>(rand())/RAND_MAX * (randomLimits.colorMax - randomLimits.colorMin);
            colorLeft.z = randomLimits.colorMin + static_cast<float>(rand())/RAND_MAX * (randomLimits.colorMax - randomLimits.colorMin);
            colorRight.x = randomLimits.colorMin + static_cast<float>(rand())/RAND_MAX * (randomLimits.colorMax - randomLimits.colorMin);
            colorRight.y = randomLimits.colorMin + static_cast<float>(rand())/RAND_MAX * (randomLimits.colorMax - randomLimits.colorMin);
            colorRight.z = randomLimits.colorMin + static_cast<float>(rand())/RAND_MAX * (randomLimits.colorMax - randomLimits.colorMin);
            numCenter = randomLimits.numCenterMin + rand() % (randomLimits.numCenterMax - randomLimits.numCenterMin + 1);
            numRight = randomLimits.numRightMin + rand() % (randomLimits.numRightMax - randomLimits.numRightMin + 1);
            numLeft = randomLimits.numLeftMin + rand() % (randomLimits.numLeftMax - randomLimits.numLeftMin + 1);
            shapeType = randomLimits.shapeMin + rand() % (randomLimits.shapeMax - randomLimits.shapeMin + 1);
            nSegments = randomLimits.segMin + rand() % (randomLimits.segMax - randomLimits.segMin + 1);
            groupAngleCenter = static_cast<float>(rand())/RAND_MAX * 2.0f * 3.14159265f;
            groupAngleRight = static_cast<float>(rand())/RAND_MAX * 2.0f * 3.14159265f;
            groupAngleLeft = static_cast<float>(rand())/RAND_MAX * 2.0f * 3.14159265f;
        }

        // Si cambió el tamaño, los colores o la figura, recrear el shape
        float curColorTop[3] = {colorTop.x, colorTop.y, colorTop.z};
        float curColorLeft[3] = {colorLeft.x, colorLeft.y, colorLeft.z};
        float curColorRight[3] = {colorRight.x, colorRight.y, colorRight.z};
        bool colorChanged = false;
        for (int i = 0; i < 3; ++i) {
            if (curColorTop[i] != colorTopArr[i] || curColorLeft[i] != colorLeftArr[i] || curColorRight[i] != colorRightArr[i]) {
                colorChanged = true;
                break;
            }
        }
        bool shapeChanged = false;
        static int prevShapeType = 0;
        int actualSegments = (shapeType == 0) ? 3 : (shapeType == 1) ? 4 : 128;
        if (shapeType != prevShapeType) shapeChanged = true;
        if (triSize != prevSize || colorChanged || shapeChanged) {
            glDeleteVertexArrays(1, &VAO);
            glDeleteBuffers(1, &VBO);
            createShape(VAO, VBO, shapeType, triSize, curColorTop, curColorLeft, curColorRight, actualSegments);
            prevSize = triSize;
            for (int i = 0; i < 3; ++i) {
                colorTopArr[i] = curColorTop[i];
                colorLeftArr[i] = curColorLeft[i];
                colorRightArr[i] = curColorRight[i];
            }
            prevShapeType = shapeType;
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
        float aspect = (float)width / (float)height;
        GLint aspectLoc = glGetUniformLocation(shaderProgram, "uAspect");
        GLint translateLoc = glGetUniformLocation(shaderProgram, "uTranslate");
        GLint scaleLoc = glGetUniformLocation(shaderProgram, "uScale");
        glUniform1f(aspectLoc, aspect);
        glUniform2f(scaleLoc, scaleX, scaleY);
        glUniform1f(angleLoc, angle); // ImGui::SliderAngle ya da radianes
        glBindVertexArray(VAO);
        // Centro
        for (int i = 0; i < numCenter; ++i) {
            float theta = (2.0f * 3.14159265f * i) / std::max(1, numCenter) + groupAngleCenter;
            float r = 0.5f;
            float tx = translateX + r * cos(theta);
            float ty = translateY + r * sin(theta);
            glUniform2f(translateLoc, tx, ty);
            if (shapeType == 0) glDrawArrays(GL_TRIANGLES, 0, 3);
            else if (shapeType == 1) glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
            else if (shapeType == 2) glDrawArrays(GL_TRIANGLE_FAN, 0, 130);
        }
        // Derecha
        for (int i = 0; i < numRight; ++i) {
            float theta = (2.0f * 3.14159265f * i) / std::max(1, numRight) + groupAngleRight;
            float r = 0.5f;
            float tx = translateX + 1.0f + r * cos(theta);
            float ty = translateY + r * sin(theta);
            glUniform2f(translateLoc, tx, ty);
            if (shapeType == 0) glDrawArrays(GL_TRIANGLES, 0, 3);
            else if (shapeType == 1) glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
            else if (shapeType == 2) glDrawArrays(GL_TRIANGLE_FAN, 0, 130);
        }
        // Izquierda
        for (int i = 0; i < numLeft; ++i) {
            float theta = (2.0f * 3.14159265f * i) / std::max(1, numLeft) + groupAngleLeft;
            float r = 0.5f;
            float tx = translateX - 1.0f + r * cos(theta);
            float ty = translateY + r * sin(theta);
            glUniform2f(translateLoc, tx, ty);
            if (shapeType == 0) glDrawArrays(GL_TRIANGLES, 0, 3);
            else if (shapeType == 1) glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
            else if (shapeType == 2) glDrawArrays(GL_TRIANGLE_FAN, 0, 130);
        }
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