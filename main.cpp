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
#include "src/audio_capture.h"
#include "src/fft_utils.h"

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
    float sizeMin = 0.05f, sizeMax = 5.0f;  // Tamaños más extremos
    float speedMin = 5.0f, speedMax = 2000.0f;  // Velocidades más variadas
    float txMin = -2.0f, txMax = 2.0f;  // Más rango de movimiento
    float tyMin = -2.0f, tyMax = 2.0f;  // Más rango de movimiento
    float sxMin = 0.05f, sxMax = 5.0f;  // Escalas más extremas
    float syMin = 0.05f, syMax = 5.0f;  // Escalas más extremas
    float colorMin = 0.0f, colorMax = 1.0f;
    int numCenterMin = 0, numCenterMax = 100;  // Mucho más objetos
    int numRightMin = 0, numRightMax = 100;   // Mucho más objetos
    int numLeftMin = 0, numLeftMax = 100;     // Mucho más objetos
    int shapeMin = 0, shapeMax = 4;  // Incluir líneas largas (4)
    int segMin = 3, segMax = 256;    // Más segmentos
};

// Estructura de flags para randomización selectiva
struct RandomAffectFlags {
    bool triSize = true;
    bool rotationSpeed = true;
    bool angle = true;
    bool translateX = true;
    bool translateY = true;
    bool scaleX = true;
    bool scaleY = true;
    bool colorTop = true;
    bool colorLeft = true;
    bool colorRight = true;
    bool shapeType = true;
    bool nSegments = true;
    bool groupAngle = true;
    bool numCenter = true;
    bool numRight = true;
    bool numLeft = true;
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
                bool randomize, const RandomLimits& randomLimits,
                const RandomAffectFlags& randomAffect,
                float groupSeparation, bool onlyRGB, bool animateColor, float bpm, 
                int fpsMode, int customFps, bool fractalMode, float fractalDepth) {
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
    j["groupSeparation"] = groupSeparation;
    j["onlyRGB"] = onlyRGB;
    j["animateColor"] = animateColor;
    j["bpm"] = bpm;
    j["fpsMode"] = fpsMode;
    j["customFps"] = customFps;
    j["fractalMode"] = fractalMode;
    j["fractalDepth"] = fractalDepth;
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
    j["randomAffect"] = {
        {"triSize", randomAffect.triSize},
        {"rotationSpeed", randomAffect.rotationSpeed},
        {"angle", randomAffect.angle},
        {"translateX", randomAffect.translateX},
        {"translateY", randomAffect.translateY},
        {"scaleX", randomAffect.scaleX},
        {"scaleY", randomAffect.scaleY},
        {"colorTop", randomAffect.colorTop},
        {"colorLeft", randomAffect.colorLeft},
        {"colorRight", randomAffect.colorRight},
        {"shapeType", randomAffect.shapeType},
        {"nSegments", randomAffect.nSegments},
        {"groupAngle", randomAffect.groupAngle},
        {"numCenter", randomAffect.numCenter},
        {"numRight", randomAffect.numRight},
        {"numLeft", randomAffect.numLeft}
    };
    std::ofstream out(filename);
    out << j.dump(4);
}

void loadPreset(const char* filename,
                float& triSize, float& rotationSpeed, float& translateX, float& translateY, float& scaleX, float& scaleY,
                ImVec4& colorTop, ImVec4& colorLeft, ImVec4& colorRight,
                int& numCenter, int& numRight, int& numLeft, int& shapeType,
                float& groupAngleCenter, float& groupAngleRight, float& groupAngleLeft,
                bool& randomize, RandomLimits& randomLimits,
                RandomAffectFlags& randomAffect,
                float& groupSeparation, bool& onlyRGB, bool& animateColor, float& bpm,
                int& fpsMode, int& customFps, bool& fractalMode, float& fractalDepth) {
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
    groupSeparation = j.value("groupSeparation", groupSeparation);
    onlyRGB = j.value("onlyRGB", onlyRGB);
    animateColor = j.value("animateColor", animateColor);
    bpm = j.value("bpm", bpm);
    fpsMode = j.value("fpsMode", fpsMode);
    customFps = j.value("customFps", customFps);
    fractalMode = j.value("fractalMode", fractalMode);
    fractalDepth = j.value("fractalDepth", fractalDepth);
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
    if (j.contains("randomAffect")) {
        auto ra = j["randomAffect"];
        randomAffect.triSize = ra.value("triSize", randomAffect.triSize);
        randomAffect.rotationSpeed = ra.value("rotationSpeed", randomAffect.rotationSpeed);
        randomAffect.angle = ra.value("angle", randomAffect.angle);
        randomAffect.translateX = ra.value("translateX", randomAffect.translateX);
        randomAffect.translateY = ra.value("translateY", randomAffect.translateY);
        randomAffect.scaleX = ra.value("scaleX", randomAffect.scaleX);
        randomAffect.scaleY = ra.value("scaleY", randomAffect.scaleY);
        randomAffect.colorTop = ra.value("colorTop", randomAffect.colorTop);
        randomAffect.colorLeft = ra.value("colorLeft", randomAffect.colorLeft);
        randomAffect.colorRight = ra.value("colorRight", randomAffect.colorRight);
        randomAffect.shapeType = ra.value("shapeType", randomAffect.shapeType);
        randomAffect.nSegments = ra.value("nSegments", randomAffect.nSegments);
        randomAffect.groupAngle = ra.value("groupAngle", randomAffect.groupAngle);
        randomAffect.numCenter = ra.value("numCenter", randomAffect.numCenter);
        randomAffect.numRight = ra.value("numRight", randomAffect.numRight);
        randomAffect.numLeft = ra.value("numLeft", randomAffect.numLeft);
    }
}

// Añadir tipo de figura línea
enum ShapeType { SHAPE_TRIANGLE = 0, SHAPE_SQUARE, SHAPE_CIRCLE, SHAPE_LINE, SHAPE_LONG_LINES, SHAPE_COUNT };
const char* shapeNames[] = {"Triángulo", "Cuadrado", "Círculo", "Línea", "Líneas largas"};

// Estructura de parámetros de un objeto visual
struct VisualObjectParams {
    float triSize = 0.8f;
    float rotationSpeed = 90.0f;
    float angle = 0.0f;
    float translateX = 0.0f, translateY = 0.0f;
    float scaleX = 1.0f, scaleY = 1.0f;
    ImVec4 colorTop = ImVec4(1.0f, 0.0f, 0.0f, 1.0f);
    ImVec4 colorLeft = ImVec4(0.0f, 1.0f, 0.0f, 1.0f);
    ImVec4 colorRight = ImVec4(0.0f, 0.0f, 1.0f, 1.0f);
    int shapeType = SHAPE_TRIANGLE;
    int nSegments = 32;
    float groupAngle = 0.0f;
};

// Estructura de objetivos random para suavidad
struct VisualObjectTargets {
    VisualObjectParams target;
};

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

    // Declarar randomAffect para randomización selectiva
    RandomAffectFlags randomAffect;

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
    int shapeType = 0;
    int nSegments = 3;
    // Declare randomLimits
    RandomLimits randomLimits;
    // Objetivos random para suavidad
    float targetTriSize = triSize;
    float targetRotationSpeed = rotationSpeed;
    float targetTranslateX = translateX, targetTranslateY = translateY;
    float targetScaleX = scaleX, targetScaleY = scaleY;
    ImVec4 targetColorTop = colorTop, targetColorLeft = colorLeft, targetColorRight = colorRight;
    int targetNumCenter = numCenter, targetNumRight = numRight, targetNumLeft = numLeft;
    int targetShapeType = shapeType;
    float targetGroupAngleCenter = groupAngleCenter, targetGroupAngleRight = groupAngleRight, targetGroupAngleLeft = groupAngleLeft;
    int targetNSegments = nSegments;
    GLuint shaderProgram = createShaderProgram(vertexShaderSource, fragmentShaderSource);
    GLuint VAO, VBO;
    float colorTopArr[3] = {colorTop.x, colorTop.y, colorTop.z};
    float colorLeftArr[3] = {colorLeft.x, colorLeft.y, colorLeft.z};
    float colorRightArr[3] = {colorRight.x, colorRight.y, colorRight.z};
    int actualSegments = (shapeType == 0) ? 3 : (shapeType == 1) ? 4 : 128;
    createShape(VAO, VBO, shapeType, triSize, colorTopArr, colorLeftArr, colorRightArr, actualSegments);

    int numTriangles = 1;

    // Variables que necesitan estar declaradas antes de loadPreset
    float groupSeparation = 1.0f;
    bool onlyRGB = false;
    bool fractalMode = false;
    float fractalDepth = 3.0f;

    // Al iniciar el programa, intenta cargar preset.json
    loadPreset("preset.json", triSize, rotationSpeed, translateX, translateY, scaleX, scaleY, colorTop, colorLeft, colorRight, numCenter, numRight, numLeft, shapeType, groupAngleCenter, groupAngleRight, groupAngleLeft, randomize, randomLimits, randomAffect, groupSeparation, onlyRGB, animateColor, bpm, fpsMode, customFps, fractalMode, fractalDepth);

    // Grupos: centro, derecha, izquierda
    const int MAX_OBJECTS = 30;
    struct VisualGroup {
        std::vector<VisualObjectParams> objects;
        std::vector<VisualObjectTargets> targets;
        int numObjects = 1;
        float groupAngle = 0.0f;
    };
    VisualGroup groups[3]; // 0: centro, 1: derecha, 2: izquierda

    // Inicializar grupos
    for (int g = 0; g < 3; ++g) {
        groups[g].numObjects = (g == 0) ? 1 : 0;
        groups[g].groupAngle = 0.0f;
        groups[g].objects.resize(MAX_OBJECTS);
        groups[g].targets.resize(MAX_OBJECTS);
    }

    // Variables globales para animación
    bool autoGroupRotate = false;

    // 1. Flags de randomización por grupo
    // Variables para randomización por grupo (ahora manejadas por randomAffect)
    // static bool randomShapeType[3] = {true, true, true};
    // static bool randomNSegments[3] = {true, true, true};

    // Declarar lambdas antes del bucle de randomización
    auto frand = []() { return static_cast<float>(rand())/RAND_MAX; };
    auto near = [](float a, float b, float eps=0.01f) { return fabs(a-b) < eps; };
    auto nearInt = [](int a, int b) { return a == b; };
    
    // Variables para randomización más natural
    static float lastRandomizeTime[3] = {0.0f, 0.0f, 0.0f}; // Tiempo de última randomización por grupo
    static float randomizeIntervals[3] = {2.0f, 3.0f, 2.5f}; // Intervalos diferentes por grupo
    static float randomizeVariation[3] = {0.5f, 0.8f, 0.6f}; // Variación en los intervalos

    // 1. Parámetro de separación de grupos
    static float targetGroupSeparation = 1.0f;
    bool randomizeGroupSeparation = false;

    // --- NUEVO: Audio Reactivo ---
    bool audioReactive = false;
    static bool audioInit = false;
    static AudioCapture* audio = nullptr;
    static FFTUtils* fft = nullptr;
    static std::vector<int32_t> audioBuffer;
    static std::vector<float> monoBuffer;
    static std::vector<float> spectrum;
    const int audioFftSize = 1024;
    const char* audioDevice = "alsa_output.pci-0000_05_00.6.analog-stereo.monitor";
    const int audioSampleRate = 48000;
    const int audioChannels = 2;

    while (!glfwWindowShouldClose(window)) {
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, true);
        }
        float currentTime = glfwGetTime();
        float deltaTime = currentTime - lastTime;
        lastTime = currentTime;

        // --- BPM y fase de beat ---
        float beatPhase = fmod(currentTime * bpm / 60.0f, 1.0f); // 0..1

        // Aplicar onlyRGB si está activo (antes de la animación de color)
        if (onlyRGB) {
            for (int g = 0; g < 3; ++g) {
                for (int i = 0; i < groups[g].numObjects; ++i) {
                    groups[g].objects[i].colorTop   = ImVec4(1.0f, 0.0f, 0.0f, 1.0f);
                    groups[g].objects[i].colorLeft  = ImVec4(0.0f, 1.0f, 0.0f, 1.0f);
                    groups[g].objects[i].colorRight = ImVec4(0.0f, 0.0f, 1.0f, 1.0f);
                }
            }
        }

        // Start ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Ventana izquierda: controles principales
        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Once);
        ImGui::Begin("Triángulo");
        ImGui::SliderFloat("Tamaño", &groups[0].objects[0].triSize, 0.1f, 2.0f, "%.2f");
        ImGui::SliderAngle("Rotación", &groups[0].objects[0].angle, 0.0f, 360.0f);
        ImGui::Checkbox("Rotación automática", &autoRotate);
        ImGui::SliderFloat("Velocidad de rotación (°/s)", &groups[0].objects[0].rotationSpeed, 10.0f, 720.0f, "%.1f");
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
        ImGui::SliderFloat("Separación de grupos", &groupSeparation, 0.0f, 2.0f, "%.2f");
        ImGui::Checkbox("Randomizar separación de grupos", &randomizeGroupSeparation);
        ImGui::Separator();
        // Grupo Centro
        ImGui::Text("=== GRUPO CENTRO ===");
        ImGui::SliderInt("Cantidad Centro", &groups[0].numObjects, 0, 100);
        ImGui::Combo("Figura Centro", &groups[0].objects[0].shapeType, shapeNames, IM_ARRAYSIZE(shapeNames));
        ImGui::SliderInt("Segmentos Centro", &groups[0].objects[0].nSegments, 3, 256);
        ImGui::SliderAngle("Ángulo Centro", &groups[0].groupAngle, 0.0f, 360.0f);
        ImGui::ColorEdit3("Color Top Centro", (float*)&groups[0].objects[0].colorTop);
        ImGui::ColorEdit3("Color Left Centro", (float*)&groups[0].objects[0].colorLeft);
        ImGui::ColorEdit3("Color Right Centro", (float*)&groups[0].objects[0].colorRight);
        ImGui::SliderFloat("Mover X Centro", &groups[0].objects[0].translateX, -1.0f, 1.0f, "%.2f");
        ImGui::SliderFloat("Mover Y Centro", &groups[0].objects[0].translateY, -1.0f, 1.0f, "%.2f");
        ImGui::SliderFloat("Escala X Centro", &groups[0].objects[0].scaleX, 0.1f, 2.0f, "%.2f");
        ImGui::SliderFloat("Escala Y Centro", &groups[0].objects[0].scaleY, 0.1f, 2.0f, "%.2f");
        ImGui::Separator();
        // Grupo Derecha
        ImGui::Text("=== GRUPO DERECHA ===");
        ImGui::SliderInt("Cantidad Derecha", &groups[1].numObjects, 0, 100);
        ImGui::Combo("Figura Derecha", &groups[1].objects[0].shapeType, shapeNames, IM_ARRAYSIZE(shapeNames));
        ImGui::SliderInt("Segmentos Derecha", &groups[1].objects[0].nSegments, 3, 256);
        ImGui::SliderAngle("Ángulo Derecha", &groups[1].groupAngle, 0.0f, 360.0f);
        ImGui::ColorEdit3("Color Top Derecha", (float*)&groups[1].objects[0].colorTop);
        ImGui::ColorEdit3("Color Left Derecha", (float*)&groups[1].objects[0].colorLeft);
        ImGui::ColorEdit3("Color Right Derecha", (float*)&groups[1].objects[0].colorRight);
        ImGui::SliderFloat("Mover X Derecha", &groups[1].objects[0].translateX, -1.0f, 1.0f, "%.2f");
        ImGui::SliderFloat("Mover Y Derecha", &groups[1].objects[0].translateY, -1.0f, 1.0f, "%.2f");
        ImGui::SliderFloat("Escala X Derecha", &groups[1].objects[0].scaleX, 0.1f, 2.0f, "%.2f");
        ImGui::SliderFloat("Escala Y Derecha", &groups[1].objects[0].scaleY, 0.1f, 2.0f, "%.2f");
        ImGui::Separator();
        // Grupo Izquierda
        ImGui::Text("=== GRUPO IZQUIERDA ===");
        ImGui::SliderInt("Cantidad Izquierda", &groups[2].numObjects, 0, 100);
        ImGui::Combo("Figura Izquierda", &groups[2].objects[0].shapeType, shapeNames, IM_ARRAYSIZE(shapeNames));
        ImGui::SliderInt("Segmentos Izquierda", &groups[2].objects[0].nSegments, 3, 256);
        ImGui::SliderAngle("Ángulo Izquierda", &groups[2].groupAngle, 0.0f, 360.0f);
        ImGui::ColorEdit3("Color Top Izquierda", (float*)&groups[2].objects[0].colorTop);
        ImGui::ColorEdit3("Color Left Izquierda", (float*)&groups[2].objects[0].colorLeft);
        ImGui::ColorEdit3("Color Right Izquierda", (float*)&groups[2].objects[0].colorRight);
        ImGui::SliderFloat("Mover X Izquierda", &groups[2].objects[0].translateX, -1.0f, 1.0f, "%.2f");
        ImGui::SliderFloat("Mover Y Izquierda", &groups[2].objects[0].translateY, -1.0f, 1.0f, "%.2f");
        ImGui::SliderFloat("Escala X Izquierda", &groups[2].objects[0].scaleX, 0.1f, 2.0f, "%.2f");
        ImGui::SliderFloat("Escala Y Izquierda", &groups[2].objects[0].scaleY, 0.1f, 2.0f, "%.2f");
        ImGui::Separator();
        ImGui::Checkbox("Animar color", &animateColor);
        ImGui::Checkbox("Solo colores RGB puros", &onlyRGB);
        ImGui::Separator();
        ImGui::Text("=== MODO FRACTAL ===");
        ImGui::Checkbox("Modo Fractal", &fractalMode);
        if (fractalMode) {
            ImGui::SliderFloat("Profundidad Fractal", &fractalDepth, 1.0f, 5.0f, "%.1f");
            ImGui::Text("Crea fractales animados y coloridos");
            ImGui::Text("basados en la figura seleccionada");
            ImGui::Text("✅ Todas las figuras son compatibles con fractales");
        }
        ImGui::Separator();
        ImGui::Text("OpenGL: %s", (const char*)glGetString(GL_VERSION));
        ImGui::Text("GPU: %s", (const char*)glGetString(GL_RENDERER));
        ImGui::Text("Resolución: %dx%d", width, height);
        ImGui::Separator();
        ImGui::SliderInt("Cantidad de triángulos", &numTriangles, 1, 10);
        ImGui::Combo("Figura", &groups[0].objects[0].shapeType, shapeNames, IM_ARRAYSIZE(shapeNames));
        ImGui::SliderAngle("Rotación centro", &groupAngleCenter, 0.0f, 360.0f);
        ImGui::SliderAngle("Rotación derecha", &groupAngleRight, 0.0f, 360.0f);
        ImGui::SliderAngle("Rotación izquierda", &groupAngleLeft, 0.0f, 360.0f);
        ImGui::Checkbox("Randomizar parámetros", &randomize);
        ImGui::SliderFloat("Suavidad randomización", &randomLerpSpeed, 0.001f, 0.2f, "%.3f");
        ImGui::SliderInt("Centro", &numCenter, 0, 100);
        ImGui::SliderInt("Derecha", &numRight, 0, 100);
        ImGui::SliderInt("Izquierda", &numLeft, 0, 100);
        if (ImGui::Button("Reset")) {
            for (int g = 0; g < 3; ++g) {
                groups[g].objects[0].triSize = 0.8f;
                groups[g].objects[0].rotationSpeed = 90.0f;
                groups[g].objects[0].colorTop = ImVec4(1.0f, 0.0f, 0.0f, 1.0f);
                groups[g].objects[0].colorLeft = ImVec4(0.0f, 1.0f, 0.0f, 1.0f);
                groups[g].objects[0].colorRight = ImVec4(0.0f, 0.0f, 1.0f, 1.0f);
                groups[g].objects[0].translateX = 0.0f; groups[g].objects[0].translateY = 0.0f;
                groups[g].objects[0].scaleX = 1.0f; groups[g].objects[0].scaleY = 1.0f;
                groups[g].objects[0].shapeType = 0;
                groups[g].objects[0].nSegments = 32;
                groups[g].groupAngle = 0.0f;
                groups[g].numObjects = (g == 0) ? 1 : 0;
            }
            groupSeparation = 1.0f;
            randomize = false;
            randomLimits = RandomLimits();
            randomAffect = RandomAffectFlags();
        }
        ImGui::SameLine();
        if (ImGui::Button("Guardar preset")) {
            savePreset("preset.json", 
                groups[0].objects[0].triSize, groups[0].objects[0].rotationSpeed, 
                groups[0].objects[0].translateX, groups[0].objects[0].translateY, 
                groups[0].objects[0].scaleX, groups[0].objects[0].scaleY, 
                groups[0].objects[0].colorTop, groups[0].objects[0].colorLeft, groups[0].objects[0].colorRight, 
                groups[0].numObjects, groups[1].numObjects, groups[2].numObjects, 
                groups[0].objects[0].shapeType, 
                groups[0].groupAngle, groups[1].groupAngle, groups[2].groupAngle, 
                randomize, randomLimits, randomAffect, groupSeparation, onlyRGB, animateColor, bpm, fpsMode, customFps, fractalMode, fractalDepth);
        }
        ImGui::SameLine();
        if (ImGui::Button("Cargar preset")) {
            loadPreset("preset.json", 
                groups[0].objects[0].triSize, groups[0].objects[0].rotationSpeed, 
                groups[0].objects[0].translateX, groups[0].objects[0].translateY, 
                groups[0].objects[0].scaleX, groups[0].objects[0].scaleY, 
                groups[0].objects[0].colorTop, groups[0].objects[0].colorLeft, groups[0].objects[0].colorRight, 
                groups[0].numObjects, groups[1].numObjects, groups[2].numObjects, 
                groups[0].objects[0].shapeType, 
                groups[0].groupAngle, groups[1].groupAngle, groups[2].groupAngle, 
                randomize, randomLimits, randomAffect, groupSeparation, onlyRGB, animateColor, bpm, fpsMode, customFps, fractalMode, fractalDepth);
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
        ImGui::SliderFloat("Suavidad randomización", &randomLerpSpeed, 0.001f, 0.2f, "%.3f");
        ImGui::SliderFloat("Frecuencia base", &randomizeIntervals[0], 0.5f, 10.0f, "%.1f");
        ImGui::Text("(Intervalo base para todos los grupos)");
        ImGui::Separator();
        ImGui::Text("¿Qué randomizar?");
        ImGui::Checkbox("Tamaño", &randomAffect.triSize);
        ImGui::Checkbox("Velocidad rotación", &randomAffect.rotationSpeed);
        ImGui::Checkbox("Ángulo", &randomAffect.angle);
        ImGui::Checkbox("Translación X", &randomAffect.translateX);
        ImGui::Checkbox("Translación Y", &randomAffect.translateY);
        ImGui::Checkbox("Escala X", &randomAffect.scaleX);
        ImGui::Checkbox("Escala Y", &randomAffect.scaleY);
        ImGui::Checkbox("Color Top", &randomAffect.colorTop);
        ImGui::Checkbox("Color Left", &randomAffect.colorLeft);
        ImGui::Checkbox("Color Right", &randomAffect.colorRight);
        ImGui::Checkbox("Tipo de figura", &randomAffect.shapeType);
        ImGui::Checkbox("Segmentos (círculo/línea)", &randomAffect.nSegments);
        ImGui::Checkbox("Ángulo de grupo", &randomAffect.groupAngle);
        ImGui::Checkbox("Cantidad Centro", &randomAffect.numCenter);
        ImGui::Checkbox("Cantidad Derecha", &randomAffect.numRight);
        ImGui::Checkbox("Cantidad Izquierda", &randomAffect.numLeft);
        ImGui::Separator();
        ImGui::Text("Límites de randomización:");
        ImGui::SliderFloat("Tamaño min", &randomLimits.sizeMin, 0.05f, 5.0f, "%.2f");
        ImGui::SliderFloat("Tamaño max", &randomLimits.sizeMax, 0.05f, 5.0f, "%.2f");
        ImGui::SliderFloat("Velocidad min", &randomLimits.speedMin, 5.0f, 2000.0f, "%.1f");
        ImGui::SliderFloat("Velocidad max", &randomLimits.speedMax, 5.0f, 2000.0f, "%.1f");
        ImGui::SliderFloat("Translación X min", &randomLimits.txMin, -2.0f, 2.0f, "%.2f");
        ImGui::SliderFloat("Translación X max", &randomLimits.txMax, -2.0f, 2.0f, "%.2f");
        ImGui::SliderFloat("Translación Y min", &randomLimits.tyMin, -2.0f, 2.0f, "%.2f");
        ImGui::SliderFloat("Translación Y max", &randomLimits.tyMax, -2.0f, 2.0f, "%.2f");
        ImGui::SliderFloat("Escala X min", &randomLimits.sxMin, 0.05f, 5.0f, "%.2f");
        ImGui::SliderFloat("Escala X max", &randomLimits.sxMax, 0.05f, 5.0f, "%.2f");
        ImGui::SliderFloat("Escala Y min", &randomLimits.syMin, 0.05f, 5.0f, "%.2f");
        ImGui::SliderFloat("Escala Y max", &randomLimits.syMax, 0.05f, 5.0f, "%.2f");
        ImGui::SliderInt("Centro min", &randomLimits.numCenterMin, 0, 100);
        ImGui::SliderInt("Centro max", &randomLimits.numCenterMax, 0, 100);
        ImGui::SliderInt("Derecha min", &randomLimits.numRightMin, 0, 100);
        ImGui::SliderInt("Derecha max", &randomLimits.numRightMax, 0, 100);
        ImGui::SliderInt("Izquierda min", &randomLimits.numLeftMin, 0, 100);
        ImGui::SliderInt("Izquierda max", &randomLimits.numLeftMax, 0, 100);
        ImGui::SliderInt("Figura min", &randomLimits.shapeMin, 0, 4);
        ImGui::SliderInt("Figura max", &randomLimits.shapeMax, 0, 4);
        ImGui::SliderInt("Segmentos min", &randomLimits.segMin, 3, 256);
        ImGui::SliderInt("Segmentos max", &randomLimits.segMax, 3, 256);
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

        // UI para animación global
        ImGui::Begin("Opciones Globales");
        ImGui::Checkbox("Rotación automática", &autoRotate);
        ImGui::Checkbox("Animar color", &animateColor);
        ImGui::Checkbox("Visuales controlados por audio del sistema", &audioReactive);
        ImGui::End();

        // --- Inicialización de audio y FFT si es necesario ---
        if (audioReactive && !audioInit) {
            audio = new AudioCapture(audioDevice, audioSampleRate, audioChannels);
            fft = new FFTUtils(audioFftSize);
            audioBuffer.resize(audioFftSize * audioChannels);
            monoBuffer.resize(audioFftSize);
            spectrum.resize(audioFftSize / 2);
            audioInit = true;
        }
        if (!audioReactive && audioInit) {
            delete audio;
            delete fft;
            audio = nullptr;
            fft = nullptr;
            audioInit = false;
        }
        // --- Procesamiento de audio y FFT ---
        if (audioReactive && audio && fft) {
            if (audio->read(audioBuffer)) {
                for (int i = 0; i < audioFftSize; ++i) {
                    int32_t left = audioBuffer[i * 2];
                    int32_t right = audioBuffer[i * 2 + 1];
                    monoBuffer[i] = (left + right) / 2.0f / 2147483648.0f;
                }
                spectrum = fft->compute(monoBuffer);
            }
        }
        // --- Modulación de visuales por audio ---
        if (audioReactive && !spectrum.empty()) {
            float bass = 0.0f, mid = 0.0f, treble = 0.0f;
            int n = spectrum.size();
            for (int i = 0; i < n; ++i) {
                if (i < n / 8) bass += spectrum[i];
                else if (i < n / 3) mid += spectrum[i];
                else treble += spectrum[i];
            }
            bass /= (n / 8);
            mid /= (n / 3 - n / 8);
            treble /= (n - n / 3);
            // Modula tamaño, color y rotación de los objetos principales
            for (int g = 0; g < 3; ++g) {
                for (int i = 0; i < groups[g].numObjects; ++i) {
                    VisualObjectParams& obj = groups[g].objects[i];
                    obj.triSize = 0.5f + bass * 2.0f;
                    obj.rotationSpeed = 90.0f + mid * 100.0f;
                    obj.colorTop.x = std::min(1.0f, 0.5f + bass * 2.0f);
                    obj.colorTop.y = std::min(1.0f, 0.5f + mid * 2.0f);
                    obj.colorTop.z = std::min(1.0f, 0.5f + treble * 2.0f);
                }
            }
        }

        // 2. UI para randomización por grupo (ahora manejado por randomAffect)
        ImGui::Separator();
        ImGui::Text("Randomización por grupo:");
        ImGui::Text("(Usar controles de '¿Qué randomizar?' en la ventana de Randomización)");
        
        // Mostrar estado de randomización por grupo
        if (randomize) {
            ImGui::Text("Estado de randomización:");
            for (int g = 0; g < 3; ++g) {
                const char* groupNames[] = {"Centro", "Derecha", "Izquierda"};
                float timeSinceLast = currentTime - lastRandomizeTime[g];
                float nextInterval = randomizeIntervals[g] + randomizeVariation[g] * sin(currentTime * 0.3f + g);
                float progress = timeSinceLast / nextInterval;
                
                ImGui::Text("%s: %.1fs (%.0f%%)", groupNames[g], nextInterval - timeSinceLast, progress * 100.0f);
            }
        }

        // 2. UI para separación de grupos
        ImGui::SliderFloat("Separación de grupos", &groupSeparation, 0.0f, 2.0f, "%.2f");
        ImGui::Checkbox("Randomizar separación de grupos", &randomizeGroupSeparation);

        // --- Actualización de objetos: rotación automática y animación de color ---
        for (int g = 0; g < 3; ++g) {
            int n = groups[g].numObjects;
            for (int i = 0; i < n; ++i) {
                VisualObjectParams& obj = groups[g].objects[i];
                // Rotación automática
                if (autoRotate) {
                    obj.angle += obj.rotationSpeed * deltaTime * (3.14159265f / 180.0f);
                    if (obj.angle > 2.0f * 3.14159265f) obj.angle -= 2.0f * 3.14159265f;
                    if (obj.angle < 0.0f) obj.angle += 2.0f * 3.14159265f;
                }
                // Animación de color - SIEMPRE activa
                float t = currentTime;
                float phase = beatPhase + (float)i * 0.3f + (float)g * 0.5f; // Diferente fase por objeto y grupo
                obj.colorTop.x = 0.5f + 0.5f * sin(2.0f * 3.14159265f * phase);
                obj.colorTop.y = 0.5f + 0.5f * sin(2.0f * 3.14159265f * phase + 2.0f);
                obj.colorTop.z = 0.5f + 0.5f * sin(2.0f * 3.14159265f * phase + 4.0f);
                obj.colorLeft.x = 0.5f + 0.5f * sin(t + 1.0f + (float)i * 0.2f);
                obj.colorLeft.y = 0.5f + 0.5f * sin(t + 3.0f + (float)i * 0.2f);
                obj.colorLeft.z = 0.5f + 0.5f * sin(t + 5.0f + (float)i * 0.2f);
                obj.colorRight.x = 0.5f + 0.5f * sin(t + 2.0f + (float)i * 0.2f);
                obj.colorRight.y = 0.5f + 0.5f * sin(t + 4.0f + (float)i * 0.2f);
                obj.colorRight.z = 0.5f + 0.5f * sin(t + 6.0f + (float)i * 0.2f);
            }
        }
        // 3. Randomización y recreación de shapes por grupo (MEJORADA)
        for (int g = 0; g < 3; ++g) {
            VisualObjectParams& obj = groups[g].objects[0];
            VisualObjectTargets& tgt = groups[g].targets[0];
            
            // Sistema de randomización más natural con intervalos variables
            bool shouldRandomize = false;
            if (randomize) {
                // Calcular si es momento de randomizar basado en intervalos variables
                float timeSinceLastRandom = currentTime - lastRandomizeTime[g];
                float currentInterval = randomizeIntervals[g] + randomizeVariation[g] * sin(currentTime * 0.3f + g);
                
                if (timeSinceLastRandom >= currentInterval) {
                    shouldRandomize = true;
                    lastRandomizeTime[g] = currentTime;
                    
                    // Variar el siguiente intervalo para hacerlo menos predecible
                    randomizeIntervals[g] = 1.0f + frand() * 4.0f; // 1-5 segundos
                    randomizeVariation[g] = 0.2f + frand() * 1.0f; // 0.2-1.2 segundos de variación
                }
            }
            
            // Randomizar shapeType por grupo
            static int tgtShapeType[3] = {obj.shapeType, obj.shapeType, obj.shapeType};
            if (shouldRandomize && randomAffect.shapeType) {
                int min = randomLimits.shapeMin;
                int max = randomLimits.shapeMax;
                tgtShapeType[g] = min + rand() % (max - min + 1);
            }
            obj.shapeType += (int)((tgtShapeType[g] - obj.shapeType) * randomLerpSpeed + 0.5f);
            
            // Randomizar nSegments por grupo
            static int tgtNSegments[3] = {obj.nSegments, obj.nSegments, obj.nSegments};
            if (shouldRandomize && randomAffect.nSegments) {
                int min = randomLimits.segMin;
                int max = randomLimits.segMax;
                tgtNSegments[g] = min + rand() % (max - min + 1);
            }
            obj.nSegments += (int)((tgtNSegments[g] - obj.nSegments) * randomLerpSpeed + 0.5f);
            
            // Inicializar targets si es la primera vez
            if (tgt.target.triSize == 0.0f) tgt.target = obj;
            
            if (randomize) {
                // triSize
                if (shouldRandomize && randomAffect.triSize)
                    tgt.target.triSize = randomLimits.sizeMin + frand() * (randomLimits.sizeMax - randomLimits.sizeMin);
                obj.triSize += (tgt.target.triSize - obj.triSize) * randomLerpSpeed;
                
                // rotationSpeed
                if (shouldRandomize && randomAffect.rotationSpeed)
                    tgt.target.rotationSpeed = randomLimits.speedMin + frand() * (randomLimits.speedMax - randomLimits.speedMin);
                obj.rotationSpeed += (tgt.target.rotationSpeed - obj.rotationSpeed) * randomLerpSpeed;
                
                // angle
                if (shouldRandomize && randomAffect.angle)
                    tgt.target.angle = frand() * 2.0f * 3.14159265f;
                obj.angle += (tgt.target.angle - obj.angle) * randomLerpSpeed;
                
                // translateX
                if (shouldRandomize && randomAffect.translateX)
                    tgt.target.translateX = randomLimits.txMin + frand() * (randomLimits.txMax - randomLimits.txMin);
                obj.translateX += (tgt.target.translateX - obj.translateX) * randomLerpSpeed;
                
                // translateY
                if (shouldRandomize && randomAffect.translateY)
                    tgt.target.translateY = randomLimits.tyMin + frand() * (randomLimits.tyMax - randomLimits.tyMin);
                obj.translateY += (tgt.target.translateY - obj.translateY) * randomLerpSpeed;
                
                // scaleX
                if (shouldRandomize && randomAffect.scaleX)
                    tgt.target.scaleX = randomLimits.sxMin + frand() * (randomLimits.sxMax - randomLimits.sxMin);
                obj.scaleX += (tgt.target.scaleX - obj.scaleX) * randomLerpSpeed;
                
                // scaleY
                if (shouldRandomize && randomAffect.scaleY)
                    tgt.target.scaleY = randomLimits.syMin + frand() * (randomLimits.syMax - randomLimits.syMin);
                obj.scaleY += (tgt.target.scaleY - obj.scaleY) * randomLerpSpeed;
                
                // colorTop
                if (shouldRandomize && randomAffect.colorTop) for (int c = 0; c < 3; ++c) {
                    ((float*)&tgt.target.colorTop)[c] = randomLimits.colorMin + frand() * (randomLimits.colorMax - randomLimits.colorMin);
                }
                for (int c = 0; c < 3; ++c) {
                    ((float*)&obj.colorTop)[c] += (((float*)&tgt.target.colorTop)[c] - ((float*)&obj.colorTop)[c]) * randomLerpSpeed;
                }
                
                // colorLeft
                if (shouldRandomize && randomAffect.colorLeft) for (int c = 0; c < 3; ++c) {
                    ((float*)&tgt.target.colorLeft)[c] = randomLimits.colorMin + frand() * (randomLimits.colorMax - randomLimits.colorMin);
                }
                for (int c = 0; c < 3; ++c) {
                    ((float*)&obj.colorLeft)[c] += (((float*)&tgt.target.colorLeft)[c] - ((float*)&obj.colorLeft)[c]) * randomLerpSpeed;
                }
                
                // colorRight
                if (shouldRandomize && randomAffect.colorRight) for (int c = 0; c < 3; ++c) {
                    ((float*)&tgt.target.colorRight)[c] = randomLimits.colorMin + frand() * (randomLimits.colorMax - randomLimits.colorMin);
                }
                for (int c = 0; c < 3; ++c) {
                    ((float*)&obj.colorRight)[c] += (((float*)&tgt.target.colorRight)[c] - ((float*)&obj.colorRight)[c]) * randomLerpSpeed;
                }
                
                // groupAngle para cada grupo
                static float tgtGroupAngle[3] = {groups[0].groupAngle, groups[1].groupAngle, groups[2].groupAngle};
                if (shouldRandomize && randomAffect.groupAngle) {
                    tgtGroupAngle[g] = frand() * 2.0f * 3.14159265f;
                }
                groups[g].groupAngle += (tgtGroupAngle[g] - groups[g].groupAngle) * randomLerpSpeed;
                
                // Randomizar cantidad de objetos por grupo
                static int tgtNumObjects[3] = {groups[0].numObjects, groups[1].numObjects, groups[2].numObjects};
                if (shouldRandomize) {
                    if (randomAffect.numCenter && g == 0) {
                        tgtNumObjects[g] = randomLimits.numCenterMin + rand() % (randomLimits.numCenterMax - randomLimits.numCenterMin + 1);
                    } else if (randomAffect.numRight && g == 1) {
                        tgtNumObjects[g] = randomLimits.numRightMin + rand() % (randomLimits.numRightMax - randomLimits.numRightMin + 1);
                    } else if (randomAffect.numLeft && g == 2) {
                        tgtNumObjects[g] = randomLimits.numLeftMin + rand() % (randomLimits.numLeftMax - randomLimits.numLeftMin + 1);
                    }
                }
                groups[g].numObjects += (int)((tgtNumObjects[g] - groups[g].numObjects) * randomLerpSpeed + 0.5f);
                groups[g].numObjects = std::max(0, std::min(100, groups[g].numObjects)); // Limitar a 0-100
            }
            // Si cambió el tamaño, los colores o la figura, recrear el shape
            float curColorTop[3] = {obj.colorTop.x, obj.colorTop.y, obj.colorTop.z};
            float curColorLeft[3] = {obj.colorLeft.x, obj.colorLeft.y, obj.colorLeft.z};
            float curColorRight[3] = {obj.colorRight.x, obj.colorRight.y, obj.colorRight.z};
            bool colorChanged = false;
            for (int i = 0; i < 3; ++i) {
                if (curColorTop[i] != colorTopArr[i] || curColorLeft[i] != colorLeftArr[i] || curColorRight[i] != colorRightArr[i]) {
                    colorChanged = true;
                    break;
                }
            }
            bool shapeChanged = false;
            static int prevShapeType = 0;
            int actualSegments = (obj.shapeType == 0) ? 3 : (obj.shapeType == 1) ? 4 : obj.nSegments;
            if (obj.shapeType != prevShapeType) shapeChanged = true;
            bool fractalChanged = false;
            static bool prevFractalMode = false;
            if (fractalMode != prevFractalMode) fractalChanged = true;
            
            // Para fractales, regenerar cada frame para la animación
            bool shouldRegenerate = obj.triSize != prevSize || colorChanged || shapeChanged || fractalChanged;
            if (fractalMode) shouldRegenerate = true; // Siempre regenerar fractales
            
            if (shouldRegenerate) {
                glDeleteVertexArrays(1, &VAO);
                glDeleteBuffers(1, &VBO);
                // Antes de crear el shape, si onlyRGB está activo, forzar colores a RGB puros
                if (onlyRGB) {
                    curColorTop[0] = 1.0f; curColorTop[1] = 0.0f; curColorTop[2] = 0.0f;
                    curColorLeft[0] = 0.0f; curColorLeft[1] = 1.0f; curColorLeft[2] = 0.0f;
                    curColorRight[0] = 0.0f; curColorRight[1] = 0.0f; curColorRight[2] = 1.0f;
                }
                
                if (fractalMode) {
                    // Verificar que el shapeType sea válido para fractales (0-4, todos soportados ahora)
                    int fractalShapeType = (obj.shapeType >= 0 && obj.shapeType <= 4) ? obj.shapeType : 0;
                    createFractal(VAO, VBO, fractalShapeType, obj.triSize, curColorTop, curColorLeft, curColorRight, fractalDepth, currentTime);
                } else {
                    createShape(VAO, VBO, obj.shapeType, obj.triSize, curColorTop, curColorLeft, curColorRight, actualSegments);
                }
                
                prevSize = obj.triSize;
                for (int i = 0; i < 3; ++i) {
                    colorTopArr[i] = curColorTop[i];
                    colorLeftArr[i] = curColorLeft[i];
                    colorRightArr[i] = curColorRight[i];
                }
                prevShapeType = obj.shapeType;
                prevFractalMode = fractalMode;
            }
        }

        // Randomización de cantidad de figuras por grupo
        static int tgtNum[3] = {groups[0].numObjects, groups[1].numObjects, groups[2].numObjects};
        for (int g = 0; g < 3; ++g) {
            // Randomizar cantidad de figuras por grupo
            bool affect = (g == 0) ? randomAffect.numCenter : (g == 1) ? randomAffect.numRight : randomAffect.numLeft;
            int min = (g == 0) ? randomLimits.numCenterMin : (g == 1) ? randomLimits.numRightMin : randomLimits.numLeftMin;
            int max = (g == 0) ? randomLimits.numCenterMax : (g == 1) ? randomLimits.numRightMax : randomLimits.numLeftMax;
            if (randomize && affect) {
                if (groups[g].numObjects == tgtNum[g]) {
                    tgtNum[g] = min + rand() % (max - min + 1);
                }
                groups[g].numObjects += (int)((tgtNum[g] - groups[g].numObjects) * randomLerpSpeed + 0.5f);
                // Asegurar que no sea negativo
                if (groups[g].numObjects < 0) groups[g].numObjects = 0;
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

        // 3. Randomización y animación de separación de grupos
        if (randomize && randomizeGroupSeparation) {
            if (fabs(groupSeparation - targetGroupSeparation) < 0.01f) {
                targetGroupSeparation = frand() * 2.0f; // entre 0 y 2
            }
            groupSeparation += (targetGroupSeparation - groupSeparation) * randomLerpSpeed;
        }

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

        glClearColor(0.0f, 0.0f, 0.0f, 1.0f); // Fondo negro
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glUseProgram(shaderProgram);
        GLint angleLoc = glGetUniformLocation(shaderProgram, "uAngle");
        float aspect = (float)width / (float)height;
        GLint aspectLoc = glGetUniformLocation(shaderProgram, "uAspect");
        GLint translateLoc = glGetUniformLocation(shaderProgram, "uTranslate");
        GLint scaleLoc = glGetUniformLocation(shaderProgram, "uScale");
        glUniform1f(aspectLoc, aspect);
        for (int g = 0; g < 3; ++g) {
            VisualObjectParams& obj = groups[g].objects[0];
            float baseX = (g == 0) ? 0.0f : (g == 1) ? groupSeparation : -groupSeparation;
            glUniform2f(scaleLoc, obj.scaleX, obj.scaleY);
            glUniform1f(angleLoc, obj.angle);
            glBindVertexArray(VAO);
            for (int i = 0; i < groups[g].numObjects; ++i) {
                float theta = (2.0f * 3.14159265f * i) / std::max(1, groups[g].numObjects) + groups[g].groupAngle;
                float r = 0.5f;
                float tx = baseX + obj.translateX + r * cos(theta);
                float ty = obj.translateY + r * sin(theta);
                glUniform2f(translateLoc, tx, ty);
                if (fractalMode) {
                    // Para fractales, usar GL_TRIANGLES ya que createFractal genera triángulos
                    // Usar un número más conservador de vértices
                    glDrawArrays(GL_TRIANGLES, 0, 3000); // Suficientes vértices para el fractal
                } else {
                    if (obj.shapeType == 0) glDrawArrays(GL_TRIANGLES, 0, 3);
                    else if (obj.shapeType == 1) glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
                    else if (obj.shapeType == 2) glDrawArrays(GL_TRIANGLE_FAN, 0, 130);
                    else if (obj.shapeType == 3) glDrawArrays(GL_LINES, 0, 2);
                    else if (obj.shapeType == 4) glDrawArrays(GL_LINES, 0, 12);
                }
            }
            glBindVertexArray(0);
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