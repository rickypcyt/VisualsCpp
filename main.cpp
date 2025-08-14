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
#include "audio_capture.h"
#include "src/audio_capture.h"
#include "src/fft_utils.h"

// Add shader sources - OPTIMIZED VERSION
const char* vertexShaderSource = R"(
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;
layout(location = 2) in vec2 aOffset;
layout(location = 3) in float aAngle;
layout(location = 4) in vec2 aScale;
out vec3 vColor;
uniform float uAspect;
uniform float uTime;
void main() {
    float s = sin(aAngle);
    float c = cos(aAngle);
    mat2 rot = mat2(c, -s, s, c);
    vec2 pos = rot * (aPos.xy * aScale) + aOffset;
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

// OPTIMIZATION: Instanced rendering structures
struct InstanceData {
    float offsetX, offsetY;
    float angle;
    float scaleX, scaleY;
};

// OPTIMIZATION: VBO caching system
struct CachedVBO {
    GLuint VAO = 0;
    GLuint VBO = 0;
    GLuint instanceVBO = 0;
    int shapeType = -1;
    float size = 0.0f;
    float colors[9] = {0}; // 3 colors * 3 components
    int nSegments = 0;
    bool fractalMode = false;
    float fractalDepth = 0.0f;
    std::vector<InstanceData> instances;
    bool dirty = true;
};

// OPTIMIZATION: Global VBO cache
std::vector<CachedVBO> vboCache;
const int MAX_CACHED_VBOS = 10;

// OPTIMIZATION: Batch rendering
const int MAX_INSTANCES_PER_BATCH = 1000;
std::vector<InstanceData> instanceBuffer;

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

// Move MAX_OBJECTS and VisualGroup to file scope
const int MAX_OBJECTS = 30;
struct VisualGroup {
    std::vector<VisualObjectParams> objects;
    std::vector<VisualObjectTargets> targets;
    int numObjects = 1;
    float groupAngle = 0.0f;
};

// OPTIMIZATION: VBO caching functions
CachedVBO* findOrCreateCachedVBO(int shapeType, float size, float colors[9], int nSegments, bool fractalMode, float fractalDepth) {
    // Buscar VBO existente
    for (auto& cached : vboCache) {
        if (cached.shapeType == shapeType && 
            cached.size == size && 
            cached.nSegments == nSegments &&
            cached.fractalMode == fractalMode &&
            cached.fractalDepth == fractalDepth) {
            bool colorsMatch = true;
            for (int i = 0; i < 9; ++i) {
                if (fabs(cached.colors[i] - colors[i]) > 0.001f) {
                    colorsMatch = false;
                    break;
                }
            }
            if (colorsMatch) {
                return &cached;
            }
        }
    }
    
    // Crear nuevo VBO si no existe
    if (vboCache.size() >= MAX_CACHED_VBOS) {
        // Limpiar el VBO más antiguo
        auto& oldest = vboCache[0];
        if (oldest.VAO) glDeleteVertexArrays(1, &oldest.VAO);
        if (oldest.VBO) glDeleteBuffers(1, &oldest.VBO);
        if (oldest.instanceVBO) glDeleteBuffers(1, &oldest.instanceVBO);
        vboCache.erase(vboCache.begin());
    }
    
    CachedVBO newCached;
    newCached.shapeType = shapeType;
    newCached.size = size;
    newCached.nSegments = nSegments;
    newCached.fractalMode = fractalMode;
    newCached.fractalDepth = fractalDepth;
    for (int i = 0; i < 9; ++i) {
        newCached.colors[i] = colors[i];
    }
    
    // Crear VAO y VBO
    if (fractalMode) {
        createFractal(newCached.VAO, newCached.VBO, shapeType, size, colors, colors+3, colors+6, fractalDepth, 0.0f);
    } else {
        createShape(newCached.VAO, newCached.VBO, shapeType, size, colors, colors+3, colors+6, nSegments);
    }
    
    // Crear VBO para instancing
    glGenBuffers(1, &newCached.instanceVBO);
    glBindVertexArray(newCached.VAO);
    glBindBuffer(GL_ARRAY_BUFFER, newCached.instanceVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(InstanceData) * MAX_INSTANCES_PER_BATCH, nullptr, GL_DYNAMIC_DRAW);
    
    // Configurar atributos de instancia
    glEnableVertexAttribArray(2); // offset
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(InstanceData), (void*)offsetof(InstanceData, offsetX));
    glVertexAttribDivisor(2, 1);
    
    glEnableVertexAttribArray(3); // angle
    glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, sizeof(InstanceData), (void*)offsetof(InstanceData, angle));
    glVertexAttribDivisor(3, 1);
    
    glEnableVertexAttribArray(4); // scale
    glVertexAttribPointer(4, 2, GL_FLOAT, GL_FALSE, sizeof(InstanceData), (void*)offsetof(InstanceData, scaleX));
    glVertexAttribDivisor(4, 1);
    
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    
    vboCache.push_back(newCached);
    return &vboCache.back();
}

// OPTIMIZATION: Batch rendering function
void renderBatch(CachedVBO* cached, const std::vector<InstanceData>& instances, GLuint shaderProgram, float aspect) {
    if (!cached || instances.empty()) return;
    
    // OPTIMIZATION: Set uniforms once per batch
    glUseProgram(shaderProgram);
    static float lastAspect = -1.0f;
    if (lastAspect != aspect) {
        glUniform1f(glGetUniformLocation(shaderProgram, "uAspect"), aspect);
        lastAspect = aspect;
    }
    
    // OPTIMIZATION: Only update time uniform if needed (every 16ms for 60fps)
    static float lastTimeUpdate = 0.0f;
    float currentTime = (float)glfwGetTime();
    if (currentTime - lastTimeUpdate > 0.016f) {
        glUniform1f(glGetUniformLocation(shaderProgram, "uTime"), currentTime);
        lastTimeUpdate = currentTime;
    }
    
    glBindVertexArray(cached->VAO);
    
    // OPTIMIZATION: Render in larger batches for better GPU utilization
    const int OPTIMAL_BATCH_SIZE = 500; // Increased from 1000 for better balance
    
    for (size_t i = 0; i < instances.size(); i += OPTIMAL_BATCH_SIZE) {
        size_t batchSize = std::min(OPTIMAL_BATCH_SIZE, (int)(instances.size() - i));
        
        // OPTIMIZATION: Use glBufferSubData more efficiently
        glBindBuffer(GL_ARRAY_BUFFER, cached->instanceVBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(InstanceData) * batchSize, &instances[i]);
        
        // OPTIMIZATION: Pre-calculate vertex counts
        int vertexCount = 0;
        if (cached->shapeType == 0) vertexCount = 3;
        else if (cached->shapeType == 1) vertexCount = 4;
        else if (cached->shapeType == 2) vertexCount = cached->nSegments + 2;
        else if (cached->shapeType == 3) vertexCount = 2;
        else if (cached->shapeType == 4) vertexCount = 12;
        
        // OPTIMIZATION: Use instanced rendering for all shapes
        if (cached->fractalMode) {
            glDrawArraysInstanced(GL_TRIANGLES, 0, 3000, batchSize);
        } else {
            GLenum drawMode = GL_TRIANGLES;
            if (cached->shapeType == 1) drawMode = GL_TRIANGLE_STRIP;
            else if (cached->shapeType == 2) drawMode = GL_TRIANGLE_FAN;
            else if (cached->shapeType == 3 || cached->shapeType == 4) drawMode = GL_LINES;
            
            glDrawArraysInstanced(drawMode, 0, vertexCount, batchSize);
        }
    }
    
    glBindVertexArray(0);
}

// OPTIMIZATION: Pre-allocate instance buffer
void prepareInstanceBuffer() {
    instanceBuffer.reserve(MAX_INSTANCES_PER_BATCH * 3); // Para 3 grupos
}

// AUDIO REACTIVE SYSTEM: Advanced control structures
struct AudioReactiveControl {
    bool enabled = false;
    float sensitivity = 1.0f;
    float minValue = 0.0f;
    float maxValue = 1.0f;
    float smoothing = 0.1f;
    float currentValue = 0.0f;
    float targetValue = 0.0f;
};

struct AudioReactiveGroup {
    // Frequency ranges
    AudioReactiveControl bass;      // 20-150 Hz
    AudioReactiveControl lowMid;    // 150-400 Hz
    AudioReactiveControl mid;       // 400-2000 Hz
    AudioReactiveControl highMid;   // 2000-6000 Hz
    AudioReactiveControl treble;    // 6000-20000 Hz
    
    // Parameters that can be controlled
    AudioReactiveControl size;
    AudioReactiveControl rotation;
    AudioReactiveControl angle;
    AudioReactiveControl translateX;
    AudioReactiveControl translateY;
    AudioReactiveControl scaleX;
    AudioReactiveControl scaleY;
    AudioReactiveControl colorIntensity;
    AudioReactiveControl groupAngle;
    AudioReactiveControl numObjects;
    
    // Mix presets
    bool useBassMix = false;
    bool useMidMix = false;
    bool useTrebleMix = false;
    bool useFullSpectrumMix = false;
};

// Audio reactive groups for each visual group
AudioReactiveGroup audioGroups[3]; // 0: center, 1: right, 2: left

// Audio reactive presets
struct AudioPreset {
    std::string name;
    std::vector<bool> enabledControls;
    std::vector<float> sensitivities;
    std::vector<bool> frequencyMixes;
};

std::vector<AudioPreset> audioPresets = {
    {"Bass Dominant", {true,false,false,false,false, true,true,false,false,false,false,true,false,false}, 
     {2.0f,1.0f,1.0f,1.0f,1.0f, 1.5f,1.5f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f},
     {true,false,false,false,false}},
    {"Mid Focus", {false,false,true,false,false, true,false,true,true,false,false,false,true,false},
     {1.0f,1.0f,2.0f,1.0f,1.0f, 1.0f,1.0f,1.5f,1.5f,1.0f,1.0f,1.0f,1.0f,1.0f},
     {false,false,true,false,false}},
    {"Treble Energy", {false,false,false,false,true, false,true,false,false,true,true,false,false,true},
     {1.0f,1.0f,1.0f,1.0f,2.0f, 1.0f,1.0f,1.0f,1.0f,1.5f,1.5f,1.0f,1.0f,1.0f},
     {false,false,false,false,true}},
    {"Full Spectrum", {true,true,true,true,true, true,true,true,true,true,true,true,true,true},
     {1.0f,1.0f,1.0f,1.0f,1.0f, 1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f},
     {true,true,true,true,true}},
    {"Pulse Mode", {true,false,false,false,false, true,false,false,false,false,false,true,false,false},
     {3.0f,1.0f,1.0f,1.0f,1.0f, 2.0f,1.0f,1.0f,1.0f,1.0f,1.0f,2.0f,1.0f,1.0f},
     {true,false,false,false,false}},
    {"Wave Mode", {false,false,true,false,false, false,true,true,true,false,false,false,true,false},
     {1.0f,1.0f,2.5f,1.0f,1.0f, 1.0f,1.5f,2.0f,2.0f,1.0f,1.0f,1.0f,1.5f,1.0f},
     {false,false,true,false,false}},
    {"Chaos Mode", {true,true,true,true,true, true,true,true,true,true,true,true,true,true},
     {2.0f,2.0f,2.0f,2.0f,2.0f, 2.0f,2.0f,2.0f,2.0f,2.0f,2.0f,2.0f,2.0f,2.0f},
     {true,true,true,true,true}},
    {"Wide Full Range", {true,true,true,true,true, true,true,true,true,true,true,true,true,true},
     {1.5f,1.5f,1.5f,1.5f,1.5f, 1.5f,1.5f,1.5f,1.5f,1.5f,1.5f,1.5f,1.5f,1.5f},
     {true,true,true,true,true}}
};

// Audio analysis variables
struct AudioAnalysis {
    float bass = 0.0f;
    float lowMid = 0.0f;
    float mid = 0.0f;
    float highMid = 0.0f;
    float treble = 0.0f;
    float overall = 0.0f;
    float peak = 0.0f;
    float rms = 0.0f;
};

AudioAnalysis currentAudio;

// AUDIO REACTIVE SYSTEM: Advanced audio analysis
void analyzeAudioSpectrum(const std::vector<float>& spectrum, AudioAnalysis& analysis) {
    if (spectrum.empty()) {
        // Reset analysis to safe values
        analysis.bass = 0.0f;
        analysis.lowMid = 0.0f;
        analysis.mid = 0.0f;
        analysis.highMid = 0.0f;
        analysis.treble = 0.0f;
        analysis.overall = 0.0f;
        analysis.peak = 0.0f;
        analysis.rms = 0.0f;
        return;
    }
    
    int n = spectrum.size();
    if (n <= 0) return;
    
    float sampleRate = 48000.0f;
    float freqPerBin = sampleRate / (2.0f * n);
    
    // Prevent division by zero
    if (freqPerBin <= 0.0f) freqPerBin = 1.0f;
    
    // Frequency ranges with safety checks
    int bassStart = std::max(0, (int)(20.0f / freqPerBin));
    int bassEnd = std::min(n - 1, (int)(150.0f / freqPerBin));
    int lowMidStart = std::max(bassEnd, (int)(150.0f / freqPerBin));
    int lowMidEnd = std::min(n - 1, (int)(400.0f / freqPerBin));
    int midStart = std::max(lowMidEnd, (int)(400.0f / freqPerBin));
    int midEnd = std::min(n - 1, (int)(2000.0f / freqPerBin));
    int highMidStart = std::max(midEnd, (int)(2000.0f / freqPerBin));
    int highMidEnd = std::min(n - 1, (int)(6000.0f / freqPerBin));
    int trebleStart = std::max(highMidEnd, (int)(6000.0f / freqPerBin));
    int trebleEnd = std::min(n - 1, (int)(20000.0f / freqPerBin));
    
    // Ensure valid ranges
    bassStart = std::max(0, std::min(n-1, bassStart));
    bassEnd = std::max(bassStart, std::min(n-1, bassEnd));
    lowMidStart = std::max(bassEnd, std::min(n-1, lowMidStart));
    lowMidEnd = std::max(lowMidStart, std::min(n-1, lowMidEnd));
    midStart = std::max(lowMidEnd, std::min(n-1, midStart));
    midEnd = std::max(midStart, std::min(n-1, midEnd));
    highMidStart = std::max(midEnd, std::min(n-1, highMidStart));
    highMidEnd = std::max(highMidStart, std::min(n-1, highMidEnd));
    trebleStart = std::max(highMidEnd, std::min(n-1, trebleStart));
    trebleEnd = std::max(trebleStart, std::min(n-1, trebleEnd));
    
    // Calculate averages for each frequency range
    float bassSum = 0.0f, lowMidSum = 0.0f, midSum = 0.0f, highMidSum = 0.0f, trebleSum = 0.0f;
    float overallSum = 0.0f;
    float peakValue = 0.0f;
    
    for (int i = 0; i < n; ++i) {
        float value = spectrum[i];
        // Check for NaN or infinite values
        if (std::isnan(value) || std::isinf(value)) {
            value = 0.0f;
        }
        
        overallSum += value;
        peakValue = std::max(peakValue, value);
        
        if (i >= bassStart && i <= bassEnd) bassSum += value;
        if (i >= lowMidStart && i <= lowMidEnd) lowMidSum += value;
        if (i >= midStart && i <= midEnd) midSum += value;
        if (i >= highMidStart && i <= highMidEnd) highMidSum += value;
        if (i >= trebleStart && i <= trebleEnd) trebleSum += value;
    }
    
    // Normalize by range size with safety checks
    int bassCount = std::max(1, bassEnd - bassStart + 1);
    int lowMidCount = std::max(1, lowMidEnd - lowMidStart + 1);
    int midCount = std::max(1, midEnd - midStart + 1);
    int highMidCount = std::max(1, highMidEnd - highMidStart + 1);
    int trebleCount = std::max(1, trebleEnd - trebleStart + 1);
    
    analysis.bass = bassSum / bassCount;
    analysis.lowMid = lowMidSum / lowMidCount;
    analysis.mid = midSum / midCount;
    analysis.highMid = highMidSum / highMidCount;
    analysis.treble = trebleSum / trebleCount;
    analysis.overall = overallSum / n;
    analysis.peak = peakValue;
    
    // Safe RMS calculation
    if (overallSum > 0.0f && n > 0) {
        analysis.rms = sqrt(overallSum / n);
    } else {
        analysis.rms = 0.0f;
    }
    
    // Final safety check for NaN values
    if (std::isnan(analysis.bass)) analysis.bass = 0.0f;
    if (std::isnan(analysis.lowMid)) analysis.lowMid = 0.0f;
    if (std::isnan(analysis.mid)) analysis.mid = 0.0f;
    if (std::isnan(analysis.highMid)) analysis.highMid = 0.0f;
    if (std::isnan(analysis.treble)) analysis.treble = 0.0f;
    if (std::isnan(analysis.overall)) analysis.overall = 0.0f;
    if (std::isnan(analysis.peak)) analysis.peak = 0.0f;
    if (std::isnan(analysis.rms)) analysis.rms = 0.0f;
}

// AUDIO REACTIVE SYSTEM: Apply audio control to parameters
void applyAudioControl(AudioReactiveControl& control, float audioValue, float deltaTime) {
    if (!control.enabled) return;
    
    // Safety checks for input values
    if (std::isnan(audioValue) || std::isinf(audioValue)) {
        audioValue = 0.0f;
    }
    
    // Prevent division by zero or very small deltaTime
    if (deltaTime <= 0.001f) {
        deltaTime = 0.016f; // Use 60fps as fallback
    }
    
    // Clamp audio value to reasonable range
    audioValue = std::max(0.0f, std::min(10.0f, audioValue));
    
    // Apply sensitivity and range
    float targetValue = control.minValue + (control.maxValue - control.minValue) * 
                       (audioValue * control.sensitivity);
    
    // Safety check for target value
    if (std::isnan(targetValue) || std::isinf(targetValue)) {
        targetValue = control.minValue;
    }
    
    // Smooth transition with safety checks
    control.targetValue = targetValue;
    float smoothingFactor = control.smoothing / deltaTime;
    
    // Clamp smoothing factor to prevent instability
    smoothingFactor = std::max(0.001f, std::min(1.0f, smoothingFactor));
    
    control.currentValue += (control.targetValue - control.currentValue) * smoothingFactor;
    
    // Final safety check for current value
    if (std::isnan(control.currentValue) || std::isinf(control.currentValue)) {
        control.currentValue = control.minValue;
    }
}

// AUDIO REACTIVE SYSTEM: Apply preset to audio group
void applyAudioPreset(AudioReactiveGroup& group, const AudioPreset& preset) {
    // Apply frequency mix settings
    group.useBassMix = preset.frequencyMixes[0];
    group.useMidMix = preset.frequencyMixes[2];
    group.useTrebleMix = preset.frequencyMixes[4];
    group.useFullSpectrumMix = preset.frequencyMixes[0] && preset.frequencyMixes[1] && 
                               preset.frequencyMixes[2] && preset.frequencyMixes[3] && 
                               preset.frequencyMixes[4];
    
    // Apply control settings
    group.bass.enabled = preset.enabledControls[0];
    group.lowMid.enabled = preset.enabledControls[1];
    group.mid.enabled = preset.enabledControls[2];
    group.highMid.enabled = preset.enabledControls[3];
    group.treble.enabled = preset.enabledControls[4];
    
    group.size.enabled = preset.enabledControls[5];
    group.rotation.enabled = preset.enabledControls[6];
    group.angle.enabled = preset.enabledControls[7];
    group.translateX.enabled = preset.enabledControls[8];
    group.translateY.enabled = preset.enabledControls[9];
    group.scaleX.enabled = preset.enabledControls[10];
    group.scaleY.enabled = preset.enabledControls[11];
    group.colorIntensity.enabled = preset.enabledControls[12];
    group.groupAngle.enabled = preset.enabledControls[13];
    
    // Apply sensitivities
    group.bass.sensitivity = preset.sensitivities[0];
    group.lowMid.sensitivity = preset.sensitivities[1];
    group.mid.sensitivity = preset.sensitivities[2];
    group.highMid.sensitivity = preset.sensitivities[3];
    group.treble.sensitivity = preset.sensitivities[4];
    
    group.size.sensitivity = preset.sensitivities[5];
    group.rotation.sensitivity = preset.sensitivities[6];
    group.angle.sensitivity = preset.sensitivities[7];
    group.translateX.sensitivity = preset.sensitivities[8];
    group.translateY.sensitivity = preset.sensitivities[9];
    group.scaleX.sensitivity = preset.sensitivities[10];
    group.scaleY.sensitivity = preset.sensitivities[11];
    group.colorIntensity.sensitivity = preset.sensitivities[12];
    group.groupAngle.sensitivity = preset.sensitivities[13];
}

// UI VISIBILITY CONTROL SYSTEM
struct UIVisibility {
    bool showMainControls = true;
    bool showAdvancedOptions = true;
    bool showRandomization = true;
    bool showSystemMonitor = true;
    bool showAudioControl = true;
    bool showGlobalOptions = true;
    bool showAudioGraph = true; // Nueva ventana de gráfico de audio
    bool showAudioTestMode = true; // Modo de prueba de audio
    bool showPresets = true; // Nueva ventana de presets
    bool showAll = true; // Master control
};

UIVisibility uiVisibility;

// AUDIO GRAPH SYSTEM: Para medir latencia y optimizar
struct AudioGraphData {
    static const int MAX_SAMPLES = 200;
    std::vector<float> audioLevels;
    std::vector<float> timestamps;
    std::vector<float> latencies;
    float lastUpdateTime = 0.0f;
    float averageLatency = 0.0f;
    float minLatency = 9999.0f;
    float maxLatency = 0.0f;
    int frameCount = 0;
    float fps = 0.0f;
    
    void addSample(float level, float timestamp, float latency) {
        audioLevels.push_back(level);
        timestamps.push_back(timestamp);
        latencies.push_back(latency);
        
        // Mantener solo los últimos MAX_SAMPLES
        if (audioLevels.size() > MAX_SAMPLES) {
            audioLevels.erase(audioLevels.begin());
            timestamps.erase(timestamps.begin());
            latencies.erase(latencies.begin());
        }
        
        // Actualizar estadísticas de latencia
        if (latency > 0.0f) {
            minLatency = std::min(minLatency, latency);
            maxLatency = std::max(maxLatency, latency);
            
            // Calcular latencia promedio
            float sum = 0.0f;
            int count = 0;
            for (float lat : latencies) {
                if (lat > 0.0f) {
                    sum += lat;
                    count++;
                }
            }
            if (count > 0) {
                averageLatency = sum / count;
            }
        }
    }
    
    void updateFPS(float currentTime) {
        frameCount++;
        if (currentTime - lastUpdateTime >= 1.0f) {
            fps = frameCount;
            frameCount = 0;
            lastUpdateTime = currentTime;
        }
    }
    
    void clear() {
        audioLevels.clear();
        timestamps.clear();
        latencies.clear();
        averageLatency = 0.0f;
        minLatency = 9999.0f;
        maxLatency = 0.0f;
    }
};

AudioGraphData audioGraph;

// AUDIO TEST MODE: Para probar audio reactivo fácilmente
struct AudioTestMode {
    bool enabled = false;
    bool testColorEnabled = true;
    bool testSizeEnabled = true;
    bool testRotationEnabled = true;
    bool testPositionEnabled = true;
    bool testQuantityEnabled = false;
    
    // Frecuencias específicas para test
    float bassTest = 0.0f;
    float midTest = 0.0f;
    float trebleTest = 0.0f;
    float overallTest = 0.0f;
    
    // Controles manuales para simular audio
    float manualBass = 0.5f;
    float manualMid = 0.5f;
    float manualTreble = 0.5f;
    bool useManualValues = false;
    
    // Objeto de prueba
    float testSize = 0.5f;
    float testRotation = 0.0f;
    ImVec4 testColor = ImVec4(1.0f, 0.0f, 0.0f, 1.0f);
    float testPosX = 0.0f;
    float testPosY = 0.0f;
    int testQuantity = 1;
    
    void updateFromAudio(const AudioAnalysis& audio) {
        if (useManualValues) {
            bassTest = manualBass;
            midTest = manualMid;
            trebleTest = manualTreble;
            overallTest = (manualBass + manualMid + manualTreble) / 3.0f;
        } else {
            bassTest = audio.bass;
            midTest = audio.mid;
            trebleTest = audio.treble;
            overallTest = audio.overall;
        }
        
        // Aplicar efectos de audio al objeto de prueba
        if (testSizeEnabled) {
            testSize = 0.2f + overallTest * 1.5f; // 0.2 a 1.7
        }
        
        if (testRotationEnabled) {
            testRotation = midTest * 360.0f; // 0 a 360 grados
        }
        
        if (testColorEnabled) {
            testColor.x = bassTest;      // Rojo = Bass
            testColor.y = midTest;       // Verde = Mid
            testColor.z = trebleTest;    // Azul = Treble
        }
        
        if (testPositionEnabled) {
            testPosX = (bassTest - 0.5f) * 2.0f;  // -1 a 1
            testPosY = (trebleTest - 0.5f) * 2.0f; // -1 a 1
        }
        
        if (testQuantityEnabled) {
            testQuantity = 1 + (int)(overallTest * 10.0f); // 1 a 11 objetos
        }
    }
    
    void reset() {
        testSize = 0.5f;
        testRotation = 0.0f;
        testColor = ImVec4(1.0f, 0.0f, 0.0f, 1.0f);
        testPosX = 0.0f;
        testPosY = 0.0f;
        testQuantity = 1;
        manualBass = 0.5f;
        manualMid = 0.5f;
        manualTreble = 0.5f;
    }
};

AudioTestMode audioTestMode;

// ANIMATION PRESETS: Predefined animation configurations
struct AnimationPreset {
    std::string name;
    std::string description;
    
    // Group configurations
    struct GroupConfig {
        int shapeType;
        int numObjects;
        float triSize;
        float rotationSpeed;
        float groupAngle;
        float translateX, translateY;
        float scaleX, scaleY;
        ImVec4 colorTop, colorLeft, colorRight;
        int nSegments;
        bool fractalMode;
        float fractalDepth;
    };
    
    GroupConfig center, right, left;
    
    // Global settings
    float groupSeparation;
    bool autoRotate;
    bool randomize;
    bool audioReactive;
    float bpm;
    
    // Audio preset to apply
    int audioPresetIndex;
    
    // AnimationPreset::apply - mark as const and remove global assignments
    void apply(VisualGroup groups[3], bool& autoRotate, bool& randomize, 
               bool& audioReactive, float& bpm, float& groupSeparation,
               RandomLimits& randomLimits, RandomAffectFlags& randomAffect) const {
        // Apply center group
        groups[0].objects[0].shapeType = center.shapeType;
        groups[0].numObjects = center.numObjects;
        groups[0].objects[0].triSize = center.triSize;
        groups[0].objects[0].rotationSpeed = center.rotationSpeed;
        groups[0].groupAngle = center.groupAngle;
        groups[0].objects[0].translateX = center.translateX;
        groups[0].objects[0].translateY = center.translateY;
        groups[0].objects[0].scaleX = center.scaleX;
        groups[0].objects[0].scaleY = center.scaleY;
        groups[0].objects[0].colorTop = center.colorTop;
        groups[0].objects[0].colorLeft = center.colorLeft;
        groups[0].objects[0].colorRight = center.colorRight;
        groups[0].objects[0].nSegments = center.nSegments;
        
        // Apply right group
        groups[1].objects[0].shapeType = right.shapeType;
        groups[1].numObjects = right.numObjects;
        groups[1].objects[0].triSize = right.triSize;
        groups[1].objects[0].rotationSpeed = right.rotationSpeed;
        groups[1].groupAngle = right.groupAngle;
        groups[1].objects[0].translateX = right.translateX;
        groups[1].objects[0].translateY = right.translateY;
        groups[1].objects[0].scaleX = right.scaleX;
        groups[1].objects[0].scaleY = right.scaleY;
        groups[1].objects[0].colorTop = right.colorTop;
        groups[1].objects[0].colorLeft = right.colorLeft;
        groups[1].objects[0].colorRight = right.colorRight;
        groups[1].objects[0].nSegments = right.nSegments;
        
        // Apply left group
        groups[2].objects[0].shapeType = left.shapeType;
        groups[2].numObjects = left.numObjects;
        groups[2].objects[0].triSize = left.triSize;
        groups[2].objects[0].rotationSpeed = left.rotationSpeed;
        groups[2].groupAngle = left.groupAngle;
        groups[2].objects[0].translateX = left.translateX;
        groups[2].objects[0].translateY = left.translateY;
        groups[2].objects[0].scaleX = left.scaleX;
        groups[2].objects[0].scaleY = left.scaleY;
        groups[2].objects[0].colorTop = left.colorTop;
        groups[2].objects[0].colorLeft = left.colorLeft;
        groups[2].objects[0].colorRight = left.colorRight;
        groups[2].objects[0].nSegments = left.nSegments;
        
        // Apply global settings by reference only
        autoRotate = this->autoRotate;
        randomize = this->randomize;
        audioReactive = this->audioReactive;
        bpm = this->bpm;
        groupSeparation = this->groupSeparation;
        
        // Configure random limits based on preset type for better randomization
        if (this->randomize) {
            // Set appropriate random limits based on the preset characteristics
            if (center.fractalMode || right.fractalMode || left.fractalMode) {
                // Fractal presets - more extreme randomization
                randomLimits.sizeMin = 0.1f; randomLimits.sizeMax = 3.0f;
                randomLimits.speedMin = 10.0f; randomLimits.speedMax = 500.0f;
                randomLimits.txMin = -1.5f; randomLimits.txMax = 1.5f;
                randomLimits.tyMin = -1.5f; randomLimits.tyMax = 1.5f;
                randomLimits.sxMin = 0.2f; randomLimits.sxMax = 2.5f;
                randomLimits.syMin = 0.2f; randomLimits.syMax = 2.5f;
            } else if (center.shapeType == SHAPE_LINE || right.shapeType == SHAPE_LINE || left.shapeType == SHAPE_LINE ||
                       center.shapeType == SHAPE_LONG_LINES || right.shapeType == SHAPE_LONG_LINES || left.shapeType == SHAPE_LONG_LINES) {
                // Line presets - moderate randomization
                randomLimits.sizeMin = 0.05f; randomLimits.sizeMax = 2.0f;
                randomLimits.speedMin = 50.0f; randomLimits.speedMax = 300.0f;
                randomLimits.txMin = -1.0f; randomLimits.txMax = 1.0f;
                randomLimits.tyMin = -1.0f; randomLimits.tyMax = 1.0f;
                randomLimits.sxMin = 0.5f; randomLimits.sxMax = 3.0f;
                randomLimits.syMin = 0.1f; randomLimits.syMax = 1.0f;
            } else if (center.shapeType == SHAPE_CIRCLE || right.shapeType == SHAPE_CIRCLE || left.shapeType == SHAPE_CIRCLE) {
                // Circle presets - balanced randomization
                randomLimits.sizeMin = 0.1f; randomLimits.sizeMax = 2.5f;
                randomLimits.speedMin = 20.0f; randomLimits.speedMax = 200.0f;
                randomLimits.txMin = -0.8f; randomLimits.txMax = 0.8f;
                randomLimits.tyMin = -0.8f; randomLimits.tyMax = 0.8f;
                randomLimits.sxMin = 0.3f; randomLimits.sxMax = 2.0f;
                randomLimits.syMin = 0.3f; randomLimits.syMax = 2.0f;
            } else if (this->name.find("Túnel Psicodélico") != std::string::npos) {
                // Tunnel preset - extreme randomization for psychedelic effect
                randomLimits.sizeMin = 0.05f; randomLimits.sizeMax = 4.0f;
                randomLimits.speedMin = 5.0f; randomLimits.speedMax = 800.0f;
                randomLimits.txMin = -2.0f; randomLimits.txMax = 2.0f;
                randomLimits.tyMin = -2.0f; randomLimits.tyMax = 2.0f;
                randomLimits.sxMin = 0.1f; randomLimits.sxMax = 4.0f;
                randomLimits.syMin = 0.1f; randomLimits.syMax = 4.0f;
                randomLimits.segMin = 3; randomLimits.segMax = 128;
            } else {
                // Default randomization for other shapes
                randomLimits.sizeMin = 0.05f; randomLimits.sizeMax = 2.0f;
                randomLimits.speedMin = 30.0f; randomLimits.speedMax = 250.0f;
                randomLimits.txMin = -1.0f; randomLimits.txMax = 1.0f;
                randomLimits.tyMin = -1.0f; randomLimits.tyMax = 1.0f;
                randomLimits.sxMin = 0.2f; randomLimits.sxMax = 2.5f;
                randomLimits.syMin = 0.2f; randomLimits.syMax = 2.5f;
            }
            
            // Enable all randomization flags for maximum variety
            randomAffect.triSize = true;
            randomAffect.rotationSpeed = true;
            randomAffect.angle = true;
            randomAffect.translateX = true;
            randomAffect.translateY = true;
            randomAffect.scaleX = true;
            randomAffect.scaleY = true;
            randomAffect.colorTop = true;
            randomAffect.colorLeft = true;
            randomAffect.colorRight = true;
            randomAffect.shapeType = true;
            randomAffect.nSegments = true;
            randomAffect.groupAngle = true;
            randomAffect.numCenter = true;
            randomAffect.numRight = true;
            randomAffect.numLeft = true;
        }
    }
};

// Predefined animation presets
std::vector<AnimationPreset> animationPresets = {
    {
        "Cilindros 3D",
        "Cilindros rotando en diferentes ejes",
        // Center
        {SHAPE_CIRCLE, 8, 0.3f, 120.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f,
         ImVec4(1.0f, 0.2f, 0.2f, 1.0f), ImVec4(0.8f, 0.1f, 0.1f, 1.0f), ImVec4(0.6f, 0.0f, 0.0f, 1.0f), 32, false, 0.0f},
        // Right
        {SHAPE_CIRCLE, 6, 0.25f, 180.0f, 45.0f, 0.0f, 0.0f, 0.8f, 0.8f,
         ImVec4(0.2f, 1.0f, 0.2f, 1.0f), ImVec4(0.1f, 0.8f, 0.1f, 1.0f), ImVec4(0.0f, 0.6f, 0.0f, 1.0f), 24, false, 0.0f},
        // Left
        {SHAPE_CIRCLE, 6, 0.25f, 150.0f, -45.0f, 0.0f, 0.0f, 0.8f, 0.8f,
         ImVec4(0.2f, 0.2f, 1.0f, 1.0f), ImVec4(0.1f, 0.1f, 0.8f, 1.0f), ImVec4(0.0f, 0.0f, 0.6f, 1.0f), 24, false, 0.0f},
        1.2f, true, true, true, 140.0f, 3 // Full Spectrum audio preset
    },
    
    {
        "Donas 3D",
        "Formaciones de donas con múltiples anillos",
        // Center
        {SHAPE_CIRCLE, 12, 0.2f, 90.0f, 0.0f, 0.0f, 0.0f, 1.2f, 1.2f,
         ImVec4(1.0f, 0.5f, 0.0f, 1.0f), ImVec4(0.8f, 0.4f, 0.0f, 1.0f), ImVec4(0.6f, 0.3f, 0.0f, 1.0f), 48, false, 0.0f},
        // Right
        {SHAPE_CIRCLE, 10, 0.15f, 120.0f, 30.0f, 0.0f, 0.0f, 1.0f, 1.0f,
         ImVec4(0.0f, 1.0f, 0.5f, 1.0f), ImVec4(0.0f, 0.8f, 0.4f, 1.0f), ImVec4(0.0f, 0.6f, 0.3f, 1.0f), 36, false, 0.0f},
        // Left
        {SHAPE_CIRCLE, 10, 0.15f, 100.0f, -30.0f, 0.0f, 0.0f, 1.0f, 1.0f,
         ImVec4(0.5f, 0.0f, 1.0f, 1.0f), ImVec4(0.4f, 0.0f, 0.8f, 1.0f), ImVec4(0.3f, 0.0f, 0.6f, 1.0f), 36, false, 0.0f},
        1.5f, true, true, true, 120.0f, 2 // Mid Focus audio preset
    },
    
    {
        "Fractales Mágicos",
        "Fractales animados con colores psicodélicos",
        // Center
        {SHAPE_TRIANGLE, 5, 0.4f, 60.0f, 0.0f, 0.0f, 0.0f, 1.5f, 1.5f,
         ImVec4(1.0f, 0.0f, 1.0f, 1.0f), ImVec4(0.8f, 0.0f, 0.8f, 1.0f), ImVec4(0.6f, 0.0f, 0.6f, 1.0f), 16, true, 4.0f},
        // Right
        {SHAPE_SQUARE, 4, 0.35f, 80.0f, 60.0f, 0.0f, 0.0f, 1.3f, 1.3f,
         ImVec4(0.0f, 1.0f, 1.0f, 1.0f), ImVec4(0.0f, 0.8f, 0.8f, 1.0f), ImVec4(0.0f, 0.6f, 0.6f, 1.0f), 12, true, 3.5f},
        // Left
        {SHAPE_CIRCLE, 6, 0.3f, 70.0f, -60.0f, 0.0f, 0.0f, 1.4f, 1.4f,
         ImVec4(1.0f, 1.0f, 0.0f, 1.0f), ImVec4(0.8f, 0.8f, 0.0f, 1.0f), ImVec4(0.6f, 0.6f, 0.0f, 1.0f), 20, true, 3.8f},
        1.8f, true, true, true, 100.0f, 6 // Chaos Mode audio preset
    },
    
    {
        "Líneas Energéticas",
        "Líneas dinámicas que fluyen con el audio",
        // Center
        {SHAPE_LINE, 15, 0.1f, 200.0f, 0.0f, 0.0f, 0.0f, 2.0f, 0.5f,
         ImVec4(1.0f, 0.0f, 0.0f, 1.0f), ImVec4(0.8f, 0.0f, 0.0f, 1.0f), ImVec4(0.6f, 0.0f, 0.0f, 1.0f), 2, false, 0.0f},
        // Right
        {SHAPE_LINE, 12, 0.08f, 180.0f, 45.0f, 0.0f, 0.0f, 1.8f, 0.4f,
         ImVec4(0.0f, 1.0f, 0.0f, 1.0f), ImVec4(0.0f, 0.8f, 0.0f, 1.0f), ImVec4(0.0f, 0.6f, 0.0f, 1.0f), 2, false, 0.0f},
        // Left
        {SHAPE_LINE, 12, 0.08f, 160.0f, -45.0f, 0.0f, 0.0f, 1.8f, 0.4f,
         ImVec4(0.0f, 0.0f, 1.0f, 1.0f), ImVec4(0.0f, 0.0f, 0.8f, 1.0f), ImVec4(0.0f, 0.0f, 0.6f, 1.0f), 2, false, 0.0f},
        1.0f, true, true, true, 160.0f, 4 // Full Spectrum audio preset
    },
    
    {
        "Pulso Cósmico",
        "Pulsos rítmicos que expanden y contraen",
        // Center
        {SHAPE_CIRCLE, 20, 0.15f, 45.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f,
         ImVec4(1.0f, 0.3f, 0.7f, 1.0f), ImVec4(0.8f, 0.2f, 0.6f, 1.0f), ImVec4(0.6f, 0.1f, 0.5f, 1.0f), 64, false, 0.0f},
        // Right
        {SHAPE_CIRCLE, 16, 0.12f, 55.0f, 30.0f, 0.0f, 0.0f, 0.9f, 0.9f,
         ImVec4(0.3f, 1.0f, 0.7f, 1.0f), ImVec4(0.2f, 0.8f, 0.6f, 1.0f), ImVec4(0.1f, 0.6f, 0.5f, 1.0f), 48, false, 0.0f},
        // Left
        {SHAPE_CIRCLE, 16, 0.12f, 50.0f, -30.0f, 0.0f, 0.0f, 0.9f, 0.9f,
         ImVec4(0.7f, 0.3f, 1.0f, 1.0f), ImVec4(0.6f, 0.2f, 0.8f, 1.0f), ImVec4(0.5f, 0.1f, 0.6f, 1.0f), 48, false, 0.0f},
        1.3f, true, true, true, 80.0f, 0 // Bass Dominant audio preset
    },
    
    {
        "Espiral Galáctica",
        "Espirales que giran como galaxias",
        // Center
        {SHAPE_TRIANGLE, 25, 0.08f, 120.0f, 0.0f, 0.0f, 0.0f, 1.2f, 1.2f,
         ImVec4(1.0f, 0.8f, 0.0f, 1.0f), ImVec4(0.8f, 0.6f, 0.0f, 1.0f), ImVec4(0.6f, 0.4f, 0.0f, 1.0f), 3, false, 0.0f},
        // Right
        {SHAPE_TRIANGLE, 20, 0.06f, 140.0f, 60.0f, 0.0f, 0.0f, 1.1f, 1.1f,
         ImVec4(0.0f, 0.8f, 1.0f, 1.0f), ImVec4(0.0f, 0.6f, 0.8f, 1.0f), ImVec4(0.0f, 0.4f, 0.6f, 1.0f), 3, false, 0.0f},
        // Left
        {SHAPE_TRIANGLE, 20, 0.06f, 130.0f, -60.0f, 0.0f, 0.0f, 1.1f, 1.1f,
         ImVec4(1.0f, 0.0f, 0.8f, 1.0f), ImVec4(0.8f, 0.0f, 0.6f, 1.0f), ImVec4(0.6f, 0.0f, 0.4f, 1.0f), 3, false, 0.0f},
        1.6f, true, true, true, 110.0f, 5 // Treble Energy audio preset
    },
    
    {
        "Cristales Geométricos",
        "Formaciones cristalinas con geometría perfecta",
        // Center
        {SHAPE_SQUARE, 8, 0.25f, 75.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f,
         ImVec4(0.5f, 1.0f, 0.5f, 1.0f), ImVec4(0.4f, 0.8f, 0.4f, 1.0f), ImVec4(0.3f, 0.6f, 0.3f, 1.0f), 4, true, 2.5f},
        // Right
        {SHAPE_SQUARE, 6, 0.2f, 90.0f, 45.0f, 0.0f, 0.0f, 0.9f, 0.9f,
         ImVec4(0.5f, 0.5f, 1.0f, 1.0f), ImVec4(0.4f, 0.4f, 0.8f, 1.0f), ImVec4(0.3f, 0.3f, 0.6f, 1.0f), 4, true, 2.2f},
        // Left
        {SHAPE_SQUARE, 6, 0.2f, 85.0f, -45.0f, 0.0f, 0.0f, 0.9f, 0.9f,
         ImVec4(1.0f, 0.5f, 0.5f, 1.0f), ImVec4(0.8f, 0.4f, 0.4f, 1.0f), ImVec4(0.6f, 0.3f, 0.3f, 1.0f), 4, true, 2.3f},
        1.4f, true, true, true, 95.0f, 7 // Wide Full Range audio preset
    },
    
    {
        "Líneas Largas Dinámicas",
        "Líneas largas que se extienden y contraen",
        // Center
        {SHAPE_LONG_LINES, 3, 0.05f, 300.0f, 0.0f, 0.0f, 0.0f, 3.0f, 0.3f,
         ImVec4(1.0f, 0.0f, 0.5f, 1.0f), ImVec4(0.8f, 0.0f, 0.4f, 1.0f), ImVec4(0.6f, 0.0f, 0.3f, 1.0f), 12, false, 0.0f},
        // Right
        {SHAPE_LONG_LINES, 2, 0.04f, 250.0f, 60.0f, 0.0f, 0.0f, 2.5f, 0.25f,
         ImVec4(0.0f, 1.0f, 0.5f, 1.0f), ImVec4(0.0f, 0.8f, 0.4f, 1.0f), ImVec4(0.0f, 0.6f, 0.3f, 1.0f), 10, false, 0.0f},
        // Left
        {SHAPE_LONG_LINES, 2, 0.04f, 280.0f, -60.0f, 0.0f, 0.0f, 2.5f, 0.25f,
         ImVec4(0.5f, 0.0f, 1.0f, 1.0f), ImVec4(0.4f, 0.0f, 0.8f, 1.0f), ImVec4(0.3f, 0.0f, 0.6f, 1.0f), 10, false, 0.0f},
        1.2f, true, true, true, 180.0f, 5 // Treble Energy audio preset
    },
    
    {
        "Vórtice Cuántico",
        "Vórtices que giran en diferentes direcciones",
        // Center
        {SHAPE_TRIANGLE, 30, 0.06f, 200.0f, 0.0f, 0.0f, 0.0f, 0.8f, 0.8f,
         ImVec4(1.0f, 0.0f, 0.0f, 1.0f), ImVec4(0.8f, 0.0f, 0.0f, 1.0f), ImVec4(0.6f, 0.0f, 0.0f, 1.0f), 3, true, 3.0f},
        // Right
        {SHAPE_TRIANGLE, 25, 0.05f, 220.0f, 90.0f, 0.0f, 0.0f, 0.7f, 0.7f,
         ImVec4(0.0f, 1.0f, 0.0f, 1.0f), ImVec4(0.0f, 0.8f, 0.0f, 1.0f), ImVec4(0.0f, 0.6f, 0.0f, 1.0f), 3, true, 2.8f},
        // Left
        {SHAPE_TRIANGLE, 25, 0.05f, 180.0f, -90.0f, 0.0f, 0.0f, 0.7f, 0.7f,
         ImVec4(0.0f, 0.0f, 1.0f, 1.0f), ImVec4(0.0f, 0.0f, 0.8f, 1.0f), ImVec4(0.0f, 0.0f, 0.6f, 1.0f), 3, true, 2.8f},
        1.8f, true, true, true, 150.0f, 6 // Chaos Mode audio preset
    },
    
    {
        "Pulso Neural",
        "Pulsos que simulan actividad neuronal",
        // Center
        {SHAPE_CIRCLE, 40, 0.08f, 30.0f, 0.0f, 0.0f, 0.0f, 0.6f, 0.6f,
         ImVec4(1.0f, 0.2f, 0.8f, 1.0f), ImVec4(0.8f, 0.1f, 0.6f, 1.0f), ImVec4(0.6f, 0.0f, 0.4f, 1.0f), 16, false, 0.0f},
        // Right
        {SHAPE_CIRCLE, 35, 0.07f, 35.0f, 45.0f, 0.0f, 0.0f, 0.5f, 0.5f,
         ImVec4(0.2f, 1.0f, 0.8f, 1.0f), ImVec4(0.1f, 0.8f, 0.6f, 1.0f), ImVec4(0.0f, 0.6f, 0.4f, 1.0f), 14, false, 0.0f},
        // Left
        {SHAPE_CIRCLE, 35, 0.07f, 32.0f, -45.0f, 0.0f, 0.0f, 0.5f, 0.5f,
         ImVec4(0.8f, 0.2f, 1.0f, 1.0f), ImVec4(0.6f, 0.1f, 0.8f, 1.0f), ImVec4(0.4f, 0.0f, 0.6f, 1.0f), 14, false, 0.0f},
        0.8f, true, true, true, 60.0f, 0 // Bass Dominant audio preset
    },
    
    {
        "Túnel Psicodélico",
        "Túnel infinito con figuras que aparecen y desaparecen",
        // Center - Múltiples formas que cambian
        {SHAPE_TRIANGLE, 15, 0.15f, 120.0f, 0.0f, 0.0f, 0.0f, 1.2f, 1.2f,
         ImVec4(1.0f, 0.0f, 1.0f, 1.0f), ImVec4(0.8f, 0.0f, 0.8f, 1.0f), ImVec4(0.6f, 0.0f, 0.6f, 1.0f), 3, true, 3.5f},
        // Right - Líneas que fluyen
        {SHAPE_LINE, 20, 0.08f, 180.0f, 60.0f, 0.0f, 0.0f, 2.5f, 0.3f,
         ImVec4(0.0f, 1.0f, 1.0f, 1.0f), ImVec4(0.0f, 0.8f, 0.8f, 1.0f), ImVec4(0.0f, 0.6f, 0.6f, 1.0f), 2, false, 0.0f},
        // Left - Círculos que pulsan
        {SHAPE_CIRCLE, 25, 0.12f, 90.0f, -60.0f, 0.0f, 0.0f, 0.8f, 0.8f,
         ImVec4(1.0f, 1.0f, 0.0f, 1.0f), ImVec4(0.8f, 0.8f, 0.0f, 1.0f), ImVec4(0.6f, 0.6f, 0.0f, 1.0f), 32, false, 0.0f},
        2.0f, true, true, true, 140.0f, 6 // Chaos Mode audio preset
    }
};

// === MONITORES DE AUDIO ===
#include <utility>
#include <vector>
#include <string>

// Lista de monitores y monitor seleccionado
static std::vector<std::pair<std::string, std::string>> audioMonitors;
static int selectedMonitor = 0;
static int prevSelectedMonitor = 0;

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

    // OPTIMIZATION: Initialize VBO caching system
    prepareInstanceBuffer();
    vboCache.reserve(MAX_CACHED_VBOS);

    // Dear ImGui: setup
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    
    // OPTIMIZATION: Configure ImGui for better performance
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    
    // OPTIMIZATION: Reduce ImGui update frequency for better performance
    io.ConfigInputTextCursorBlink = false; // Disable cursor blink
    io.ConfigInputTextEnterKeepActive = false; // Don't keep active on enter
    
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    // Declarar randomAffect para randomización selectiva
    RandomAffectFlags randomAffect;

    // Triángulo
    float triSize = 0.8f;
    float prevSize = triSize;
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
    // Declare randomLimits
    RandomLimits randomLimits;
    GLuint shaderProgram = createShaderProgram(vertexShaderSource, fragmentShaderSource);
    
    // OPTIMIZATION: Remove old VAO/VBO variables - now using caching system
    float colorTopArr[3] = {colorTop.x, colorTop.y, colorTop.z};
    float colorLeftArr[3] = {colorLeft.x, colorLeft.y, colorLeft.z};
    float colorRightArr[3] = {colorRight.x, colorRight.y, colorRight.z};
    int actualSegments = (shapeType == 0) ? 3 : (shapeType == 1) ? 4 : 128;
    
    // OPTIMIZATION: Initialize first cached VBO
    float colors[9] = {colorTopArr[0], colorTopArr[1], colorTopArr[2],
                      colorLeftArr[0], colorLeftArr[1], colorLeftArr[2],
                      colorRightArr[0], colorRightArr[1], colorRightArr[2]};
    CachedVBO* currentCachedVBO = findOrCreateCachedVBO(shapeType, triSize, colors, actualSegments, false, 0.0f);

    int numTriangles = 1;

    // Variables que necesitan estar declaradas antes de loadPreset
    float groupSeparation = 1.0f;
    bool onlyRGB = false;
    bool fractalMode = false;
    float fractalDepth = 3.0f;

    // Al iniciar el programa, intenta cargar preset.json
    loadPreset("preset.json", triSize, rotationSpeed, translateX, translateY, scaleX, scaleY, colorTop, colorLeft, colorRight, numCenter, numRight, numLeft, shapeType, groupAngleCenter, groupAngleRight, groupAngleLeft, randomize, randomLimits, randomAffect, groupSeparation, onlyRGB, animateColor, bpm, fpsMode, customFps, fractalMode, fractalDepth);

    // Grupos: centro, derecha, izquierda
    VisualGroup groups[3]; // 0: centro, 1: derecha, 2: izquierda

    // Inicializar grupos
    for (int g = 0; g < 3; ++g) {
        groups[g].numObjects = (g == 0) ? 1 : 0;
        groups[g].groupAngle = 0.0f;
        groups[g].objects.resize(MAX_OBJECTS);
        groups[g].targets.resize(MAX_OBJECTS);
    }

    // Default startup preset: single centered triangle
    // Ensure exactly one object in the center and none on sides
    groups[0].numObjects = 1;
    groups[1].numObjects = 0;
    groups[2].numObjects = 0;
    // Configure the center object as a triangle at the origin
    groups[0].objects[0].shapeType = SHAPE_TRIANGLE;
    groups[0].objects[0].translateX = 0.0f;
    groups[0].objects[0].translateY = 0.0f;
    groups[0].objects[0].scaleX = 1.0f;
    groups[0].objects[0].scaleY = 1.0f;
    groups[0].objects[0].triSize = 0.8f;
    groups[0].objects[0].rotationSpeed = 0.0f;
    groups[0].objects[0].angle = 0.0f;
    groups[0].objects[0].nSegments = 3;

    // Variables globales para animación

    // 1. Flags de randomización por grupo
    // Variables para randomización por grupo (ahora manejadas por randomAffect)
    // static bool randomShapeType[3] = {true, true, true};
    // static bool randomNSegments[3] = {true, true, true};

    // Declarar lambdas antes del bucle de randomización
    auto frand = []() { return static_cast<float>(rand())/RAND_MAX; };
    
    // Variables para randomización más natural
    static float lastRandomizeTime[3] = {0.0f, 0.0f, 0.0f}; // Tiempo de última randomización por grupo
    static float randomizeIntervals[3] = {2.0f, 3.0f, 2.5f}; // Intervalos diferentes por grupo
    
    // Auto-randomization variables
    static bool autoRandomizePresets = false;
    static float lastPresetRandomizeTime = 0.0f;
    static float presetRandomizeInterval = 5.0f; // 5 seconds
    static bool randomizeOnlyFractals = false;
    static bool randomizeOnlyLines = false;
    static bool randomizeOnlyCylinders = false;
    static float randomizeVariation[3] = {0.5f, 0.8f, 0.6f}; // Variación en los intervalos

    // --- NUEVO: Modo Fractal Toggle ---
    static bool fractalToggleMode = false;
    static float fractalToggleInterval = 1.5f; // 1.5 segundos
    static float lastFractalToggleTime = 0.0f;
    static bool fractalToggleState = false; // true = fractal activo, false = fractal inactivo

    // --- NUEVO: Efecto Glitch ---
    static bool glitchEffectEnabled = false;
    static float glitchIntensity = 0.5f;
    static float glitchFrequency = 0.1f; // Frecuencia del glitch
    static float lastGlitchTime = 0.0f;
    static float glitchDelay = 0.05f; // Delay del efecto glitch
    static bool glitchActive = false;
    static float glitchSplitRatio = 0.5f; // Ratio de división de objetos (0.5 = mitad)
    static float glitchOffsetX = 0.0f;
    static float glitchOffsetY = 0.0f;
    static float glitchScaleX = 1.0f;
    static float glitchScaleY = 1.0f;

    // --- NUEVO: Randomización basada en frecuencias de música ---
    static bool frequencyBasedRandomization = false;
    static float bassRandomizationThreshold = 0.3f;
    static float midRandomizationThreshold = 0.4f;
    static float trebleRandomizationThreshold = 0.5f;
    static float lastBassRandomizeTime = 0.0f;
    static float lastMidRandomizeTime = 0.0f;
    static float lastTrebleRandomizeTime = 0.0f;
    static float frequencyRandomizeCooldown = 0.5f; // Cooldown entre randomizaciones por frecuencia

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
    const char* audioDevice = "default"; // Use default device instead of specific one
    const int audioSampleRate = 48000;
    const int audioChannels = 2;

    // Obtener lista de monitores de audio al inicio
    audioMonitors = get_monitor_sources();
    if (audioMonitors.empty()) {
        std::cerr << "No se encontraron monitores de audio.\n";
    }
    selectedMonitor = 0;
    prevSelectedMonitor = 0;

    while (!glfwWindowShouldClose(window)) {
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, true);
        }
        
        // UI MASTER CONTROL: Keyboard shortcuts
        static bool hKeyPressed = false;
        if (glfwGetKey(window, GLFW_KEY_H) == GLFW_PRESS && !hKeyPressed) {
            uiVisibility.showAll = !uiVisibility.showAll;
            if (!uiVisibility.showAll) {
                uiVisibility.showMainControls = false;
                uiVisibility.showAdvancedOptions = false;
                uiVisibility.showRandomization = false;
                uiVisibility.showSystemMonitor = false;
                uiVisibility.showAudioControl = false;
                uiVisibility.showGlobalOptions = false;
                uiVisibility.showAudioGraph = false;
                uiVisibility.showAudioTestMode = false;
            } else {
                uiVisibility.showMainControls = true;
                uiVisibility.showAdvancedOptions = true;
                uiVisibility.showRandomization = true;
                uiVisibility.showSystemMonitor = true;
                uiVisibility.showAudioControl = true;
                uiVisibility.showGlobalOptions = true;
                uiVisibility.showAudioGraph = true;
                uiVisibility.showAudioTestMode = true;
            }
            hKeyPressed = true;
        }
        if (glfwGetKey(window, GLFW_KEY_H) == GLFW_RELEASE) {
            hKeyPressed = false;
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

        // OPTIMIZATION: Reduce ImGui update frequency for better performance
        static float lastImGuiUpdate = 0.0f;
        bool shouldUpdateImGui = (currentTime - lastImGuiUpdate > 0.016f); // ~60fps for UI
        if (shouldUpdateImGui) {
            lastImGuiUpdate = currentTime;
        }

        // Ventana izquierda: controles principales + monitor de sistema + opciones globales
        if (shouldUpdateImGui && uiVisibility.showMainControls) {
            ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Once);
            ImGui::Begin("Triángulo (Opciones Globales) + Monitor del sistema");
            ImGui::SliderFloat("Tamaño", &groups[0].objects[0].triSize, 0.1f, 2.0f, "%.2f");
            ImGui::SliderAngle("Rotación", &groups[0].objects[0].angle, 0.0f, 360.0f);
            ImGui::SliderFloat("Velocidad de rotación (°/s)", &groups[0].objects[0].rotationSpeed, 10.0f, 720.0f, "%.1f");
            ImGui::SliderFloat("BPM", &bpm, 30.0f, 300.0f, "%.1f");
            ImGui::Text("Beat phase: %.2f", beatPhase);
            const char* fpsModes[] = { "VSync", "Ilimitado", "Custom" };
            ImGui::Combo("FPS Mode", &fpsMode, fpsModes, IM_ARRAYSIZE(fpsModes));
            if (fpsMode == FPS_CUSTOM) {
                ImGui::SliderInt("Custom FPS", &customFps, 10, 1000);
            }
            ImGui::Text("ESC para salir | H para ocultar/mostrar UI");
            ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
            ImGui::Separator();
            // Opciones Globales
            ImGui::Checkbox("Rotación automática", &autoRotate);
            ImGui::Checkbox("Animar color", &animateColor);
            ImGui::Checkbox("Visuales controlados por audio del sistema", &audioReactive);
            ImGui::Separator();
            // Debug/Info: Randomización por grupo
            ImGui::Separator();
            ImGui::Text("Randomización por grupo:");
            ImGui::Text("(Usar controles de '¿Qué randomizar?' en la ventana de Randomización)");
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
            ImGui::Separator();
            ImGui::SliderFloat("Separación de grupos", &groupSeparation, 0.0f, 2.0f, "%.2f");
            ImGui::Checkbox("Randomizar separación de grupos", &randomizeGroupSeparation);
            ImGui::Separator();
            // Monitor del sistema
            float cpuUsage = getCPUUsage();
            float cpuTemp = getCPUTemp();
            float gpuTemp = getGPUTemp();
            ImGui::Text("CPU uso: %s", cpuUsage >= 0.0f ? (std::to_string(cpuUsage) + "%").c_str() : "No disponible");
            ImGui::Text("CPU temp: %s", cpuTemp >= 0.0f ? (std::to_string(cpuTemp) + " °C").c_str() : "No disponible");
            ImGui::Text("GPU temp: %s", gpuTemp >= 0.0f ? (std::to_string(gpuTemp) + " °C").c_str() : "No disponible");
            ImGui::End();
        }

        // Ventana derecha: opciones avanzadas
        if (uiVisibility.showAdvancedOptions) {
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
        
        // AUDIO-DRIVEN RANDOMIZATION INFO
        if (audioReactive && randomize) {
            ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "🎵 Randomización Controlada por Audio ACTIVA");
            ImGui::Text("Centro: Bass %.2f | Derecha: Mid %.2f | Izquierda: Treble %.2f", 
                       currentAudio.bass, currentAudio.mid, currentAudio.treble);
        }
        
        ImGui::Separator();
        ImGui::Text("=== MODO FRACTAL ===");
        ImGui::Checkbox("Modo Fractal", &fractalMode);
        if (fractalMode) {
            ImGui::SliderFloat("Profundidad Fractal", &fractalDepth, 1.0f, 5.0f, "%.1f");
            ImGui::Text("Crea fractales animados y coloridos");
            ImGui::Text("basados en la figura seleccionada");
            ImGui::Text("✅ Todas las figuras son compatibles con fractales");
        }
        
        // --- NUEVO: Modo Fractal Toggle ---
        ImGui::Separator();
        ImGui::Text("=== MODO FRACTAL TOGGLE ===");
        ImGui::Checkbox("Modo Fractal Toggle (1.5s)", &fractalToggleMode);
        if (fractalToggleMode) {
            ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "🔄 Fractal se activa/desactiva cada 1.5 segundos");
            ImGui::Text("Estado actual: %s", fractalToggleState ? "ACTIVO" : "INACTIVO");
            ImGui::SliderFloat("Intervalo (segundos)", &fractalToggleInterval, 0.5f, 5.0f, "%.1f");
        }
        
        // --- NUEVO: Efecto Glitch ---
        ImGui::Separator();
        ImGui::Text("=== EFECTO GLITCH ===");
        ImGui::Checkbox("Efecto Glitch", &glitchEffectEnabled);
        if (glitchEffectEnabled) {
            ImGui::SliderFloat("Intensidad Glitch", &glitchIntensity, 0.1f, 2.0f, "%.2f");
            ImGui::SliderFloat("Frecuencia Glitch", &glitchFrequency, 0.01f, 1.0f, "%.2f");
            ImGui::SliderFloat("Delay Glitch (ms)", &glitchDelay, 0.01f, 0.2f, "%.3f");
            ImGui::SliderFloat("Ratio División", &glitchSplitRatio, 0.1f, 1.0f, "%.2f");
            ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "🎭 Efecto de división y delay de objetos");
            ImGui::Text("Estado: %s", glitchActive ? "ACTIVO" : "INACTIVO");
        }
        
        // --- NUEVO: Randomización basada en frecuencias ---
        ImGui::Separator();
        ImGui::Text("=== RANDOMIZACIÓN POR FRECUENCIAS ===");
        ImGui::Checkbox("Randomización por Frecuencias", &frequencyBasedRandomization);
        if (frequencyBasedRandomization) {
            ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "🎵 Randomización basada en frecuencias de música");
            ImGui::SliderFloat("Umbral Bass", &bassRandomizationThreshold, 0.1f, 0.8f, "%.2f");
            ImGui::SliderFloat("Umbral Mid", &midRandomizationThreshold, 0.1f, 0.8f, "%.2f");
            ImGui::SliderFloat("Umbral Treble", &trebleRandomizationThreshold, 0.1f, 0.8f, "%.2f");
            ImGui::SliderFloat("Cooldown (segundos)", &frequencyRandomizeCooldown, 0.1f, 2.0f, "%.1f");
            ImGui::Text("Bass: %.2f | Mid: %.2f | Treble: %.2f", 
                       currentAudio.bass, currentAudio.mid, currentAudio.treble);
        }
        
        ImGui::Separator();
        ImGui::Text("OpenGL: %s", (const char*)glGetString(GL_VERSION));
        ImGui::Text("GPU: %s", (const char*)glGetString(GL_RENDERER));
        ImGui::Text("Resolución: %dx%d", width, height);
        ImGui::Separator();
        
        // INDEPENDENT SHAPES INFO
        ImGui::TextColored(ImVec4(0.0f, 1.0f, 1.0f, 1.0f), "🎯 Figuras Independientes por Grupo:");
        ImGui::Text("Centro: %s", shapeNames[groups[0].objects[0].shapeType]);
        ImGui::Text("Derecha: %s", shapeNames[groups[1].objects[0].shapeType]);
        ImGui::Text("Izquierda: %s", shapeNames[groups[2].objects[0].shapeType]);
        
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
        }

        // Ventana randomización
        if (uiVisibility.showRandomization) {
        ImGui::SetNextWindowPos(ImVec2(width - 350, height - 400), ImGuiCond_Once);
        ImGui::SetNextWindowSize(ImVec2(340, 390), ImGuiCond_Once);
        ImGui::Begin("Randomización");
        ImGui::Checkbox("Activar random", &randomize);
        ImGui::SliderFloat("Suavidad randomización", &randomLerpSpeed, 0.001f, 0.2f, "%.3f");
        ImGui::SliderFloat("Frecuencia base", &randomizeIntervals[0], 0.5f, 10.0f, "%.1f");
        ImGui::Text("(Intervalo base para todos los grupos)");
        ImGui::Separator();
        
        // --- NUEVO: Semilla de randomización ---
        static int randomSeedMode = 0; // 0 = hora actual, 1 = audio
        const char* seedModes[] = {"Semilla: Hora actual", "Semilla: Audio del sistema"};
        ImGui::Combo("Modo de semilla", &randomSeedMode, seedModes, IM_ARRAYSIZE(seedModes));
        if (randomSeedMode == 0) {
            // Semilla por hora actual
            srand((unsigned int)time(nullptr));
        } else {
            // Semilla por audio (usar suma de frecuencias como semilla)
            float audioSum = 0.0f;
            if (!spectrum.empty()) {
                for (float v : spectrum) audioSum += v;
            }
            unsigned int audioSeed = (unsigned int)(audioSum * 100000.0f);
            srand(audioSeed);
        }
        ImGui::Text("La randomización será única según el modo de semilla seleccionado.");
        ImGui::Separator();
        
        // RANDOMIZATION STATUS: Show which parameters are being affected
        if (randomize) {
            ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "🎲 Parámetros siendo randomizados:");
            if (randomAffect.triSize) ImGui::Text("✅ Tamaño");
            if (randomAffect.rotationSpeed) ImGui::Text("✅ Velocidad rotación");
            if (randomAffect.angle) ImGui::Text("✅ Ángulo");
            if (randomAffect.translateX) ImGui::Text("✅ Translación X");
            if (randomAffect.translateY) ImGui::Text("✅ Translación Y");
            if (randomAffect.scaleX) ImGui::Text("✅ Escala X");
            if (randomAffect.scaleY) ImGui::Text("✅ Escala Y");
            if (randomAffect.colorTop) ImGui::Text("✅ Color Top");
            if (randomAffect.colorLeft) ImGui::Text("✅ Color Left");
            if (randomAffect.colorRight) ImGui::Text("✅ Color Right");
            if (randomAffect.shapeType) ImGui::Text("✅ Tipo de figura");
            if (randomAffect.nSegments) ImGui::Text("✅ Segmentos");
            if (randomAffect.groupAngle) ImGui::Text("✅ Ángulo de grupo");
            if (randomAffect.numCenter) ImGui::Text("✅ Cantidad Centro");
            if (randomAffect.numRight) ImGui::Text("✅ Cantidad Derecha");
            if (randomAffect.numLeft) ImGui::Text("✅ Cantidad Izquierda");
            
            // Check if no parameters are selected
            bool anySelected = randomAffect.triSize || randomAffect.rotationSpeed || randomAffect.angle ||
                              randomAffect.translateX || randomAffect.translateY || randomAffect.scaleX ||
                              randomAffect.scaleY || randomAffect.colorTop || randomAffect.colorLeft ||
                              randomAffect.colorRight || randomAffect.shapeType || randomAffect.nSegments ||
                              randomAffect.groupAngle || randomAffect.numCenter || randomAffect.numRight ||
                              randomAffect.numLeft;
            
            if (!anySelected) {
                ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "⚠️ ¡Ningún parámetro seleccionado!");
                ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "   La randomización no afectará nada.");
            }
        } else {
            ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "❌ Randomización desactivada");
        }
        
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
        }

        // AUDIO REACTIVE SYSTEM: Advanced Audio Control Window
        if (uiVisibility.showAudioControl) {
            ImGui::SetNextWindowPos(ImVec2(width - 700, height - 500), ImGuiCond_Once);
            ImGui::SetNextWindowSize(ImVec2(680, 480), ImGuiCond_Once);
            ImGui::Begin("🎵 Control de Audio Reactivo Avanzado");
            // --- Combo box para elegir monitor ---
            if (!audioMonitors.empty()) {
                std::vector<const char*> items;
                for (auto& m : audioMonitors) items.push_back(m.second.c_str());
                int prev = selectedMonitor;
                ImGui::Combo("Monitor de audio", &selectedMonitor, items.data(), items.size());
                if (selectedMonitor != prev) {
                    // Cambió el monitor, reinicializar audio
                    if (audioInit) {
                        delete audio;
                        delete fft;
                        audio = nullptr;
                        fft = nullptr;
                        audioInit = false;
                    }
                    audioReactive = false; // Forzar a reactivar para que se reinicialice
                }
                ImGui::Text("Monitor actual: %s", audioMonitors[selectedMonitor].second.c_str());
            } else {
                ImGui::TextColored(ImVec4(1,0,0,1), "No se encontraron monitores de audio");
            }
        
        // Audio Status with more detailed information
        ImGui::Text("Estado Audio: %s", audioReactive ? "✅ ACTIVO" : "❌ INACTIVO");
        ImGui::Text("Dispositivo: %s", audioDevice);
        ImGui::Text("Inicializado: %s", audioInit ? "✅ Sí" : "❌ No");
        
        if (audioReactive && !spectrum.empty()) {
            ImGui::Text("Análisis: Bass: %.3f | Mid: %.3f | Treble: %.3f | Peak: %.3f", 
                       currentAudio.bass, currentAudio.mid, currentAudio.treble, currentAudio.peak);
            ImGui::Text("RMS: %.3f | Overall: %.3f", currentAudio.rms, currentAudio.overall);
        } else if (audioReactive) {
            ImGui::Text("⚠️ No hay datos de audio disponibles");
        }
        
        ImGui::Separator();
        
        // Audio Presets
        ImGui::Text("🎛️ Presets de Audio:");
        ImGui::SameLine();
        if (ImGui::Button("Aplicar a Todos")) {
            for (int g = 0; g < 3; ++g) {
                applyAudioPreset(audioGroups[g], audioPresets[0]); // Default to first preset
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("Wide Full Range")) {
            for (int g = 0; g < 3; ++g) {
                applyAudioPreset(audioGroups[g], audioPresets[7]); // Wide Full Range preset
            }
        }
        
        // Preset buttons in a grid
        for (int i = 0; i < audioPresets.size(); ++i) {
            if (i > 0 && i % 3 != 0) ImGui::SameLine();
            if (ImGui::Button(audioPresets[i].name.c_str())) {
                for (int g = 0; g < 3; ++g) {
                    applyAudioPreset(audioGroups[g], audioPresets[i]);
                }
            }
        }
        
        ImGui::Separator();
        
        // Group-specific controls
        const char* groupNames[] = {"Centro", "Derecha", "Izquierda"};
        for (int g = 0; g < 3; ++g) {
            if (ImGui::CollapsingHeader(groupNames[g])) {
                AudioReactiveGroup& group = audioGroups[g];
                
                // Frequency range controls
                ImGui::Text("🎵 Rangos de Frecuencia:");
                ImGui::Checkbox("Bass (20-150Hz)", &group.bass.enabled);
                ImGui::SameLine();
                ImGui::SliderFloat("Sens Bass", &group.bass.sensitivity, 0.1f, 5.0f, "%.1f");
                
                ImGui::Checkbox("Low Mid (150-400Hz)", &group.lowMid.enabled);
                ImGui::SameLine();
                ImGui::SliderFloat("Sens LM", &group.lowMid.sensitivity, 0.1f, 5.0f, "%.1f");
                
                ImGui::Checkbox("Mid (400-2kHz)", &group.mid.enabled);
                ImGui::SameLine();
                ImGui::SliderFloat("Sens Mid", &group.mid.sensitivity, 0.1f, 5.0f, "%.1f");
                
                ImGui::Checkbox("High Mid (2-6kHz)", &group.highMid.enabled);
                ImGui::SameLine();
                ImGui::SliderFloat("Sens HM", &group.highMid.sensitivity, 0.1f, 5.0f, "%.1f");
                
                ImGui::Checkbox("Treble (6-20kHz)", &group.treble.enabled);
                ImGui::SameLine();
                ImGui::SliderFloat("Sens Treb", &group.treble.sensitivity, 0.1f, 5.0f, "%.1f");
                
                ImGui::Separator();
                
                // Parameter controls
                ImGui::Text("🎛️ Parámetros Controlados:");
                
                // Row 1
                ImGui::Checkbox("Tamaño", &group.size.enabled);
                ImGui::SameLine();
                ImGui::SliderFloat("Min Size", &group.size.minValue, 0.1f, 2.0f, "%.2f");
                ImGui::SameLine();
                ImGui::SliderFloat("Max Size", &group.size.maxValue, 0.1f, 5.0f, "%.2f");
                ImGui::SameLine();
                ImGui::SliderFloat("Sens Size", &group.size.sensitivity, 0.1f, 5.0f, "%.1f");
                
                // Row 2
                ImGui::Checkbox("Rotación", &group.rotation.enabled);
                ImGui::SameLine();
                ImGui::SliderFloat("Min Rot", &group.rotation.minValue, 0.0f, 360.0f, "%.0f");
                ImGui::SameLine();
                ImGui::SliderFloat("Max Rot", &group.rotation.maxValue, 0.0f, 1000.0f, "%.0f");
                ImGui::SameLine();
                ImGui::SliderFloat("Sens Rot", &group.rotation.sensitivity, 0.1f, 5.0f, "%.1f");
                
                // Row 3
                ImGui::Checkbox("Ángulo", &group.angle.enabled);
                ImGui::SameLine();
                ImGui::SliderFloat("Min Ang", &group.angle.minValue, 0.0f, 360.0f, "%.0f");
                ImGui::SameLine();
                ImGui::SliderFloat("Max Ang", &group.angle.maxValue, 0.0f, 360.0f, "%.0f");
                ImGui::SameLine();
                ImGui::SliderFloat("Sens Ang", &group.angle.sensitivity, 0.1f, 5.0f, "%.1f");
                
                // Row 4
                ImGui::Checkbox("Mover X", &group.translateX.enabled);
                ImGui::SameLine();
                ImGui::SliderFloat("Min TX", &group.translateX.minValue, -2.0f, 2.0f, "%.2f");
                ImGui::SameLine();
                ImGui::SliderFloat("Max TX", &group.translateX.maxValue, -2.0f, 2.0f, "%.2f");
                ImGui::SameLine();
                ImGui::SliderFloat("Sens TX", &group.translateX.sensitivity, 0.1f, 5.0f, "%.1f");
                
                // Row 5
                ImGui::Checkbox("Mover Y", &group.translateY.enabled);
                ImGui::SameLine();
                ImGui::SliderFloat("Min TY", &group.translateY.minValue, -2.0f, 2.0f, "%.2f");
                ImGui::SameLine();
                ImGui::SliderFloat("Max TY", &group.translateY.maxValue, -2.0f, 2.0f, "%.2f");
                ImGui::SameLine();
                ImGui::SliderFloat("Sens TY", &group.translateY.sensitivity, 0.1f, 5.0f, "%.1f");
                
                // Row 6
                ImGui::Checkbox("Escala X", &group.scaleX.enabled);
                ImGui::SameLine();
                ImGui::SliderFloat("Min SX", &group.scaleX.minValue, 0.1f, 2.0f, "%.2f");
                ImGui::SameLine();
                ImGui::SliderFloat("Max SX", &group.scaleX.maxValue, 0.1f, 5.0f, "%.2f");
                ImGui::SameLine();
                ImGui::SliderFloat("Sens SX", &group.scaleX.sensitivity, 0.1f, 5.0f, "%.1f");
                
                // Row 7
                ImGui::Checkbox("Escala Y", &group.scaleY.enabled);
                ImGui::SameLine();
                ImGui::SliderFloat("Min SY", &group.scaleY.minValue, 0.1f, 2.0f, "%.2f");
                ImGui::SameLine();
                ImGui::SliderFloat("Max SY", &group.scaleY.maxValue, 0.1f, 5.0f, "%.2f");
                ImGui::SameLine();
                ImGui::SliderFloat("Sens SY", &group.scaleY.sensitivity, 0.1f, 5.0f, "%.1f");
                
                // Row 8
                ImGui::Checkbox("Intensidad Color", &group.colorIntensity.enabled);
                ImGui::SameLine();
                ImGui::SliderFloat("Min Col", &group.colorIntensity.minValue, 0.0f, 1.0f, "%.2f");
                ImGui::SameLine();
                ImGui::SliderFloat("Max Col", &group.colorIntensity.maxValue, 0.0f, 2.0f, "%.2f");
                ImGui::SameLine();
                ImGui::SliderFloat("Sens Col", &group.colorIntensity.sensitivity, 0.1f, 5.0f, "%.1f");
                
                // Row 9
                ImGui::Checkbox("Ángulo Grupo", &group.groupAngle.enabled);
                ImGui::SameLine();
                ImGui::SliderFloat("Min GA", &group.groupAngle.minValue, 0.0f, 360.0f, "%.0f");
                ImGui::SameLine();
                ImGui::SliderFloat("Max GA", &group.groupAngle.maxValue, 0.0f, 360.0f, "%.0f");
                ImGui::SameLine();
                ImGui::SliderFloat("Sens GA", &group.groupAngle.sensitivity, 0.1f, 5.0f, "%.1f");
                
                // Row 10
                ImGui::Checkbox("Cantidad Objetos", &group.numObjects.enabled);
                ImGui::SameLine();
                ImGui::SliderFloat("Min Obj", &group.numObjects.minValue, 0.0f, 50.0f, "%.0f");
                ImGui::SameLine();
                ImGui::SliderFloat("Max Obj", &group.numObjects.maxValue, 0.0f, 100.0f, "%.0f");
                ImGui::SameLine();
                ImGui::SliderFloat("Sens Obj", &group.numObjects.sensitivity, 0.1f, 5.0f, "%.1f");
                
                ImGui::Separator();
                
                // Mix presets for this group
                ImGui::Text("🎚️ Mix de Frecuencias:");
                ImGui::Checkbox("Mix Bass", &group.useBassMix);
                ImGui::SameLine();
                ImGui::Checkbox("Mix Mid", &group.useMidMix);
                ImGui::SameLine();
                ImGui::Checkbox("Mix Treble", &group.useTrebleMix);
                ImGui::SameLine();
                ImGui::Checkbox("Mix Completo", &group.useFullSpectrumMix);
            }
        }
        
        ImGui::End();
        }

        // UI MASTER CONTROL: Ventana de control maestro
        ImGui::SetNextWindowPos(ImVec2(width - 200, 10), ImGuiCond_Once);
        ImGui::SetNextWindowSize(ImVec2(180, 200), ImGuiCond_Once);
        ImGui::Begin("🎛️ Control Maestro", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
        
        ImGui::Text("🎮 Control de Ventanas");
        ImGui::Text("⌨️ Presiona 'H' para mostrar/ocultar todo");
        ImGui::Separator();
        
        // Master toggle
        if (ImGui::Button(uiVisibility.showAll ? "🙈 Ocultar Todo" : "👁️ Mostrar Todo")) {
            uiVisibility.showAll = !uiVisibility.showAll;
            if (!uiVisibility.showAll) {
                uiVisibility.showMainControls = false;
                uiVisibility.showAdvancedOptions = false;
                uiVisibility.showRandomization = false;
                uiVisibility.showSystemMonitor = false;
                uiVisibility.showAudioControl = false;
                uiVisibility.showGlobalOptions = false;
                uiVisibility.showAudioGraph = false;
                uiVisibility.showAudioTestMode = false;
                uiVisibility.showPresets = false;
            } else {
                uiVisibility.showMainControls = true;
                uiVisibility.showAdvancedOptions = true;
                uiVisibility.showRandomization = true;
                uiVisibility.showSystemMonitor = true;
                uiVisibility.showAudioControl = true;
                uiVisibility.showGlobalOptions = true;
                uiVisibility.showAudioGraph = true;
                uiVisibility.showAudioTestMode = true;
                uiVisibility.showPresets = true;
            }
        }
        
        ImGui::Separator();
        
        // Individual window toggles
        ImGui::Text("Ventanas Individuales:");
        ImGui::Checkbox("Controles Principales", &uiVisibility.showMainControls);
        ImGui::Checkbox("Opciones Avanzadas", &uiVisibility.showAdvancedOptions);
        ImGui::Checkbox("Randomización", &uiVisibility.showRandomization);
        ImGui::Checkbox("Monitor Sistema", &uiVisibility.showSystemMonitor);
        ImGui::Checkbox("Control Audio", &uiVisibility.showAudioControl);
        ImGui::Checkbox("Gráfico Audio", &uiVisibility.showAudioGraph);
        ImGui::Checkbox("Opciones Globales", &uiVisibility.showGlobalOptions);
        ImGui::Checkbox("Modo de Prueba", &uiVisibility.showAudioTestMode);
        ImGui::Checkbox("Presets", &uiVisibility.showPresets);
        
        ImGui::Separator();
        
        // Quick presets
        ImGui::Text("Presets Rápidos:");
        if (ImGui::Button("🎵 Solo Audio")) {
            uiVisibility.showMainControls = false;
            uiVisibility.showAdvancedOptions = false;
            uiVisibility.showRandomization = false;
            uiVisibility.showSystemMonitor = false;
            uiVisibility.showAudioControl = true;
            uiVisibility.showAudioGraph = true;
            uiVisibility.showGlobalOptions = false;
            uiVisibility.showAll = false;
        }
        
        if (ImGui::Button("📊 Solo Gráficos")) {
            uiVisibility.showMainControls = false;
            uiVisibility.showAdvancedOptions = false;
            uiVisibility.showRandomization = false;
            uiVisibility.showSystemMonitor = true;
            uiVisibility.showAudioControl = false;
            uiVisibility.showAudioGraph = true;
            uiVisibility.showGlobalOptions = false;
            uiVisibility.showAll = false;
        }
        
        if (ImGui::Button("🎲 Solo Random")) {
            uiVisibility.showMainControls = false;
            uiVisibility.showAdvancedOptions = false;
            uiVisibility.showRandomization = true;
            uiVisibility.showSystemMonitor = false;
            uiVisibility.showAudioControl = false;
            uiVisibility.showAudioGraph = false;
            uiVisibility.showGlobalOptions = false;
            uiVisibility.showAll = false;
        }
        
        if (ImGui::Button("⚙️ Solo Controles")) {
            uiVisibility.showMainControls = true;
            uiVisibility.showAdvancedOptions = true;
            uiVisibility.showRandomization = false;
            uiVisibility.showSystemMonitor = false;
            uiVisibility.showAudioControl = false;
            uiVisibility.showAudioGraph = false;
            uiVisibility.showGlobalOptions = false;
            uiVisibility.showAll = false;
        }
        
        if (ImGui::Button("🧪 Solo Prueba")) {
            uiVisibility.showMainControls = false;
            uiVisibility.showAdvancedOptions = false;
            uiVisibility.showRandomization = false;
            uiVisibility.showSystemMonitor = false;
            uiVisibility.showAudioControl = false;
            uiVisibility.showAudioGraph = false;
            uiVisibility.showGlobalOptions = false;
            uiVisibility.showAudioTestMode = true;
            uiVisibility.showPresets = false;
            uiVisibility.showAll = false;
        }
        
        if (ImGui::Button("🎨 Solo Presets")) {
            uiVisibility.showMainControls = false;
            uiVisibility.showAdvancedOptions = false;
            uiVisibility.showRandomization = false;
            uiVisibility.showSystemMonitor = false;
            uiVisibility.showAudioControl = false;
            uiVisibility.showAudioGraph = false;
            uiVisibility.showGlobalOptions = false;
            uiVisibility.showAudioTestMode = false;
            uiVisibility.showPresets = true;
            uiVisibility.showAll = false;
        }
        
        ImGui::End();

        // AUDIO GRAPH WINDOW: Para medir latencia y optimizar
        if (uiVisibility.showAudioGraph) {
            ImGui::SetNextWindowPos(ImVec2(10, height - 300), ImGuiCond_Once);
            ImGui::SetNextWindowSize(ImVec2(400, 280), ImGuiCond_Once);
            ImGui::Begin("📊 Gráfico de Audio y Latencia");
            
            // Estadísticas de latencia
            ImGui::Text("🎯 Métricas de Latencia:");
            ImGui::Text("Promedio: %.2f ms", audioGraph.averageLatency * 1000.0f);
            ImGui::Text("Mínima: %.2f ms", audioGraph.minLatency * 1000.0f);
            ImGui::Text("Máxima: %.2f ms", audioGraph.maxLatency * 1000.0f);
            ImGui::Text("FPS Audio: %.1f", audioGraph.fps);
            
            ImGui::Separator();
            
            // Gráfico de niveles de audio en tiempo real
            if (!audioGraph.audioLevels.empty()) {
                ImGui::Text("📈 Nivel de Audio (últimos %d frames):", (int)audioGraph.audioLevels.size());
                ImGui::PlotLines("Audio Level", audioGraph.audioLevels.data(), audioGraph.audioLevels.size(), 
                                0, nullptr, 0.0f, 1.0f, ImVec2(380, 80));
                
                ImGui::Text("⏱️ Latencia de Procesamiento:");
                ImGui::PlotLines("Latency (ms)", [](void* data, int idx) -> float {
                    AudioGraphData* graph = (AudioGraphData*)data;
                    if (idx < graph->latencies.size()) {
                        return graph->latencies[idx] * 1000.0f; // Convertir a ms
                    }
                    return 0.0f;
                }, &audioGraph, audioGraph.latencies.size(), 0, nullptr, 0.0f, 50.0f, ImVec2(380, 80));

                // MINI ECUALIZADOR DE FRECUENCIAS (FFT)
                if (!spectrum.empty()) {
                    ImGui::Text("🎚️ Espectro de Frecuencias (FFT):");
                    ImGui::PlotLines("Espectro (FFT)", spectrum.data(), spectrum.size(), 0, nullptr, 0.0f, 1.0f, ImVec2(380, 80));
                } else {
                    ImGui::Text("No hay datos de espectro disponibles");
                }
            } else {
                ImGui::Text("⏳ Esperando datos de audio...");
            }
            
            ImGui::Separator();
            
            // Controles de optimización
            ImGui::Text("⚙️ Optimización:");
            if (ImGui::Button("Limpiar Datos")) {
                audioGraph.clear();
            }
            ImGui::SameLine();
            if (ImGui::Button("Reset Estadísticas")) {
                audioGraph.minLatency = 9999.0f;
                audioGraph.maxLatency = 0.0f;
                audioGraph.averageLatency = 0.0f;
            }
            
            ImGui::Separator();
            
            // Controles de FFT para optimización
            ImGui::Text("🎛️ Ajustes de FFT:");
            // Ajuste duplicado eliminado para evitar variables sin uso; ver lógica más abajo
            
            ImGui::Text("Tamaño actual: %d", audioFftSize);
            ImGui::Text("Frecuencia de muestreo: %d Hz", audioSampleRate);
            ImGui::Text("Resolución: %.1f Hz", (float)audioSampleRate / audioFftSize);
            
            ImGui::Separator();
            
            // Recomendaciones basadas en latencia
            ImGui::Text("💡 Recomendaciones:");
            if (audioGraph.averageLatency > 0.016f) { // Más de 16ms
                ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "⚠️ Latencia alta - Considera reducir FFT size");
            } else if (audioGraph.averageLatency > 0.008f) { // Más de 8ms
                ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "⚡ Latencia moderada - OK para la mayoría de usos");
            } else {
                ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "✅ Latencia excelente - Rendimiento óptimo");
            }
            
            ImGui::End();
        }

        // --- Inicialización de audio y FFT si es necesario ---
        if (audioReactive && !audioInit) {
            try {
                // Usar el monitor seleccionado
                const char* audioDevice = audioMonitors.empty() ? "default" : audioMonitors[selectedMonitor].first.c_str();
                int blockSize = audioFftSize; // Use FFT size as block size for now
                audio = new AudioCapture(audioDevice, audioSampleRate, audioChannels, blockSize);
                fft = new FFTUtils(audioFftSize);
                audioBuffer.resize(audioFftSize * audioChannels);
                monoBuffer.resize(audioFftSize);
                spectrum.resize(audioFftSize / 2);
                audio->start();
                audioInit = true;
                
                // Initialize audio groups with default values
                for (int g = 0; g < 3; ++g) {
                    audioGroups[g].size.minValue = 0.1f;
                    audioGroups[g].size.maxValue = 2.0f;
                    audioGroups[g].rotation.minValue = 0.0f;
                    audioGroups[g].rotation.maxValue = 500.0f;
                    audioGroups[g].angle.minValue = 0.0f;
                    audioGroups[g].angle.maxValue = 360.0f;
                    audioGroups[g].translateX.minValue = -1.0f;
                    audioGroups[g].translateX.maxValue = 1.0f;
                    audioGroups[g].translateY.minValue = -1.0f;
                    audioGroups[g].translateY.maxValue = 1.0f;
                    audioGroups[g].scaleX.minValue = 0.1f;
                    audioGroups[g].scaleX.maxValue = 3.0f;
                    audioGroups[g].scaleY.minValue = 0.1f;
                    audioGroups[g].scaleY.maxValue = 3.0f;
                    audioGroups[g].colorIntensity.minValue = 0.0f;
                    audioGroups[g].colorIntensity.maxValue = 2.0f;
                    audioGroups[g].groupAngle.minValue = 0.0f;
                    audioGroups[g].groupAngle.maxValue = 360.0f;
                    audioGroups[g].numObjects.minValue = 0.0f;
                    audioGroups[g].numObjects.maxValue = 50.0f;
                }
            } catch (const std::exception& e) {
                std::cerr << "Error initializing audio: " << e.what() << std::endl;
                audioReactive = false;
                audioInit = false;
            }
        }
        if (!audioReactive && audioInit) {
            if (audio) audio->stop();
            delete audio;
            delete fft;
            audio = nullptr;
            fft = nullptr;
            audioInit = false;
        }
        // --- Procesamiento de audio y FFT ---
        static int prevFftSize = audioFftSize;
        static int currentFftSize = audioFftSize;
        static int fftSizeIndex = 2; // 1024 por defecto
        // UI FFT size selection (already present in audio graph window)
        // If user changes FFT size, reinitialize audio and FFT
        if (fftSizeIndex == 0) currentFftSize = 256;
        else if (fftSizeIndex == 1) currentFftSize = 512;
        else if (fftSizeIndex == 2) currentFftSize = 1024;
        else if (fftSizeIndex == 3) currentFftSize = 2048;
        else if (fftSizeIndex == 4) currentFftSize = 4096;
        if (currentFftSize != prevFftSize && audioReactive) {
            // Stop and delete old audio/FFT
            if (audioInit) {
                if (audio) audio->stop();
                delete audio;
                delete fft;
                audio = nullptr;
                fft = nullptr;
                audioInit = false;
            }
            // Reinitialize with new FFT size
            int blockSize = currentFftSize;
            const char* audioDevice = audioMonitors.empty() ? "default" : audioMonitors[selectedMonitor].first.c_str();
            audio = new AudioCapture(audioDevice, audioSampleRate, audioChannels, blockSize);
            fft = new FFTUtils(currentFftSize);
            audioBuffer.resize(currentFftSize * audioChannels);
            monoBuffer.resize(currentFftSize);
            spectrum.resize(currentFftSize / 2);
            audio->start();
            audioInit = true;
            prevFftSize = currentFftSize;
        }
        if (audioReactive && audio && fft) {
            try {
                float audioStartTime = glfwGetTime(); // Medir tiempo de inicio
                if (audio->getLatestBlock(audioBuffer)) {
                    for (int i = 0; i < currentFftSize; ++i) {
                        int32_t left = audioBuffer[i * 2];
                        int32_t right = audioBuffer[i * 2 + 1];
                        monoBuffer[i] = (left + right) / 2.0f / 2147483648.0f;
                    }
                    spectrum = fft->compute(monoBuffer);
                    
                    // AUDIO REACTIVE SYSTEM: Advanced analysis
                    analyzeAudioSpectrum(spectrum, currentAudio);
                    
                    // Medir latencia de procesamiento
                    float audioEndTime = glfwGetTime();
                    float processingLatency = audioEndTime - audioStartTime;
                    
                    // Actualizar gráfico de audio
                    audioGraph.addSample(currentAudio.overall, currentTime, processingLatency);
                    audioGraph.updateFPS(currentTime);
                    
                    // Apply audio controls to each group
                    for (int g = 0; g < 3; ++g) {
                        AudioReactiveGroup& audioGroup = audioGroups[g];
                        
                        // Determine which frequency to use based on mix settings
                        float bassValue = currentAudio.bass;
                        float midValue = currentAudio.mid;
                        float trebleValue = currentAudio.treble;
                        float overallValue = currentAudio.overall;
                        
                        // Apply frequency mix
                        if (audioGroup.useBassMix) {
                            bassValue *= 2.0f; // Boost bass
                        }
                        if (audioGroup.useMidMix) {
                            midValue *= 2.0f; // Boost mid
                        }
                        if (audioGroup.useTrebleMix) {
                            trebleValue *= 2.0f; // Boost treble
                        }
                        if (audioGroup.useFullSpectrumMix) {
                            overallValue *= 1.5f; // Boost overall
                        }
                        
                        // Apply controls with delta time for smooth transitions
                        applyAudioControl(audioGroup.size, overallValue, deltaTime);
                        applyAudioControl(audioGroup.rotation, midValue, deltaTime);
                        applyAudioControl(audioGroup.angle, trebleValue, deltaTime);
                        applyAudioControl(audioGroup.translateX, bassValue, deltaTime);
                        applyAudioControl(audioGroup.translateY, midValue, deltaTime);
                        applyAudioControl(audioGroup.scaleX, trebleValue, deltaTime);
                        applyAudioControl(audioGroup.scaleY, bassValue, deltaTime);
                        applyAudioControl(audioGroup.colorIntensity, overallValue, deltaTime);
                        applyAudioControl(audioGroup.groupAngle, midValue, deltaTime);
                        applyAudioControl(audioGroup.numObjects, bassValue, deltaTime);
                        
                        // Apply the audio-controlled values to visual objects
                        if (groups[g].objects.size() < static_cast<size_t>(groups[g].numObjects)) {
                            groups[g].objects.resize(groups[g].numObjects);
                            groups[g].targets.resize(groups[g].numObjects);
                        }
                        
                        for (int i = 0; i < groups[g].numObjects; ++i) {
                            VisualObjectParams& obj = groups[g].objects[i];
                            
                            // Apply audio-controlled parameters with safety checks
                            if (audioGroup.size.enabled) {
                                float sizeValue = audioGroup.size.currentValue;
                                if (!std::isnan(sizeValue) && !std::isinf(sizeValue)) {
                                    obj.triSize = std::max(0.01f, std::min(10.0f, sizeValue));
                                }
                            }
                            if (audioGroup.rotation.enabled) {
                                float rotValue = audioGroup.rotation.currentValue;
                                if (!std::isnan(rotValue) && !std::isinf(rotValue)) {
                                    obj.rotationSpeed = std::max(0.0f, std::min(2000.0f, rotValue));
                                }
                            }
                            if (audioGroup.angle.enabled) {
                                float angleValue = audioGroup.angle.currentValue;
                                if (!std::isnan(angleValue) && !std::isinf(angleValue)) {
                                    obj.angle = angleValue * (3.14159265f / 180.0f);
                                }
                            }
                            if (audioGroup.translateX.enabled) {
                                float txValue = audioGroup.translateX.currentValue;
                                if (!std::isnan(txValue) && !std::isinf(txValue)) {
                                    obj.translateX = std::max(-5.0f, std::min(5.0f, txValue));
                                }
                            }
                            if (audioGroup.translateY.enabled) {
                                float tyValue = audioGroup.translateY.currentValue;
                                if (!std::isnan(tyValue) && !std::isinf(tyValue)) {
                                    obj.translateY = std::max(-5.0f, std::min(5.0f, tyValue));
                                }
                            }
                            if (audioGroup.scaleX.enabled) {
                                float sxValue = audioGroup.scaleX.currentValue;
                                if (!std::isnan(sxValue) && !std::isinf(sxValue)) {
                                    obj.scaleX = std::max(0.01f, std::min(10.0f, sxValue));
                                }
                            }
                            if (audioGroup.scaleY.enabled) {
                                float syValue = audioGroup.scaleY.currentValue;
                                if (!std::isnan(syValue) && !std::isinf(syValue)) {
                                    obj.scaleY = std::max(0.01f, std::min(10.0f, syValue));
                                }
                            }
                            if (audioGroup.colorIntensity.enabled) {
                                float intensity = audioGroup.colorIntensity.currentValue;
                                if (!std::isnan(intensity) && !std::isinf(intensity)) {
                                    intensity = std::max(0.0f, std::min(5.0f, intensity));
                                    obj.colorTop.x = std::min(1.0f, obj.colorTop.x * intensity);
                                    obj.colorTop.y = std::min(1.0f, obj.colorTop.y * intensity);
                                    obj.colorTop.z = std::min(1.0f, obj.colorTop.z * intensity);
                                }
                            }
                        }
                        
                        // Apply group-level controls with safety checks
                        if (audioGroup.groupAngle.enabled) {
                            float gaValue = audioGroup.groupAngle.currentValue;
                            if (!std::isnan(gaValue) && !std::isinf(gaValue)) {
                                groups[g].groupAngle = gaValue * (3.14159265f / 180.0f);
                            }
                        }
                        if (audioGroup.numObjects.enabled) {
                            float numValue = audioGroup.numObjects.currentValue;
                            if (!std::isnan(numValue) && !std::isinf(numValue)) {
                                groups[g].numObjects = (int)std::max(0.0f, std::min(100.0f, numValue));
                            }
                        }
                    }
                }
            } catch (const std::exception& e) {
                std::cerr << "Error processing audio: " << e.what() << std::endl;
                // Reset audio analysis to safe values
                currentAudio.bass = 0.0f;
                currentAudio.lowMid = 0.0f;
                currentAudio.mid = 0.0f;
                currentAudio.highMid = 0.0f;
                currentAudio.treble = 0.0f;
                currentAudio.overall = 0.0f;
                currentAudio.peak = 0.0f;
                currentAudio.rms = 0.0f;
            }
        }

        
        
        // --- Actualización de objetos: rotación automática y animación de color ---
        for (int g = 0; g < 3; ++g) {
            // Asegurar que el vector tenga el tamaño correcto
            if (groups[g].objects.size() < static_cast<size_t>(groups[g].numObjects)) {
                groups[g].objects.resize(groups[g].numObjects);
                groups[g].targets.resize(groups[g].numObjects);
            }
            int n = groups[g].numObjects;
            for (int i = 0; i < n; ++i) {
                VisualObjectParams& obj = groups[g].objects[i];
                // Rotación automática
                if (autoRotate) {
                    obj.angle += obj.rotationSpeed * deltaTime * (3.14159265f / 180.0f);
                    if (obj.angle > 2.0f * 3.14159265f) obj.angle -= 2.0f * 3.14159265f;
                    if (obj.angle < 0.0f) obj.angle += 2.0f * 3.14159265f;
                }
                // INDEPENDENT COLOR ANIMATION: Each group has different color phases
                if (animateColor) {
                    float t = currentTime;
                    float phase = beatPhase + (float)i * 0.3f + (float)g * 0.5f; // Different phase per object and group
                    float groupPhaseOffset = (float)g * 2.0f * 3.14159265f / 3.0f; // 120° offset per group
                    float objectPhase = phase + groupPhaseOffset;
                    if (g == 0) { // Center group - Red dominant
                        obj.colorTop.x = 0.7f + 0.3f * sin(2.0f * 3.14159265f * objectPhase);
                        obj.colorTop.y = 0.2f + 0.2f * sin(2.0f * 3.14159265f * objectPhase + 1.0f);
                        obj.colorTop.z = 0.2f + 0.2f * sin(2.0f * 3.14159265f * objectPhase + 2.0f);
                    } else if (g == 1) { // Right group - Green dominant
                        obj.colorTop.x = 0.2f + 0.2f * sin(2.0f * 3.14159265f * objectPhase + 1.0f);
                        obj.colorTop.y = 0.7f + 0.3f * sin(2.0f * 3.14159265f * objectPhase);
                        obj.colorTop.z = 0.2f + 0.2f * sin(2.0f * 3.14159265f * objectPhase + 2.0f);
                    } else { // Left group - Blue dominant
                        obj.colorTop.x = 0.2f + 0.2f * sin(2.0f * 3.14159265f * objectPhase + 2.0f);
                        obj.colorTop.y = 0.2f + 0.2f * sin(2.0f * 3.14159265f * objectPhase + 1.0f);
                        obj.colorTop.z = 0.7f + 0.3f * sin(2.0f * 3.14159265f * objectPhase);
                    }
                    obj.colorLeft.x = 0.5f + 0.5f * sin(t + 1.0f + (float)i * 0.2f + groupPhaseOffset);
                    obj.colorLeft.y = 0.5f + 0.5f * sin(t + 3.0f + (float)i * 0.2f + groupPhaseOffset);
                    obj.colorLeft.z = 0.5f + 0.5f * sin(t + 5.0f + (float)i * 0.2f + groupPhaseOffset);
                    obj.colorRight.x = 0.5f + 0.5f * sin(t + 2.0f + (float)i * 0.2f + groupPhaseOffset);
                    obj.colorRight.y = 0.5f + 0.5f * sin(t + 4.0f + (float)i * 0.2f + groupPhaseOffset);
                    obj.colorRight.z = 0.5f + 0.5f * sin(t + 6.0f + (float)i * 0.2f + groupPhaseOffset);
                }
                // Si animateColor es false, NO modificar los colores (se quedan fijos)
            }
        }
        // 3. Randomización y recreación de shapes por grupo (MEJORADA)
        for (int g = 0; g < 3; ++g) {
            VisualObjectParams& obj = groups[g].objects[0];
            VisualObjectTargets& tgt = groups[g].targets[0];
            
            // AUDIO-DRIVEN RANDOMIZATION: Use audio frequencies to drive randomization
            float audioRandomFactor = 1.0f;
            if (audioReactive && !spectrum.empty()) {
                // Use different frequency bands for different groups
                if (g == 0) { // Center - Bass driven
                    audioRandomFactor = currentAudio.bass * 2.0f;
                } else if (g == 1) { // Right - Mid driven
                    audioRandomFactor = currentAudio.mid * 2.0f;
                } else { // Left - Treble driven
                    audioRandomFactor = currentAudio.treble * 2.0f;
                }
                
                // Clamp audio factor
                audioRandomFactor = std::max(0.1f, std::min(3.0f, audioRandomFactor));
            }
            
            // Sistema de randomización más natural con intervalos variables
            bool shouldRandomize = false;
            if (randomize) {
                // Calcular si es momento de randomizar basado en intervalos variables
                float timeSinceLastRandom = currentTime - lastRandomizeTime[g];
                float currentInterval = randomizeIntervals[g] + randomizeVariation[g] * sin(currentTime * 0.3f + g);
                
                // AUDIO-DRIVEN INTERVALS: Audio affects randomization frequency
                if (audioReactive && !spectrum.empty()) {
                    float audioIntensity = (currentAudio.bass + currentAudio.mid + currentAudio.treble) / 3.0f;
                    currentInterval *= (1.0f - audioIntensity * 0.5f); // Faster randomization with more audio
                    currentInterval = std::max(0.1f, currentInterval); // Minimum interval
                }
                
                if (timeSinceLastRandom >= currentInterval) {
                    shouldRandomize = true;
                    lastRandomizeTime[g] = currentTime;
                    
                    // Variar el siguiente intervalo para hacerlo menos predecible
                    randomizeIntervals[g] = 1.0f + frand() * 4.0f; // 1-5 segundos
                    randomizeVariation[g] = 0.2f + frand() * 1.0f; // 0.2-1.2 segundos de variación
                }
            }
            
            // Randomizar shapeType por grupo - MEJORADO para más variedad
            static int tgtShapeType[3] = {obj.shapeType, obj.shapeType, obj.shapeType};
            if (shouldRandomize && randomAffect.shapeType) {
                // Crear una distribución más interesante de tipos de formas
                float shapeChoice = frand();
                if (shapeChoice < 0.3f) {
                    tgtShapeType[g] = SHAPE_TRIANGLE; // 30% triángulos
                } else if (shapeChoice < 0.5f) {
                    tgtShapeType[g] = SHAPE_SQUARE;   // 20% cuadrados
                } else if (shapeChoice < 0.8f) {
                    tgtShapeType[g] = SHAPE_CIRCLE;   // 30% círculos
                } else if (shapeChoice < 0.95f) {
                    tgtShapeType[g] = SHAPE_LINE;     // 15% líneas
                } else {
                    tgtShapeType[g] = SHAPE_LONG_LINES; // 5% líneas largas
                }
                
                // Asegurar que no se repita el mismo tipo inmediatamente
                if (tgtShapeType[g] == obj.shapeType) {
                    tgtShapeType[g] = (obj.shapeType + 1) % SHAPE_COUNT;
                }
            }
            obj.shapeType += (int)((tgtShapeType[g] - obj.shapeType) * randomLerpSpeed * audioRandomFactor + 0.5f);
            
            // Randomizar nSegments por grupo
            static int tgtNSegments[3] = {obj.nSegments, obj.nSegments, obj.nSegments};
            if (shouldRandomize && randomAffect.nSegments) {
                int min = randomLimits.segMin;
                int max = randomLimits.segMax;
                tgtNSegments[g] = min + rand() % (max - min + 1);
            }
            obj.nSegments += (int)((tgtNSegments[g] - obj.nSegments) * randomLerpSpeed * audioRandomFactor + 0.5f);
            
            // Inicializar targets si es la primera vez
            if (tgt.target.triSize == 0.0f) tgt.target = obj;
            
            if (randomize) {
                // AUDIO-DRIVEN RANDOMIZATION: Apply audio factor to all random changes
                float adjustedLerpSpeed = randomLerpSpeed * audioRandomFactor;
                
                // triSize - only if selected
                if (shouldRandomize && randomAffect.triSize) {
                    tgt.target.triSize = randomLimits.sizeMin + frand() * (randomLimits.sizeMax - randomLimits.sizeMin);
                    obj.triSize += (tgt.target.triSize - obj.triSize) * adjustedLerpSpeed;
                }
                
                // rotationSpeed - only if selected
                if (shouldRandomize && randomAffect.rotationSpeed) {
                    tgt.target.rotationSpeed = randomLimits.speedMin + frand() * (randomLimits.speedMax - randomLimits.speedMin);
                    obj.rotationSpeed += (tgt.target.rotationSpeed - obj.rotationSpeed) * adjustedLerpSpeed;
                }
                
                // angle - only if selected
                if (shouldRandomize && randomAffect.angle) {
                    tgt.target.angle = frand() * 2.0f * 3.14159265f;
                    obj.angle += (tgt.target.angle - obj.angle) * adjustedLerpSpeed;
                }
                
                // translateX - only if selected
                if (shouldRandomize && randomAffect.translateX) {
                    tgt.target.translateX = randomLimits.txMin + frand() * (randomLimits.txMax - randomLimits.txMin);
                    obj.translateX += (tgt.target.translateX - obj.translateX) * adjustedLerpSpeed;
                }
                
                // translateY - only if selected
                if (shouldRandomize && randomAffect.translateY) {
                    tgt.target.translateY = randomLimits.tyMin + frand() * (randomLimits.tyMax - randomLimits.tyMin);
                    obj.translateY += (tgt.target.translateY - obj.translateY) * adjustedLerpSpeed;
                }
                
                // scaleX - only if selected
                if (shouldRandomize && randomAffect.scaleX) {
                    tgt.target.scaleX = randomLimits.sxMin + frand() * (randomLimits.sxMax - randomLimits.sxMin);
                    obj.scaleX += (tgt.target.scaleX - obj.scaleX) * adjustedLerpSpeed;
                }
                
                // scaleY - only if selected
                if (shouldRandomize && randomAffect.scaleY) {
                    tgt.target.scaleY = randomLimits.syMin + frand() * (randomLimits.syMax - randomLimits.syMin);
                    obj.scaleY += (tgt.target.scaleY - obj.scaleY) * adjustedLerpSpeed;
                }
                
                // colorTop - only if selected
                if (shouldRandomize && randomAffect.colorTop) {
                    for (int c = 0; c < 3; ++c) {
                        ((float*)&tgt.target.colorTop)[c] = randomLimits.colorMin + frand() * (randomLimits.colorMax - randomLimits.colorMin);
                    }
                    for (int c = 0; c < 3; ++c) {
                        ((float*)&obj.colorTop)[c] += (((float*)&tgt.target.colorTop)[c] - ((float*)&obj.colorTop)[c]) * adjustedLerpSpeed;
                    }
                }
                
                // colorLeft - only if selected
                if (shouldRandomize && randomAffect.colorLeft) {
                    for (int c = 0; c < 3; ++c) {
                        ((float*)&tgt.target.colorLeft)[c] = randomLimits.colorMin + frand() * (randomLimits.colorMax - randomLimits.colorMin);
                    }
                    for (int c = 0; c < 3; ++c) {
                        ((float*)&obj.colorLeft)[c] += (((float*)&tgt.target.colorLeft)[c] - ((float*)&obj.colorLeft)[c]) * adjustedLerpSpeed;
                    }
                }
                
                // colorRight - only if selected
                if (shouldRandomize && randomAffect.colorRight) {
                    for (int c = 0; c < 3; ++c) {
                        ((float*)&tgt.target.colorRight)[c] = randomLimits.colorMin + frand() * (randomLimits.colorMax - randomLimits.colorMin);
                    }
                    for (int c = 0; c < 3; ++c) {
                        ((float*)&obj.colorRight)[c] += (((float*)&tgt.target.colorRight)[c] - ((float*)&obj.colorRight)[c]) * adjustedLerpSpeed;
                    }
                }
                
                // groupAngle para cada grupo - only if selected
                if (shouldRandomize && randomAffect.groupAngle) {
                    static float tgtGroupAngle[3] = {groups[0].groupAngle, groups[1].groupAngle, groups[2].groupAngle};
                    tgtGroupAngle[g] = frand() * 2.0f * 3.14159265f;
                    groups[g].groupAngle += (tgtGroupAngle[g] - groups[g].groupAngle) * adjustedLerpSpeed;
                }
                
                            // Randomizar cantidad de objetos por grupo - only if selected
            if (shouldRandomize) {
                static int tgtNumObjects[3] = {groups[0].numObjects, groups[1].numObjects, groups[2].numObjects};
                
                // Configuración especial para el túnel psicodélico
                bool isTunnelPreset = false;
                for (const auto& preset : animationPresets) {
                    if (preset.name.find("Túnel Psicodélico") != std::string::npos) {
                        isTunnelPreset = true;
                        break;
                    }
                }
                
                if (randomAffect.numCenter && g == 0) {
                    if (isTunnelPreset) {
                        // Para el túnel, cambios más dramáticos en la cantidad
                        float tunnelChoice = frand();
                        if (tunnelChoice < 0.3f) {
                            tgtNumObjects[g] = 5 + rand() % 10; // 5-15 objetos
                        } else if (tunnelChoice < 0.7f) {
                            tgtNumObjects[g] = 20 + rand() % 30; // 20-50 objetos
                        } else {
                            tgtNumObjects[g] = 50 + rand() % 50; // 50-100 objetos
                        }
                    } else {
                        tgtNumObjects[g] = randomLimits.numCenterMin + rand() % (randomLimits.numCenterMax - randomLimits.numCenterMin + 1);
                    }
                    groups[g].numObjects += (int)((tgtNumObjects[g] - groups[g].numObjects) * adjustedLerpSpeed + 0.5f);
                } else if (randomAffect.numRight && g == 1) {
                    if (isTunnelPreset) {
                        float tunnelChoice = frand();
                        if (tunnelChoice < 0.4f) {
                            tgtNumObjects[g] = 8 + rand() % 12; // 8-20 objetos
                        } else if (tunnelChoice < 0.8f) {
                            tgtNumObjects[g] = 25 + rand() % 25; // 25-50 objetos
                        } else {
                            tgtNumObjects[g] = 60 + rand() % 40; // 60-100 objetos
                        }
                    } else {
                        tgtNumObjects[g] = randomLimits.numRightMin + rand() % (randomLimits.numRightMax - randomLimits.numRightMin + 1);
                    }
                    groups[g].numObjects += (int)((tgtNumObjects[g] - groups[g].numObjects) * adjustedLerpSpeed + 0.5f);
                } else if (randomAffect.numLeft && g == 2) {
                    if (isTunnelPreset) {
                        float tunnelChoice = frand();
                        if (tunnelChoice < 0.4f) {
                            tgtNumObjects[g] = 8 + rand() % 12; // 8-20 objetos
                        } else if (tunnelChoice < 0.8f) {
                            tgtNumObjects[g] = 25 + rand() % 25; // 25-50 objetos
                        } else {
                            tgtNumObjects[g] = 60 + rand() % 40; // 60-100 objetos
                        }
                    } else {
                        tgtNumObjects[g] = randomLimits.numLeftMin + rand() % (randomLimits.numLeftMax - randomLimits.numLeftMin + 1);
                    }
                    groups[g].numObjects += (int)((tgtNumObjects[g] - groups[g].numObjects) * adjustedLerpSpeed + 0.5f);
                }
                groups[g].numObjects = std::max(0, std::min(100, groups[g].numObjects)); // Limitar a 0-100
            }
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
            // OPTIMIZATION: Reduce fractal regeneration frequency
            if (fractalMode) {
                static float lastFractalUpdate = 0.0f;
                float fractalUpdateInterval = 0.1f; // Update every 100ms instead of every frame
                shouldRegenerate = shouldRegenerate || (currentTime - lastFractalUpdate > fractalUpdateInterval);
                if (shouldRegenerate) lastFractalUpdate = currentTime;
            }
            
            if (shouldRegenerate) {
                // OPTIMIZATION: Only delete and recreate if we have a valid cached VBO
                if (currentCachedVBO) {
                    // Find and remove the current VBO from cache
                    for (auto it = vboCache.begin(); it != vboCache.end(); ++it) {
                        if (&(*it) == currentCachedVBO) {
                            if (it->VAO) glDeleteVertexArrays(1, &it->VAO);
                            if (it->VBO) glDeleteBuffers(1, &it->VBO);
                            if (it->instanceVBO) glDeleteBuffers(1, &it->instanceVBO);
                            vboCache.erase(it);
                            break;
                        }
                    }
                    currentCachedVBO = nullptr;
                }
                
                // Antes de crear el shape, si onlyRGB está activo, forzar colores a RGB puros
                if (onlyRGB) {
                    curColorTop[0] = 1.0f; curColorTop[1] = 0.0f; curColorTop[2] = 0.0f;
                    curColorLeft[0] = 0.0f; curColorLeft[1] = 1.0f; curColorLeft[2] = 0.0f;
                    curColorRight[0] = 0.0f; curColorRight[1] = 0.0f; curColorRight[2] = 1.0f;
                }
                
                // OPTIMIZATION: Create new cached VBO
                float newColors[9] = {curColorTop[0], curColorTop[1], curColorTop[2],
                                    curColorLeft[0], curColorLeft[1], curColorLeft[2],
                                    curColorRight[0], curColorRight[1], curColorRight[2]};
                
                currentCachedVBO = findOrCreateCachedVBO(
                    obj.shapeType, obj.triSize, newColors, actualSegments, fractalMode, fractalDepth
                );
                
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

        // OPTIMIZATION: Clear screen once at the beginning
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // OPTIMIZATION: Prepare instance data for all groups
        std::vector<InstanceData> allInstances;
        allInstances.clear();
        
        // AUDIO TEST MODE: Render single test triangle if enabled
        if (audioTestMode.enabled) {
            // Create single test instance
            InstanceData testInstance;
            testInstance.offsetX = audioTestMode.testPosX;
            testInstance.offsetY = audioTestMode.testPosY;
            testInstance.angle = audioTestMode.testRotation * (3.14159265f / 180.0f); // Convert to radians
            testInstance.scaleX = audioTestMode.testSize;
            testInstance.scaleY = audioTestMode.testSize;
            allInstances.push_back(testInstance);
            
            // Use test colors for VBO
            float testColors[9] = {
                audioTestMode.testColor.x, audioTestMode.testColor.y, audioTestMode.testColor.z,
                audioTestMode.testColor.x, audioTestMode.testColor.y, audioTestMode.testColor.z,
                audioTestMode.testColor.x, audioTestMode.testColor.y, audioTestMode.testColor.z
            };
            
            // Create or update VBO for test triangle
            CachedVBO* testVBO = findOrCreateCachedVBO(
                SHAPE_TRIANGLE, // Always triangle for test
                audioTestMode.testSize,
                testColors,
                3, // Triangle has 3 vertices
                false, // No fractal for test
                0.0f
            );
            
            // Render test triangle
            if (testVBO && !allInstances.empty()) {
                renderBatch(testVBO, allInstances, shaderProgram, (float)width / (float)height);
            }
        } else {
            // Normal rendering for all groups
            for (int g = 0; g < 3; ++g) {
                VisualObjectParams& obj = groups[g].objects[0];
                float baseX = (g == 0) ? 0.0f : (g == 1) ? groupSeparation : -groupSeparation;
                
                // CENTERED RENDERING
                
                for (int i = 0; i < groups[g].numObjects; ++i) {
                    float theta = (2.0f * 3.14159265f * i) / std::max(1, groups[g].numObjects) + groups[g].groupAngle;
                    float r = 1.0f; // Use full normalized space
                    float tx = baseX + obj.translateX + r * cos(theta);
                    float ty = obj.translateY + r * sin(theta);
                    
                    InstanceData instance;
                    instance.offsetX = tx;
                    instance.offsetY = ty;
                    instance.angle = obj.angle;
                    instance.scaleX = obj.scaleX;
                    instance.scaleY = obj.scaleY;
                    
                    // --- NUEVO: Aplicar efecto glitch ---
                    if (glitchEffectEnabled && glitchActive) {
                        // Aplicar offset de glitch
                        instance.offsetX += glitchOffsetX;
                        instance.offsetY += glitchOffsetY;
                        
                        // Aplicar escala de glitch
                        instance.scaleX *= glitchScaleX;
                        instance.scaleY *= glitchScaleY;
                        
                        // Crear efecto de división: algunos objetos se dividen en dos
                        if (frand() < glitchSplitRatio) {
                            // Objeto original
                            allInstances.push_back(instance);
                            
                            // Objeto dividido (con offset adicional)
                            InstanceData splitInstance = instance;
                            splitInstance.offsetX += (frand() - 0.5f) * glitchIntensity * 0.3f;
                            splitInstance.offsetY += (frand() - 0.5f) * glitchIntensity * 0.3f;
                            splitInstance.scaleX *= 0.7f; // Más pequeño
                            splitInstance.scaleY *= 0.7f;
                            allInstances.push_back(splitInstance);
                        } else {
                            allInstances.push_back(instance);
                        }
                    } else {
                        allInstances.push_back(instance);
                    }
                }
            }
            
            // OPTIMIZATION: Update cached VBO if needed
            float colors[9] = {colorTopArr[0], colorTopArr[1], colorTopArr[2],
                              colorLeftArr[0], colorLeftArr[1], colorLeftArr[2],
                              colorRightArr[0], colorRightArr[1], colorRightArr[2]};
            
            bool needNewVBO = false;
            if (!currentCachedVBO || 
                currentCachedVBO->shapeType != groups[0].objects[0].shapeType ||
                currentCachedVBO->size != groups[0].objects[0].triSize ||
                currentCachedVBO->fractalMode != fractalMode ||
                currentCachedVBO->fractalDepth != fractalDepth) {
                needNewVBO = true;
            }
            
            for (int i = 0; i < 9; ++i) {
                if (fabs(currentCachedVBO->colors[i] - colors[i]) > 0.001f) {
                    needNewVBO = true;
                    break;
                }
            }
            
            if (needNewVBO) {
                currentCachedVBO = findOrCreateCachedVBO(
                    groups[0].objects[0].shapeType,
                    groups[0].objects[0].triSize,
                    colors,
                    groups[0].objects[0].nSegments,
                    fractalMode,
                    fractalDepth
                );
            }
            
            // OPTIMIZATION: Render all instances in one batch
            if (currentCachedVBO && !allInstances.empty()) {
                renderBatch(currentCachedVBO, allInstances, shaderProgram, (float)width / (float)height);
            }
        }

        // FPS custom: sleep si es necesario
        if (fpsMode == FPS_CUSTOM && customFps > 0) {
            float frameTime = 1.0f / (float)customFps;
            float elapsed = glfwGetTime() - currentTime;
            if (elapsed < frameTime) {
                int ms = (int)((frameTime - elapsed) * 1000.0f);
                if (ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(ms));
            }
        }

        // AUDIO TEST MODE WINDOW: Para probar audio reactivo fácilmente
        if (uiVisibility.showAudioTestMode) {
            ImGui::SetNextWindowPos(ImVec2(width - 400, height - 400), ImGuiCond_Once);
            ImGui::SetNextWindowSize(ImVec2(380, 380), ImGuiCond_Once);
            ImGui::Begin("🧪 Modo de Prueba de Audio");
            
            // Estado del modo de prueba
            ImGui::Text("🎯 Modo de Prueba de Audio Reactivo");
            ImGui::Checkbox("Activar Modo de Prueba", &audioTestMode.enabled);
            
            if (audioTestMode.enabled) {
                ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "✅ MODO ACTIVO - Solo se muestra 1 triángulo de prueba");
            } else {
                ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "❌ MODO INACTIVO - Visualización normal");
            }
            
            ImGui::Separator();
            
            // Controles de audio
            ImGui::Text("🎵 Fuente de Audio:");
            ImGui::Checkbox("Usar valores manuales (simular audio)", &audioTestMode.useManualValues);
            
            if (audioTestMode.useManualValues) {
                ImGui::Text("🎛️ Controles Manuales:");
                ImGui::SliderFloat("Bass Manual", &audioTestMode.manualBass, 0.0f, 1.0f, "%.2f");
                ImGui::SliderFloat("Mid Manual", &audioTestMode.manualMid, 0.0f, 1.0f, "%.2f");
                ImGui::SliderFloat("Treble Manual", &audioTestMode.manualTreble, 0.0f, 1.0f, "%.2f");
            } else {
                ImGui::Text("📊 Valores Reales de Audio:");
                ImGui::Text("Bass: %.3f", audioTestMode.bassTest);
                ImGui::Text("Mid: %.3f", audioTestMode.midTest);
                ImGui::Text("Treble: %.3f", audioTestMode.trebleTest);
                ImGui::Text("Overall: %.3f", audioTestMode.overallTest);
            }
            
            ImGui::Separator();
            
            // Controles de efectos
            ImGui::Text("🎨 Efectos a Probar:");
            ImGui::Checkbox("Color (RGB = Bass/Mid/Treble)", &audioTestMode.testColorEnabled);
            ImGui::Checkbox("Tamaño (Overall)", &audioTestMode.testSizeEnabled);
            ImGui::Checkbox("Rotación (Mid)", &audioTestMode.testRotationEnabled);
            ImGui::Checkbox("Posición (Bass/Treble)", &audioTestMode.testPositionEnabled);
            ImGui::Checkbox("Cantidad (Overall)", &audioTestMode.testQuantityEnabled);
            
            ImGui::Separator();
            
            // Valores actuales del objeto de prueba
            ImGui::Text("📐 Valores Actuales del Objeto:");
            ImGui::Text("Tamaño: %.2f", audioTestMode.testSize);
            ImGui::Text("Rotación: %.1f°", audioTestMode.testRotation);
            ImGui::Text("Posición: (%.2f, %.2f)", audioTestMode.testPosX, audioTestMode.testPosY);
            ImGui::Text("Cantidad: %d", audioTestMode.testQuantity);
            ImGui::ColorEdit3("Color", (float*)&audioTestMode.testColor);
            
            ImGui::Separator();
            
            // Controles rápidos
            ImGui::Text("⚡ Controles Rápidos:");
            if (ImGui::Button("Reset Objeto")) {
                audioTestMode.reset();
            }
            ImGui::SameLine();
            if (ImGui::Button("Test Bass")) {
                audioTestMode.useManualValues = true;
                audioTestMode.manualBass = 1.0f;
                audioTestMode.manualMid = 0.0f;
                audioTestMode.manualTreble = 0.0f;
            }
            ImGui::SameLine();
            if (ImGui::Button("Test Mid")) {
                audioTestMode.useManualValues = true;
                audioTestMode.manualBass = 0.0f;
                audioTestMode.manualMid = 1.0f;
                audioTestMode.manualTreble = 0.0f;
            }
            ImGui::SameLine();
            if (ImGui::Button("Test Treble")) {
                audioTestMode.useManualValues = true;
                audioTestMode.manualBass = 0.0f;
                audioTestMode.manualMid = 0.0f;
                audioTestMode.manualTreble = 1.0f;
            }
            
            ImGui::End();
        }

        // AUDIO TEST MODE: Update test mode with current audio data
        audioTestMode.updateFromAudio(currentAudio);
        
        // --- NUEVO: Modo Fractal Toggle ---
        if (fractalToggleMode && currentTime - lastFractalToggleTime >= fractalToggleInterval) {
            fractalToggleState = !fractalToggleState;
            fractalMode = fractalToggleState;
            lastFractalToggleTime = currentTime;
            
            // Forzar regeneración de VBO cuando cambia el modo fractal
            currentCachedVBO = nullptr;
        }
        
        // --- NUEVO: Efecto Glitch ---
        if (glitchEffectEnabled) {
            // Calcular frecuencia del glitch basada en audio
            float glitchFreq = glitchFrequency;
            if (audioReactive) {
                glitchFreq *= (1.0f + currentAudio.overall * 2.0f); // Más glitch con más audio
            }
            
            // Activar glitch basado en frecuencia
            if (currentTime - lastGlitchTime >= (1.0f / glitchFreq)) {
                glitchActive = true;
                lastGlitchTime = currentTime;
                
                // Calcular parámetros del glitch
                glitchOffsetX = (frand() - 0.5f) * glitchIntensity * 0.5f;
                glitchOffsetY = (frand() - 0.5f) * glitchIntensity * 0.5f;
                glitchScaleX = 1.0f + (frand() - 0.5f) * glitchIntensity;
                glitchScaleY = 1.0f + (frand() - 0.5f) * glitchIntensity;
                
                // Aplicar delay
                std::this_thread::sleep_for(std::chrono::milliseconds((int)(glitchDelay * 1000)));
            } else {
                glitchActive = false;
            }
        }
        
        // --- NUEVO: Randomización basada en frecuencias de música ---
        if (frequencyBasedRandomization && audioReactive) {
            // Randomización por Bass
            if (currentAudio.bass > bassRandomizationThreshold && 
                currentTime - lastBassRandomizeTime >= frequencyRandomizeCooldown) {
                // Randomizar parámetros basados en bass
                for (int g = 0; g < 3; ++g) {
                    if (randomAffect.triSize) {
                        groups[g].objects[0].triSize = randomLimits.sizeMin + 
                            frand() * (randomLimits.sizeMax - randomLimits.sizeMin);
                    }
                    if (randomAffect.rotationSpeed) {
                        groups[g].objects[0].rotationSpeed = randomLimits.speedMin + 
                            frand() * (randomLimits.speedMax - randomLimits.speedMin);
                    }
                }
                lastBassRandomizeTime = currentTime;
            }
            
            // Randomización por Mid
            if (currentAudio.mid > midRandomizationThreshold && 
                currentTime - lastMidRandomizeTime >= frequencyRandomizeCooldown) {
                // Randomizar colores basados en mid
                for (int g = 0; g < 3; ++g) {
                    if (randomAffect.colorTop) {
                        groups[g].objects[0].colorTop = ImVec4(frand(), frand(), frand(), 1.0f);
                    }
                    if (randomAffect.colorLeft) {
                        groups[g].objects[0].colorLeft = ImVec4(frand(), frand(), frand(), 1.0f);
                    }
                    if (randomAffect.colorRight) {
                        groups[g].objects[0].colorRight = ImVec4(frand(), frand(), frand(), 1.0f);
                    }
                }
                lastMidRandomizeTime = currentTime;
            }
            
            // Randomización por Treble
            if (currentAudio.treble > trebleRandomizationThreshold && 
                currentTime - lastTrebleRandomizeTime >= frequencyRandomizeCooldown) {
                // Randomizar posición y escala basados en treble
                for (int g = 0; g < 3; ++g) {
                    if (randomAffect.translateX) {
                        groups[g].objects[0].translateX = randomLimits.txMin + 
                            frand() * (randomLimits.txMax - randomLimits.txMin);
                    }
                    if (randomAffect.translateY) {
                        groups[g].objects[0].translateY = randomLimits.tyMin + 
                            frand() * (randomLimits.tyMax - randomLimits.tyMin);
                    }
                    if (randomAffect.scaleX) {
                        groups[g].objects[0].scaleX = randomLimits.sxMin + 
                            frand() * (randomLimits.sxMax - randomLimits.sxMin);
                    }
                    if (randomAffect.scaleY) {
                        groups[g].objects[0].scaleY = randomLimits.syMin + 
                            frand() * (randomLimits.syMax - randomLimits.syMin);
                    }
                }
                lastTrebleRandomizeTime = currentTime;
            }
        }
        
        // AUTO-RANDOMIZATION: Check if we need to randomize presets
        // --- NUEVO: Usar randomización basada en frecuencias si está habilitada ---
        float currentRandomizeInterval = presetRandomizeInterval;
        if (frequencyBasedRandomization && audioReactive) {
            // Calcular intervalo basado en la intensidad del audio
            float audioIntensity = currentAudio.overall;
            currentRandomizeInterval = 5.0f / (1.0f + audioIntensity * 3.0f); // 5s a ~1.25s
            currentRandomizeInterval = std::max(0.5f, std::min(10.0f, currentRandomizeInterval)); // Limitar entre 0.5s y 10s
        }
        
        if (autoRandomizePresets && currentTime - lastPresetRandomizeTime >= currentRandomizeInterval) {
            // Create filtered list of presets based on options
            std::vector<int> availablePresets;
            for (int i = 0; i < animationPresets.size(); ++i) {
                const AnimationPreset& preset = animationPresets[i];
                bool includePreset = true;
                
                // Apply filters
                if (randomizeOnlyFractals) {
                    includePreset = preset.center.fractalMode || preset.right.fractalMode || preset.left.fractalMode;
                } else if (randomizeOnlyLines) {
                    includePreset = (preset.center.shapeType == SHAPE_LINE || preset.center.shapeType == SHAPE_LONG_LINES) ||
                                   (preset.right.shapeType == SHAPE_LINE || preset.right.shapeType == SHAPE_LONG_LINES) ||
                                   (preset.left.shapeType == SHAPE_LINE || preset.left.shapeType == SHAPE_LONG_LINES);
                } else if (randomizeOnlyCylinders) {
                    includePreset = (preset.center.shapeType == SHAPE_CIRCLE) ||
                                   (preset.right.shapeType == SHAPE_CIRCLE) ||
                                   (preset.left.shapeType == SHAPE_CIRCLE);
                }
                
                if (includePreset) {
                    availablePresets.push_back(i);
                }
            }
            
            // If no presets match the filter, use all presets
            if (availablePresets.empty()) {
                for (int i = 0; i < animationPresets.size(); ++i) {
                    availablePresets.push_back(i);
                }
            }
            
            // Randomly select from filtered presets
            int randomIndex = rand() % availablePresets.size();
            int randomPresetIndex = availablePresets[randomIndex];
            const AnimationPreset& randomPreset = animationPresets[randomPresetIndex];
            
            // Apply the random preset
            randomPreset.apply(groups, autoRotate, randomize, audioReactive, bpm, groupSeparation, randomLimits, randomAffect);
            
            // Apply audio preset if audio reactive
            if (randomPreset.audioReactive && randomPreset.audioPresetIndex < audioPresets.size()) {
                for (int g = 0; g < 3; ++g) {
                    applyAudioPreset(audioGroups[g], audioPresets[randomPreset.audioPresetIndex]);
                }
            }
            
            // Set fractal mode if any group uses it
            fractalMode = randomPreset.center.fractalMode || randomPreset.right.fractalMode || randomPreset.left.fractalMode;
            if (fractalMode) {
                fractalDepth = randomPreset.center.fractalDepth; // Use center as default
            }
            
            // Force VBO regeneration
            currentCachedVBO = nullptr;
            
            // Update last randomize time
            lastPresetRandomizeTime = currentTime;
        }

        // PRESETS WINDOW: Predefined animation configurations
        if (uiVisibility.showPresets) {
            ImGui::SetNextWindowPos(ImVec2(width - 450, height - 600), ImGuiCond_Once);
            ImGui::SetNextWindowSize(ImVec2(430, 580), ImGuiCond_Once);
            ImGui::Begin("🎨 Presets de Animación");
            
            ImGui::Text("🌟 Presets Predefinidos");
            ImGui::Text("Selecciona una animación para aplicarla instantáneamente");
            ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "🎲 ¡Todos los presets incluyen randomización automática!");
            ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.0f, 1.0f), "💡 La randomización se configura automáticamente según el tipo de preset");
            ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "🌀 ¡Nuevo! Preset 'Túnel Psicodélico' con randomización extrema");
            
            // Randomize presets button
            if (ImGui::Button("🎲 Randomizar Presets Activos")) {
                // Create filtered list of presets based on options
                std::vector<int> availablePresets;
                for (int i = 0; i < animationPresets.size(); ++i) {
                    const AnimationPreset& preset = animationPresets[i];
                    bool includePreset = true;
                    
                    // Apply filters
                    if (randomizeOnlyFractals) {
                        includePreset = preset.center.fractalMode || preset.right.fractalMode || preset.left.fractalMode;
                    } else if (randomizeOnlyLines) {
                        includePreset = (preset.center.shapeType == SHAPE_LINE || preset.center.shapeType == SHAPE_LONG_LINES) ||
                                       (preset.right.shapeType == SHAPE_LINE || preset.right.shapeType == SHAPE_LONG_LINES) ||
                                       (preset.left.shapeType == SHAPE_LINE || preset.left.shapeType == SHAPE_LONG_LINES);
                    } else if (randomizeOnlyCylinders) {
                        includePreset = (preset.center.shapeType == SHAPE_CIRCLE) ||
                                       (preset.right.shapeType == SHAPE_CIRCLE) ||
                                       (preset.left.shapeType == SHAPE_CIRCLE);
                    }
                    
                    if (includePreset) {
                        availablePresets.push_back(i);
                    }
                }
                
                // If no presets match the filter, use all presets
                if (availablePresets.empty()) {
                    for (int i = 0; i < animationPresets.size(); ++i) {
                        availablePresets.push_back(i);
                    }
                }
                
                // Randomly select from filtered presets
                int randomIndex = rand() % availablePresets.size();
                int randomPresetIndex = availablePresets[randomIndex];
                const AnimationPreset& randomPreset = animationPresets[randomPresetIndex];
                
                // Apply the random preset
                randomPreset.apply(groups, autoRotate, randomize, audioReactive, bpm, groupSeparation, randomLimits, randomAffect);
                
                // Apply audio preset if audio reactive
                if (randomPreset.audioReactive && randomPreset.audioPresetIndex < audioPresets.size()) {
                    for (int g = 0; g < 3; ++g) {
                        applyAudioPreset(audioGroups[g], audioPresets[randomPreset.audioPresetIndex]);
                    }
                }
                
                // Set fractal mode if any group uses it
                fractalMode = randomPreset.center.fractalMode || randomPreset.right.fractalMode || randomPreset.left.fractalMode;
                if (fractalMode) {
                    fractalDepth = randomPreset.center.fractalDepth; // Use center as default
                }
                
                // Force VBO regeneration
                currentCachedVBO = nullptr;
                
                ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "✅ Preset aleatorio aplicado: %s", randomPreset.name.c_str());
            }
            ImGui::SameLine();
            if (ImGui::Button("🔄 Randomizar Cada 5s")) {
                // Toggle auto-randomization
                autoRandomizePresets = !autoRandomizePresets;
                if (autoRandomizePresets) {
                    lastPresetRandomizeTime = currentTime; // Reset timer
                    ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "🔄 Auto-randomización activada");
                } else {
                    ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "⏹️ Auto-randomización desactivada");
                }
            }
            
            // Show auto-randomization status
            if (autoRandomizePresets) {
                float timeUntilNext = presetRandomizeInterval - (currentTime - lastPresetRandomizeTime);
                if (timeUntilNext < 0) timeUntilNext = 0;
                ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "⏱️ Próximo preset en: %.1fs", timeUntilNext);
            }
            
            // Interval control
            ImGui::SliderFloat("Intervalo (segundos)", &presetRandomizeInterval, 1.0f, 30.0f, "%.1f");
            
            // Randomization options
            ImGui::Text("🎯 Opciones de Randomización:");
            
            ImGui::Checkbox("Solo Fractales", &randomizeOnlyFractals);
            ImGui::SameLine();
            ImGui::Checkbox("Solo Líneas", &randomizeOnlyLines);
            ImGui::SameLine();
            ImGui::Checkbox("Solo Cilindros", &randomizeOnlyCylinders);
            
            // Additional randomization features
            if (ImGui::Button("🎲 Randomizar 3 Presets")) {
                // Apply 3 random presets in sequence
                for (int i = 0; i < 3; ++i) {
                    // Create filtered list
                    std::vector<int> availablePresets;
                    for (int j = 0; j < animationPresets.size(); ++j) {
                        const AnimationPreset& preset = animationPresets[j];
                        bool includePreset = true;
                        
                        if (randomizeOnlyFractals) {
                            includePreset = preset.center.fractalMode || preset.right.fractalMode || preset.left.fractalMode;
                        } else if (randomizeOnlyLines) {
                            includePreset = (preset.center.shapeType == SHAPE_LINE || preset.center.shapeType == SHAPE_LONG_LINES) ||
                                           (preset.right.shapeType == SHAPE_LINE || preset.right.shapeType == SHAPE_LONG_LINES) ||
                                           (preset.left.shapeType == SHAPE_LINE || preset.left.shapeType == SHAPE_LONG_LINES);
                        } else if (randomizeOnlyCylinders) {
                            includePreset = (preset.center.shapeType == SHAPE_CIRCLE) ||
                                           (preset.right.shapeType == SHAPE_CIRCLE) ||
                                           (preset.left.shapeType == SHAPE_CIRCLE);
                        }
                        
                        if (includePreset) {
                            availablePresets.push_back(j);
                        }
                    }
                    
                    if (availablePresets.empty()) {
                        for (int j = 0; j < animationPresets.size(); ++j) {
                            availablePresets.push_back(j);
                        }
                    }
                    
                    int randomIndex = rand() % availablePresets.size();
                    int randomPresetIndex = availablePresets[randomIndex];
                    const AnimationPreset& randomPreset = animationPresets[randomPresetIndex];
                    
                    // Apply the preset
                    randomPreset.apply(groups, autoRotate, randomize, audioReactive, bpm, groupSeparation, randomLimits, randomAffect);
                    
                    if (randomPreset.audioReactive && randomPreset.audioPresetIndex < audioPresets.size()) {
                        for (int g = 0; g < 3; ++g) {
                            applyAudioPreset(audioGroups[g], audioPresets[randomPreset.audioPresetIndex]);
                        }
                    }
                    
                    fractalMode = randomPreset.center.fractalMode || randomPreset.right.fractalMode || randomPreset.left.fractalMode;
                    if (fractalMode) {
                        fractalDepth = randomPreset.center.fractalDepth;
                    }
                }
                
                currentCachedVBO = nullptr;
                ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "✅ 3 presets aleatorios aplicados!");
            }
            
            // Show statistics
            int totalPresets = animationPresets.size();
            int availablePresets = totalPresets;
            
            if (randomizeOnlyFractals || randomizeOnlyLines || randomizeOnlyCylinders) {
                availablePresets = 0;
                for (const auto& preset : animationPresets) {
                    bool includePreset = true;
                    
                    if (randomizeOnlyFractals) {
                        includePreset = preset.center.fractalMode || preset.right.fractalMode || preset.left.fractalMode;
                    } else if (randomizeOnlyLines) {
                        includePreset = (preset.center.shapeType == SHAPE_LINE || preset.center.shapeType == SHAPE_LONG_LINES) ||
                                       (preset.right.shapeType == SHAPE_LINE || preset.right.shapeType == SHAPE_LONG_LINES) ||
                                       (preset.left.shapeType == SHAPE_LINE || preset.left.shapeType == SHAPE_LONG_LINES);
                    } else if (randomizeOnlyCylinders) {
                        includePreset = (preset.center.shapeType == SHAPE_CIRCLE) ||
                                       (preset.right.shapeType == SHAPE_CIRCLE) ||
                                       (preset.left.shapeType == SHAPE_CIRCLE);
                    }
                    
                    if (includePreset) {
                        availablePresets++;
                    }
                }
            }
            
            ImGui::Text("📊 Estadísticas: %d/%d presets disponibles", availablePresets, totalPresets);
            
            // Randomization info
            ImGui::Separator();
            ImGui::Text("🎲 Información de Randomización:");
            ImGui::Text("• Fractales: Randomización extrema (más variación)");
            ImGui::Text("• Líneas: Randomización moderada (movimiento fluido)");
            ImGui::Text("• Círculos: Randomización balanceada (equilibrio)");
            ImGui::Text("• Túnel Psicodélico: Randomización extrema (efecto psicodélico)");
            ImGui::Text("• Otros: Randomización estándar (versatilidad)");
            ImGui::Text("• Tipos de objetos: Cambian dinámicamente entre triángulos, cuadrados, círculos, líneas");
            
            ImGui::Separator();
            
            // Display presets in a grid
            for (int i = 0; i < animationPresets.size(); ++i) {
                const AnimationPreset& preset = animationPresets[i];
                
                // Create a card-like layout for each preset
                ImGui::BeginChild(("preset_" + std::to_string(i)).c_str(), ImVec2(200, 120), true);
                
                // Preset name and description
                ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.0f, 1.0f), "%s", preset.name.c_str());
                ImGui::TextWrapped("%s", preset.description.c_str());
                
                // Preset details
                ImGui::Text("Centro: %s x%d", shapeNames[preset.center.shapeType], preset.center.numObjects);
                ImGui::Text("Derecha: %s x%d", shapeNames[preset.right.shapeType], preset.right.numObjects);
                ImGui::Text("Izquierda: %s x%d", shapeNames[preset.left.shapeType], preset.left.numObjects);
                
                // Audio preset info
                if (preset.audioReactive) {
                    ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "🎵 Audio: %s", 
                                      audioPresets[preset.audioPresetIndex].name.c_str());
                }
                
                // Apply button
                if (ImGui::Button(("Aplicar##" + std::to_string(i)).c_str())) {
                    // Apply the preset
                    preset.apply(groups, autoRotate, randomize, audioReactive, bpm, groupSeparation, randomLimits, randomAffect);
                    
                    // Apply audio preset if audio reactive
                    if (preset.audioReactive && preset.audioPresetIndex < audioPresets.size()) {
                        for (int g = 0; g < 3; ++g) {
                            applyAudioPreset(audioGroups[g], audioPresets[preset.audioPresetIndex]);
                        }
                    }
                    
                    // Set fractal mode if any group uses it
                    fractalMode = preset.center.fractalMode || preset.right.fractalMode || preset.left.fractalMode;
                    if (fractalMode) {
                        fractalDepth = preset.center.fractalDepth; // Use center as default
                    }
                    
                    // Force VBO regeneration
                    currentCachedVBO = nullptr;
                    
                    ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "✅ Preset aplicado!");
                }
                
                ImGui::EndChild();
                
                // Arrange in 2 columns
                if (i % 2 == 0 && i + 1 < animationPresets.size()) {
                    ImGui::SameLine();
                }
            }
            
            ImGui::Separator();
            
            // Quick preset categories
            ImGui::Text("⚡ Acceso Rápido por Categoría:");
            
            if (ImGui::Button("🎯 Cilindros y Donas")) {
                // Apply first two presets
                animationPresets[0].apply(groups, autoRotate, randomize, audioReactive, bpm, groupSeparation, randomLimits, randomAffect);
                for (int g = 0; g < 3; ++g) {
                    applyAudioPreset(audioGroups[g], audioPresets[3]); // Full Spectrum
                }
                currentCachedVBO = nullptr;
            }
            ImGui::SameLine();
            
            if (ImGui::Button("✨ Fractales")) {
                animationPresets[2].apply(groups, autoRotate, randomize, audioReactive, bpm, groupSeparation, randomLimits, randomAffect);
                fractalMode = true;
                fractalDepth = 4.0f;
                for (int g = 0; g < 3; ++g) {
                    applyAudioPreset(audioGroups[g], audioPresets[6]); // Chaos Mode
                }
                currentCachedVBO = nullptr;
            }
            ImGui::SameLine();
            
            if (ImGui::Button("⚡ Líneas")) {
                animationPresets[3].apply(groups, autoRotate, randomize, audioReactive, bpm, groupSeparation, randomLimits, randomAffect);
                for (int g = 0; g < 3; ++g) {
                    applyAudioPreset(audioGroups[g], audioPresets[4]); // Full Spectrum
                }
                currentCachedVBO = nullptr;
            }
            
            ImGui::SameLine();
            if (ImGui::Button("🌀 Vórtices")) {
                animationPresets[9].apply(groups, autoRotate, randomize, audioReactive, bpm, groupSeparation, randomLimits, randomAffect);
                fractalMode = true;
                fractalDepth = 3.0f;
                for (int g = 0; g < 3; ++g) {
                    applyAudioPreset(audioGroups[g], audioPresets[6]); // Chaos Mode
                }
                currentCachedVBO = nullptr;
            }
            
            ImGui::SameLine();
            if (ImGui::Button("🧠 Neural")) {
                animationPresets[10].apply(groups, autoRotate, randomize, audioReactive, bpm, groupSeparation, randomLimits, randomAffect);
                for (int g = 0; g < 3; ++g) {
                    applyAudioPreset(audioGroups[g], audioPresets[0]); // Bass Dominant
                }
                currentCachedVBO = nullptr;
            }
            
            ImGui::SameLine();
            if (ImGui::Button("🌀 Túnel")) {
                animationPresets[11].apply(groups, autoRotate, randomize, audioReactive, bpm, groupSeparation, randomLimits, randomAffect);
                fractalMode = true;
                fractalDepth = 3.5f;
                for (int g = 0; g < 3; ++g) {
                    applyAudioPreset(audioGroups[g], audioPresets[6]); // Chaos Mode
                }
                currentCachedVBO = nullptr;
            }
            
            ImGui::Separator();
            
            // Preset info
            ImGui::Text("💡 Información:");
            ImGui::Text("• Los presets incluyen configuraciones completas de audio");
            ImGui::Text("• Cada preset tiene colores y formas únicas");
            ImGui::Text("• Algunos presets activan automáticamente el modo fractal");
            ImGui::Text("• Los presets se pueden combinar con controles manuales");
            ImGui::Text("• Total de presets disponibles: %d", (int)animationPresets.size());
            
            ImGui::Separator();
            
            // Preset categories
            ImGui::Text("📂 Categorías de Presets:");
            ImGui::Text("🎯 Cilindros/Donas: Presets 1-2");
            ImGui::Text("✨ Fractales: Preset 3");
            ImGui::Text("⚡ Líneas: Presets 4, 8");
            ImGui::Text("🌊 Pulsos: Preset 5");
            ImGui::Text("🌌 Espirales: Preset 6");
            ImGui::Text("💎 Cristales: Preset 7");
            ImGui::Text("🌀 Vórtices: Preset 9");
            ImGui::Text("🧠 Neural: Preset 10");
            ImGui::Text("🌀 Túnel Psicodélico: Preset 11");
            
            ImGui::End();
        }

        // Render ImGui
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
        glfwPollEvents();
    } // End of main while loop

    // Cleanup ImGui
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    // OPTIMIZATION: Cleanup all cached VBOs
    for (auto& cached : vboCache) {
        if (cached.VAO) glDeleteVertexArrays(1, &cached.VAO);
        if (cached.VBO) glDeleteBuffers(1, &cached.VBO);
        if (cached.instanceVBO) glDeleteBuffers(1, &cached.instanceVBO);
    }
    vboCache.clear();
    
    glDeleteProgram(shaderProgram);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
} 