#pragma once

#include <GL/glew.h>
#include "visual_object.h"

// Forward declarations
class VisualEngine;

class VisualFractalEngine {
public:
    VisualFractalEngine();
    ~VisualFractalEngine();
    
    bool initialize();
    void shutdown();
    void update(const VisualEngine& engine);
    
    // Fractal control
    void setEnabled(bool enabled) { this->enabled = enabled; }
    bool isEnabled() const { return enabled; }
    
    void setDepth(float depth) { fractalDepth = depth; }
    float getDepth() const { return fractalDepth; }
    
    // Fractal generation
    void generateFractal(GLuint& VAO, GLuint& VBO, ShapeType shapeType, 
                        float size, const float* colorTop, const float* colorLeft, 
                        const float* colorRight, float depth, float time);
    
    // Fractal rendering
    void render(GLuint shaderProgram, const VisualEngine& engine, bool autoRotate, bool animateColor);
    
    // Fractal types
    enum class FractalType {
        SIERPINSKI_TRIANGLE,
        KOCH_SNOWFLAKE,
        MANDELBROT,
        JULIA_SET,
        CUSTOM
    };
    
    void setFractalType(FractalType type) { fractalType = type; }
    FractalType getFractalType() const { return fractalType; }
    
    // Audio-reactividad
    void setAudioLevel(float level) { audioLevel = level; }
    void setAudioSpectrum(const std::vector<float>& spec) { audioSpectrum = spec; }
    float getAudioLevel() const { return audioLevel; }
    const std::vector<float>& getAudioSpectrum() const { return audioSpectrum; }
    
private:
    bool enabled;
    float fractalDepth;
    FractalType fractalType;
    
    float audioLevel = 0.0f;
    std::vector<float> audioSpectrum;
    
    // Fractal generation methods
    void generateSierpinskiTriangle(GLuint& VAO, GLuint& VBO, float size, 
                                   const float* colors, float depth);
    void generateKochSnowflake(GLuint& VAO, GLuint& VBO, float size, 
                              const float* colors, float depth);
    void generateMandelbrot(GLuint& VAO, GLuint& VBO, float size, 
                           const float* colors, float depth, float time);
    void generateJuliaSet(GLuint& VAO, GLuint& VBO, float size, 
                         const float* colors, float depth, float time);
    
    // Helper methods
    void createFractalVertices(std::vector<float>& vertices, 
                              std::vector<unsigned int>& indices);
    void applyFractalTransformation(std::vector<float>& vertices, 
                                   const std::vector<float>& baseVertices, 
                                   float depth);
}; 