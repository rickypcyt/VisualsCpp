#pragma once

#include "visual_object.h"
#include <vector>
#include <memory>

class VisualGroup {
public:
    VisualGroup();
    ~VisualGroup();
    
    void initialize();
    void update(float deltaTime, bool autoRotate, bool animateColor, bool onlyRGB, float time, float phase);
    void render(GLuint shaderProgram, float aspect, float baseX, float baseY);
    
    // Object management
    void setNumObjects(int num);
    int getNumObjects() const { return numObjects; }
    void addObject();
    void removeObject();
    
    // Group properties
    float getGroupAngle() const { return groupAngle; }
    void setGroupAngle(float angle) { groupAngle = angle; }
    
    // Object access
    VisualObject* getObject(int index);
    const VisualObject* getObject(int index) const;
    VisualObject* getTemplateObject() { return &templateObject; }
    
    // Group positioning
    void setBasePosition(float x, float y) { baseX = x; baseY = y; }
    float getBaseX() const { return baseX; }
    float getBaseY() const { return baseY; }
    
    // Separation and layout
    void setSeparation(float sep) { separation = sep; }
    float getSeparation() const { return separation; }
    
    // Audio-reactividad
    void setAudioLevel(float level) { audioLevel = level; }
    void setAudioSpectrum(const std::vector<float>& spec) { audioSpectrum = spec; }
    float getAudioLevel() const { return audioLevel; }
    const std::vector<float>& getAudioSpectrum() const { return audioSpectrum; }
    
private:
    std::vector<std::unique_ptr<VisualObject>> objects;
    VisualObject templateObject;
    
    int numObjects;
    float groupAngle;
    float baseX, baseY;
    float separation;
    
    float audioLevel = 0.0f;
    std::vector<float> audioSpectrum;
    
    static const int MAX_OBJECTS;
    
    void updateObjectPositions();
    void ensureObjectsExist();
}; 