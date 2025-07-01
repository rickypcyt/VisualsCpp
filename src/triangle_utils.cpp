#include "triangle_utils.h"
#include <vector>
#include <cmath>
#include <functional>

void createShape(GLuint& VAO, GLuint& VBO, int shapeType, float size, float colorTop[3], float colorLeft[3], float colorRight[3], int nSegments) {
    // Limpiar VAO y VBO existentes
    if (VAO != 0) {
        glDeleteVertexArrays(1, &VAO);
        VAO = 0;
    }
    if (VBO != 0) {
        glDeleteBuffers(1, &VBO);
        VBO = 0;
    }
    
    std::vector<float> vertices;
    float half = size / 2.0f;
    float yOffset = size / 6.0f;
    if (shapeType == 0) { // Triángulo
        vertices = {
            0.0f,   size / 2.0f - yOffset, 0.0f,  colorTop[0], colorTop[1], colorTop[2],
           -half, -size / 2.0f - yOffset, 0.0f,  colorLeft[0], colorLeft[1], colorLeft[2],
            half, -size / 2.0f - yOffset, 0.0f,  colorRight[0], colorRight[1], colorRight[2]
        };
    } else if (shapeType == 1) { // Cuadrado
        // Usa los tres colores para 3 vértices, el cuarto es promedio
        float colorBottom[3] = {
            (colorLeft[0] + colorRight[0]) / 2.0f,
            (colorLeft[1] + colorRight[1]) / 2.0f,
            (colorLeft[2] + colorRight[2]) / 2.0f
        };
        vertices = {
            -half,  half, 0.0f, colorTop[0], colorTop[1], colorTop[2], // top-left
             half,  half, 0.0f, colorTop[0], colorTop[1], colorTop[2], // top-right
            -half, -half, 0.0f, colorBottom[0], colorBottom[1], colorBottom[2], // bottom-left
             half, -half, 0.0f, colorBottom[0], colorBottom[1], colorBottom[2]  // bottom-right
        };
    } else if (shapeType == 2) { // Círculo
        // Centro
        vertices.push_back(0.0f); vertices.push_back(0.0f); vertices.push_back(0.0f);
        vertices.push_back((colorTop[0] + colorLeft[0] + colorRight[0]) / 3.0f);
        vertices.push_back((colorTop[1] + colorLeft[1] + colorRight[1]) / 3.0f);
        vertices.push_back((colorTop[2] + colorLeft[2] + colorRight[2]) / 3.0f);
        for (int i = 0; i <= nSegments; ++i) {
            float theta = 2.0f * M_PI * float(i) / float(nSegments);
            float x = half * cos(theta);
            float y = half * sin(theta);
            float t = float(i) / float(nSegments);
            float r, g, b;
            if (t < 1.0f/3.0f) {
                float localT = t * 3.0f;
                r = (1-localT)*colorTop[0] + localT*colorLeft[0];
                g = (1-localT)*colorTop[1] + localT*colorLeft[1];
                b = (1-localT)*colorTop[2] + localT*colorLeft[2];
            } else if (t < 2.0f/3.0f) {
                float localT = (t-1.0f/3.0f)*3.0f;
                r = (1-localT)*colorLeft[0] + localT*colorRight[0];
                g = (1-localT)*colorLeft[1] + localT*colorRight[1];
                b = (1-localT)*colorLeft[2] + localT*colorRight[2];
            } else {
                float localT = (t-2.0f/3.0f)*3.0f;
                r = (1-localT)*colorRight[0] + localT*colorTop[0];
                g = (1-localT)*colorRight[1] + localT*colorTop[1];
                b = (1-localT)*colorRight[2] + localT*colorTop[2];
            }
            vertices.push_back(x);
            vertices.push_back(y);
            vertices.push_back(0.0f);
            vertices.push_back(r);
            vertices.push_back(g);
            vertices.push_back(b);
        }
    } else if (shapeType == 3) { // Línea
        // Línea simple horizontal
        vertices = {
            -half, 0.0f, 0.0f, colorLeft[0], colorLeft[1], colorLeft[2],
             half, 0.0f, 0.0f, colorRight[0], colorRight[1], colorRight[2]
        };
    } else if (shapeType == 4) { // Líneas largas
        // 6 líneas cruzando el origen en diferentes ángulos
        for (int i = 0; i < 6; ++i) {
            float angle = (float)i * M_PI / 6.0f;
            float x = half * cos(angle);
            float y = half * sin(angle);
            // Línea desde -x,-y a +x,+y
            vertices.push_back(-x); vertices.push_back(-y); vertices.push_back(0.0f);
            vertices.push_back(colorLeft[0]); vertices.push_back(colorLeft[1]); vertices.push_back(colorLeft[2]);
            vertices.push_back(x); vertices.push_back(y); vertices.push_back(0.0f);
            vertices.push_back(colorRight[0]); vertices.push_back(colorRight[1]); vertices.push_back(colorRight[2]);
        }
    }
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * vertices.size(), vertices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void createFractal(GLuint& VAO, GLuint& VBO, int baseShapeType, float size, float colorTop[3], float colorLeft[3], float colorRight[3], float depth, float time) {
    // Limpiar VAO y VBO existentes
    if (VAO != 0) {
        glDeleteVertexArrays(1, &VAO);
        VAO = 0;
    }
    if (VBO != 0) {
        glDeleteBuffers(1, &VBO);
        VBO = 0;
    }
    
    std::vector<float> vertices;
    float half = size / 2.0f;
    
    // Límites de seguridad para evitar demasiados vértices
    const int MAX_VERTICES = 10000; // Máximo 10k vértices
    const int MAX_DEPTH = 4; // Máximo 4 niveles de recursión
    
    int vertexCount = 0;
    
    // Función recursiva para generar el fractal usando std::function
    std::function<void(float, float, float, int, float)> addFractalShape;
    addFractalShape = [&](float x, float y, float s, int level, float angle) {
        // Límites de seguridad
        if (level <= 0 || level > MAX_DEPTH || vertexCount >= MAX_VERTICES) return;
        
        float scale = s * 0.5f;
        float animOffset = sin(time * 2.0f + level * 0.5f) * 0.1f;
        
        // Colores animados por nivel
        float r = colorTop[0] + sin(time + level) * 0.3f;
        float g = colorLeft[1] + cos(time + level * 0.7f) * 0.3f;
        float b = colorRight[2] + sin(time * 1.5f + level * 0.3f) * 0.3f;
        
        // Clamp colores
        r = std::max(0.0f, std::min(1.0f, r));
        g = std::max(0.0f, std::min(1.0f, g));
        b = std::max(0.0f, std::min(1.0f, b));
        
        if (baseShapeType == 0) { // Triángulo fractal
            if (vertexCount + 9 <= MAX_VERTICES) { // 3 vértices * 3 floats cada uno
                float x1 = x + (scale + animOffset) * cos(angle);
                float y1 = y + (scale + animOffset) * sin(angle);
                float x2 = x + (scale + animOffset) * cos(angle + 2.0944f);
                float y2 = y + (scale + animOffset) * sin(angle + 2.0944f);
                float x3 = x + (scale + animOffset) * cos(angle + 4.1888f);
                float y3 = y + (scale + animOffset) * sin(angle + 4.1888f);
                
                vertices.push_back(x1); vertices.push_back(y1); vertices.push_back(0.0f);
                vertices.push_back(r); vertices.push_back(g); vertices.push_back(b);
                vertices.push_back(x2); vertices.push_back(y2); vertices.push_back(0.0f);
                vertices.push_back(g); vertices.push_back(b); vertices.push_back(r);
                vertices.push_back(x3); vertices.push_back(y3); vertices.push_back(0.0f);
                vertices.push_back(b); vertices.push_back(r); vertices.push_back(g);
                
                vertexCount += 9;
                
                // Sub-triángulos solo si no hemos alcanzado el límite
                if (vertexCount < MAX_VERTICES) {
                    addFractalShape(x1, y1, scale, level - 1, angle + time * 0.5f);
                    addFractalShape(x2, y2, scale, level - 1, angle + time * 0.7f);
                    addFractalShape(x3, y3, scale, level - 1, angle + time * 0.3f);
                }
            }
            
        } else if (baseShapeType == 1) { // Cuadrado fractal
            if (vertexCount + 12 <= MAX_VERTICES) { // 4 vértices * 3 floats cada uno
                float x1 = x - scale; float y1 = y - scale;
                float x2 = x + scale; float y2 = y - scale;
                float x3 = x + scale; float y3 = y + scale;
                float x4 = x - scale; float y4 = y + scale;
                
                // Cuadrado principal
                vertices.push_back(x1); vertices.push_back(y1); vertices.push_back(0.0f);
                vertices.push_back(r); vertices.push_back(g); vertices.push_back(b);
                vertices.push_back(x2); vertices.push_back(y2); vertices.push_back(0.0f);
                vertices.push_back(g); vertices.push_back(b); vertices.push_back(r);
                vertices.push_back(x3); vertices.push_back(y3); vertices.push_back(0.0f);
                vertices.push_back(b); vertices.push_back(r); vertices.push_back(g);
                vertices.push_back(x4); vertices.push_back(y4); vertices.push_back(0.0f);
                vertices.push_back(r); vertices.push_back(g); vertices.push_back(b);
                
                vertexCount += 12;
                
                // Sub-cuadrados solo si no hemos alcanzado el límite
                if (vertexCount < MAX_VERTICES) {
                    addFractalShape(x1, y1, scale, level - 1, angle + time * 0.2f);
                    addFractalShape(x2, y2, scale, level - 1, angle + time * 0.4f);
                    addFractalShape(x3, y3, scale, level - 1, angle + time * 0.6f);
                    addFractalShape(x4, y4, scale, level - 1, angle + time * 0.8f);
                }
            }
            
        } else if (baseShapeType == 2) { // Círculo fractal
            int segments = 8;
            int verticesNeeded = segments * 9; // 3 vértices por segmento * 3 floats cada uno
            
            if (vertexCount + verticesNeeded <= MAX_VERTICES) {
                for (int i = 0; i < segments; ++i) {
                    float theta1 = 2.0f * M_PI * float(i) / float(segments);
                    float theta2 = 2.0f * M_PI * float(i + 1) / float(segments);
                    
                    float x1 = x + scale * cos(theta1);
                    float y1 = y + scale * sin(theta1);
                    float x2 = x + scale * cos(theta2);
                    float y2 = y + scale * sin(theta2);
                    
                    vertices.push_back(x); vertices.push_back(y); vertices.push_back(0.0f);
                    vertices.push_back(r); vertices.push_back(g); vertices.push_back(b);
                    vertices.push_back(x1); vertices.push_back(y1); vertices.push_back(0.0f);
                    vertices.push_back(g); vertices.push_back(b); vertices.push_back(r);
                    vertices.push_back(x2); vertices.push_back(y2); vertices.push_back(0.0f);
                    vertices.push_back(b); vertices.push_back(r); vertices.push_back(g);
                }
                
                vertexCount += verticesNeeded;
                
                // Sub-círculos solo si no hemos alcanzado el límite
                if (vertexCount < MAX_VERTICES) {
                    for (int i = 0; i < 4; ++i) {
                        float angle = i * M_PI / 2.0f + time * 0.3f;
                        float subX = x + scale * 0.7f * cos(angle);
                        float subY = y + scale * 0.7f * sin(angle);
                        addFractalShape(subX, subY, scale * 0.5f, level - 1, angle);
                    }
                }
            }
        } else if (baseShapeType == 3) { // Línea fractal
            if (vertexCount + 6 <= MAX_VERTICES) { // 2 vértices * 3 floats cada uno
                float x1 = x - scale * cos(angle);
                float y1 = y - scale * sin(angle);
                float x2 = x + scale * cos(angle);
                float y2 = y + scale * sin(angle);
                
                vertices.push_back(x1); vertices.push_back(y1); vertices.push_back(0.0f);
                vertices.push_back(r); vertices.push_back(g); vertices.push_back(b);
                vertices.push_back(x2); vertices.push_back(y2); vertices.push_back(0.0f);
                vertices.push_back(g); vertices.push_back(b); vertices.push_back(r);
                
                vertexCount += 6;
                
                // Sub-líneas solo si no hemos alcanzado el límite
                if (vertexCount < MAX_VERTICES) {
                    for (int i = 0; i < 3; ++i) {
                        float subAngle = angle + i * M_PI / 3.0f + time * 0.2f;
                        float subX = x + scale * 0.6f * cos(subAngle);
                        float subY = y + scale * 0.6f * sin(subAngle);
                        addFractalShape(subX, subY, scale * 0.4f, level - 1, subAngle);
                    }
                }
            }
        } else if (baseShapeType == 4) { // Líneas largas fractal
            if (vertexCount + 36 <= MAX_VERTICES) { // 6 líneas * 2 vértices * 3 floats cada uno
                // 6 líneas cruzando el centro en diferentes ángulos
                for (int i = 0; i < 6; ++i) {
                    float lineAngle = angle + i * M_PI / 3.0f + time * 0.1f;
                    float x1 = x - scale * 1.5f * cos(lineAngle);
                    float y1 = y - scale * 1.5f * sin(lineAngle);
                    float x2 = x + scale * 1.5f * cos(lineAngle);
                    float y2 = y + scale * 1.5f * sin(lineAngle);
                    
                    vertices.push_back(x1); vertices.push_back(y1); vertices.push_back(0.0f);
                    vertices.push_back(r); vertices.push_back(g); vertices.push_back(b);
                    vertices.push_back(x2); vertices.push_back(y2); vertices.push_back(0.0f);
                    vertices.push_back(g); vertices.push_back(b); vertices.push_back(r);
                }
                
                vertexCount += 36;
                
                // Sub-líneas largas solo si no hemos alcanzado el límite
                if (vertexCount < MAX_VERTICES) {
                    for (int i = 0; i < 4; ++i) {
                        float subAngle = angle + i * M_PI / 2.0f + time * 0.3f;
                        float subX = x + scale * 0.8f * cos(subAngle);
                        float subY = y + scale * 0.8f * sin(subAngle);
                        addFractalShape(subX, subY, scale * 0.5f, level - 1, subAngle);
                    }
                }
            }
        }
    };
    
    // Iniciar el fractal desde el centro con profundidad limitada
    int actualDepth = std::min((int)depth, MAX_DEPTH);
    addFractalShape(0.0f, 0.0f, half, actualDepth, time);
    
    // Solo crear VAO/VBO si tenemos vértices
    if (!vertices.empty()) {
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(float) * vertices.size(), vertices.data(), GL_STATIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(1);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
    }
} 