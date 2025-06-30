#include "triangle_utils.h"
#include <vector>
#include <cmath>

void createShape(GLuint& VAO, GLuint& VBO, int shapeType, float size, float colorTop[3], float colorLeft[3], float colorRight[3], int nSegments) {
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