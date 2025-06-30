#include "triangle_utils.h"

void createTriangle(GLuint& VAO, GLuint& VBO, float width, float height) {
    float halfWidth = width / 2.0f;
    float yOffset = height / 6.0f;
    float vertices[] = {
        // posiciones                              // colores
         0.0f,   height / 2.0f - yOffset, 0.0f,    1.0f, 0.0f, 0.0f, // top (rojo)
        -halfWidth, -height / 2.0f - yOffset, 0.0f, 0.0f, 1.0f, 0.0f, // left (verde)
         halfWidth, -height / 2.0f - yOffset, 0.0f, 0.0f, 0.0f, 1.0f  // right (azul)
    };
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    // Posición
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // Color
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
} 