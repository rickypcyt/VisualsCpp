#pragma once
#include <GL/glew.h>

void createShape(GLuint& VAO, GLuint& VBO, int shapeType, float size, float colorTop[3], float colorLeft[3], float colorRight[3], int nSegments = 32);
void createFractal(GLuint& VAO, GLuint& VBO, int baseShapeType, float size, float colorTop[3], float colorLeft[3], float colorRight[3], float depth, float time); 