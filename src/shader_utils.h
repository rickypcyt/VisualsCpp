#pragma once
#include <GL/glew.h>

GLuint createShader(GLenum type, const char* source);
GLuint createShaderProgram(const char* vertexSrc, const char* fragmentSrc); 