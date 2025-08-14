#include "window_utils.h"

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

GLFWwindow* createFullscreenWindow(int& width, int& height) {
    width = 1920;
    height = 1080;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4); // Antialiasing 4x (mejor rendimiento)
    return glfwCreateWindow(width, height, "OpenGL Multicolor Triangle", glfwGetPrimaryMonitor(), nullptr);
} 