CXX = g++
CXXFLAGS = -I/usr/include -I./src -I./imgui -I./imgui/backends -Wall -std=c++17
LDFLAGS = -lGLEW -lglfw -ldl -lGL -lX11 -lpthread -lXrandr -lXi -lpulse-simple -lpulse
SRC = main.cpp src/window_utils.cpp src/shader_utils.cpp src/triangle_utils.cpp \
      src/audio_capture.cpp src/fft_utils.cpp \
      audio_capture.cpp waveform.cpp \
      imgui/imgui.cpp imgui/imgui_draw.cpp imgui/imgui_tables.cpp imgui/imgui_widgets.cpp \
      imgui/backends/imgui_impl_glfw.cpp imgui/backends/imgui_impl_opengl3.cpp \
      kissfft/kiss_fft.c
TARGET = triangle

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET) 