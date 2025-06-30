CXX = g++
CXXFLAGS = -I/usr/include -I./src -Wall -std=c++17
LDFLAGS = -lGLEW -lglfw -ldl -lGL -lX11 -lpthread -lXrandr -lXi
SRC = main.cpp src/window_utils.cpp src/shader_utils.cpp src/triangle_utils.cpp
TARGET = triangle

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET) 