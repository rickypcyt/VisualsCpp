CXX = g++
CXXFLAGS = -I/usr/include -Wall -std=c++17
LDFLAGS = -lGLEW -lglfw -ldl -lGL -lX11 -lpthread -lXrandr -lXi
SRC = main.cpp
TARGET = triangle

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET) 