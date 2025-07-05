#pragma once
#include <vector>
#include <mutex>

class WaveformBuffer {
public:
    WaveformBuffer(size_t size);
    void push_samples(const float* data, size_t count);
    std::vector<float> get_samples();
private:
    std::vector<float> buffer;
    size_t head;
    std::mutex mtx;
}; 