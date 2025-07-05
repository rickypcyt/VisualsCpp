#include "waveform.h"
#include <algorithm>

WaveformBuffer::WaveformBuffer(size_t size) : buffer(size, 0.0f), head(0) {}

void WaveformBuffer::push_samples(const float* data, size_t count) {
    std::lock_guard<std::mutex> lock(mtx);
    for (size_t i = 0; i < count; ++i) {
        buffer[head] = data[i];
        head = (head + 1) % buffer.size();
    }
}

std::vector<float> WaveformBuffer::get_samples() {
    std::lock_guard<std::mutex> lock(mtx);
    std::vector<float> out(buffer.size());
    size_t idx = head;
    for (size_t i = 0; i < buffer.size(); ++i) {
        out[i] = buffer[idx];
        idx = (idx + 1) % buffer.size();
    }
    return out;
} 