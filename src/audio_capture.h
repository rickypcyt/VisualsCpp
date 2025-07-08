#pragma once
#include <vector>
#include <pulse/simple.h>
#include "utils/ring_buffer.h"
#include <thread>
#include <atomic>

class AudioCapture {
public:
    AudioCapture(const char* device, int sample_rate, int channels, int block_size = 512);
    ~AudioCapture();
    void start();
    void stop();
    // Get the latest block of samples, returns false if not enough data
    bool getLatestBlock(std::vector<int32_t>& out);
    int getSampleRate() const { return sample_rate; }
    int getChannels() const { return channels; }
private:
    void captureThreadFunc();
    pa_simple* s;
    int sample_rate;
    int channels;
    int block_size;
    std::thread capture_thread;
    std::atomic<bool> running;
    RingBuffer<int32_t, 16384> ring_buffer; // 16K samples buffer
}; 