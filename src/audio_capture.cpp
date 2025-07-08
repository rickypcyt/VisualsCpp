#include "audio_capture.h"
#include <pulse/simple.h>
#include <pulse/error.h>
#include <pulse/pulseaudio.h>
#include <iostream>
#include <cstring>
#include <vector>
#include <thread>
#include <atomic>
#include "utils/ring_buffer.h"
#include <chrono>

AudioCapture::AudioCapture(const char* device, int sample_rate, int channels, int block_size)
    : s(nullptr), sample_rate(sample_rate), channels(channels), block_size(block_size), running(false) {
    pa_sample_spec ss;
    ss.format = PA_SAMPLE_S32LE;
    ss.rate = sample_rate;
    ss.channels = channels;

    pa_buffer_attr attr;
    attr.maxlength = block_size * channels * sizeof(int32_t) * 4; // 4x block for safety
    attr.tlength = block_size * channels * sizeof(int32_t);
    attr.prebuf = 0;
    attr.minreq = block_size * channels * sizeof(int32_t);
    attr.fragsize = block_size * channels * sizeof(int32_t);

    int error;
    s = pa_simple_new(
        NULL, "VisualsCpp", PA_STREAM_RECORD,
        device, "record", &ss, NULL, &attr, &error
    );
    if (!s) {
        std::cerr << "pa_simple_new() failed: " << pa_strerror(error) << std::endl;
    }
}

AudioCapture::~AudioCapture() {
    stop();
    if (s) pa_simple_free(s);
}

void AudioCapture::start() {
    if (running) return;
    running = true;
    capture_thread = std::thread(&AudioCapture::captureThreadFunc, this);
}

void AudioCapture::stop() {
    running = false;
    if (capture_thread.joinable()) capture_thread.join();
}

void AudioCapture::captureThreadFunc() {
    std::vector<int32_t> block(block_size * channels);
    while (running) {
        if (!s) break;
        int error;
        if (pa_simple_read(s, block.data(), block.size() * sizeof(int32_t), &error) < 0) {
            std::cerr << "pa_simple_read() failed: " << pa_strerror(error) << std::endl;
            break;
        }
        // Push samples into ring buffer
        for (size_t i = 0; i < block.size(); ++i) {
            while (!ring_buffer.push(block[i]) && running) {
                std::this_thread::sleep_for(std::chrono::microseconds(100)); // Wait for space
            }
        }
    }
}

// Get the latest block of samples, returns false if not enough data
bool AudioCapture::getLatestBlock(std::vector<int32_t>& out) {
    if (out.size() != block_size * channels) out.resize(block_size * channels);
    for (int i = 0; i < block_size * channels; ++i) {
        if (!ring_buffer.pop(out[i])) return false;
    }
    return true;
} 