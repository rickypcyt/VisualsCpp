#include "audio_capture.h"
#include <pulse/error.h>
#include <iostream>

AudioCapture::AudioCapture(const char* device, int sample_rate, int channels)
    : s(nullptr), sample_rate(sample_rate), channels(channels) {
    pa_sample_spec ss;
    ss.format = PA_SAMPLE_S32LE;
    ss.rate = sample_rate;
    ss.channels = channels;

    int error;
    s = pa_simple_new(
        NULL, "VisualsCpp", PA_STREAM_RECORD,
        device, "record", &ss, NULL, NULL, &error
    );
    if (!s) {
        std::cerr << "pa_simple_new() failed: " << pa_strerror(error) << std::endl;
    }
}

AudioCapture::~AudioCapture() {
    if (s) pa_simple_free(s);
}

bool AudioCapture::read(std::vector<int32_t>& buffer) {
    if (!s) return false;
    int error;
    if (pa_simple_read(s, buffer.data(), buffer.size() * sizeof(int32_t), &error) < 0) {
        std::cerr << "pa_simple_read() failed: " << pa_strerror(error) << std::endl;
        return false;
    }
    return true;
} 