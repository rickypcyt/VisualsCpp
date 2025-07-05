#pragma once
#include <vector>
#include <pulse/simple.h>

class AudioCapture {
public:
    AudioCapture(const char* device, int sample_rate, int channels);
    ~AudioCapture();
    bool read(std::vector<int32_t>& buffer);
    int getSampleRate() const { return sample_rate; }
    int getChannels() const { return channels; }
private:
    pa_simple* s;
    int sample_rate;
    int channels;
}; 