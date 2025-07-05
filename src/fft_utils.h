#pragma once
#include <vector>
#include "../kissfft/kiss_fft.h"

class FFTUtils {
public:
    FFTUtils(int fft_size);
    ~FFTUtils();
    std::vector<float> compute(const std::vector<float>& input);
private:
    int fft_size;
    kiss_fft_cfg cfg;
}; 