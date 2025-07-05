#include "fft_utils.h"
#include <cmath>
#include <cstring>

FFTUtils::FFTUtils(int fft_size)
    : fft_size(fft_size) {
    cfg = kiss_fft_alloc(fft_size, 0, nullptr, nullptr);
}

FFTUtils::~FFTUtils() {
    if (cfg) free(cfg);
}

std::vector<float> FFTUtils::compute(const std::vector<float>& input) {
    std::vector<kiss_fft_cpx> in(fft_size), out(fft_size);
    for (size_t i = 0; i < static_cast<size_t>(fft_size); ++i) {
        in[i].r = (i < input.size()) ? input[i] : 0.0f;
        in[i].i = 0.0f;
    }
    kiss_fft(cfg, in.data(), out.data());

    std::vector<float> magnitudes(fft_size / 2);
    for (size_t i = 0; i < static_cast<size_t>(fft_size / 2); ++i) {
        magnitudes[i] = std::sqrt(out[i].r * out[i].r + out[i].i * out[i].i);
    }
    return magnitudes;
} 