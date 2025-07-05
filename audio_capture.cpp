#include "audio_capture.h"
#include <pulse/simple.h>
#include <pulse/error.h>
#include <pulse/pulseaudio.h>
#include <cstring>
#include <vector>
#include <iostream>

// Listar monitores disponibles
std::vector<std::pair<std::string, std::string>> get_monitor_sources() {
    std::vector<std::pair<std::string, std::string>> result;
    pa_mainloop* m = pa_mainloop_new();
    pa_context* c = pa_context_new(pa_mainloop_get_api(m), "MusicVisualizer");
    pa_context_connect(c, nullptr, PA_CONTEXT_NOFLAGS, nullptr);
    while (pa_context_get_state(c) != PA_CONTEXT_READY) pa_mainloop_iterate(m, 1, nullptr);
    struct Data { std::vector<std::pair<std::string, std::string>>* out; } data = { &result };
    pa_operation* o = pa_context_get_source_info_list(c, [](pa_context*, const pa_source_info* i, int eol, void* userdata) {
        if (eol || !i) return;
        if (strstr(i->name, ".monitor")) {
            ((Data*)userdata)->out->emplace_back(i->name, i->description ? i->description : i->name);
        }
    }, &data);
    while (pa_operation_get_state(o) == PA_OPERATION_RUNNING) pa_mainloop_iterate(m, 1, nullptr);
    pa_operation_unref(o);
    pa_context_disconnect(c);
    pa_context_unref(c);
    pa_mainloop_free(m);
    return result;
}

// Captura desde monitor específico
void capture_audio_to_waveform(WaveformBuffer& buffer, std::atomic<bool>& running, const std::string& monitor_name) {
    pa_sample_spec ss;
    ss.format = PA_SAMPLE_FLOAT32LE;
    ss.rate = 48000;
    ss.channels = 1;
    int error;
    pa_simple* s = pa_simple_new(nullptr, "MusicVisualizer", PA_STREAM_RECORD, monitor_name.c_str(), "record", &ss, nullptr, nullptr, &error);
    if (!s) {
        std::cerr << "PulseAudio error: " << pa_strerror(error)
                  << " (monitor: " << monitor_name << ")" << std::endl;
        return;
    }
    std::vector<float> buf(256);
    while (running) {
        if (pa_simple_read(s, buf.data(), buf.size() * sizeof(float), &error) < 0) {
            std::cerr << "PulseAudio read error: " << pa_strerror(error)
                      << " (monitor: " << monitor_name << ")" << std::endl;
            break;
        }
        buffer.push_samples(buf.data(), buf.size());
    }
    pa_simple_free(s);
} 