#pragma once
#include "waveform.h"
#include <atomic>
#include <string>
#include <vector>

// Devuelve un vector de pares (nombre_monitor, descripcion)
std::vector<std::pair<std::string, std::string>> get_monitor_sources();

// Captura audio desde el monitor especificado
void capture_audio_to_waveform(WaveformBuffer& buffer, std::atomic<bool>& running, const std::string& monitor_name); 