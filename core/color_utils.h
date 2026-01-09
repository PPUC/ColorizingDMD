#pragma once

#include <cstdint>

uint16_t rgb888_to_rgb565(uint8_t r, uint8_t g, uint8_t b);
void rgb565_to_rgb888(uint16_t rgb565, uint8_t* r, uint8_t* g, uint8_t* b);
void rgb565_to_rgb888(uint16_t rgb565, uint8_t* rgb888);
