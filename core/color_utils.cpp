#include "color_utils.h"

void rgb565_to_rgb888(uint16_t rgb565, uint8_t* r, uint8_t* g, uint8_t* b)
{
    *r = ((rgb565 >> 8) & 0xF8) | ((rgb565 >> 13) & 0x07);
    *g = ((rgb565 >> 3) & 0xFC) | ((rgb565 >> 9) & 0x03);
    *b = ((rgb565 << 3) & 0xF8) | ((rgb565 >> 2) & 0x07);
}

void rgb565_to_rgb888(uint16_t rgb565, uint8_t* rgb888)
{
    rgb888[0] = ((rgb565 >> 8) & 0xF8) | ((rgb565 >> 13) & 0x07);
    rgb888[1] = ((rgb565 >> 3) & 0xFC) | ((rgb565 >> 9) & 0x03);
    rgb888[2] = ((rgb565 << 3) & 0xF8) | ((rgb565 >> 2) & 0x07);
}

uint16_t rgb888_to_rgb565(uint8_t r, uint8_t g, uint8_t b)
{
    uint8_t r5 = r >> 3;
    uint8_t g6 = g >> 2;
    uint8_t b5 = b >> 3;

    return static_cast<uint16_t>((r5 << 11) | (g6 << 5) | b5);
}
