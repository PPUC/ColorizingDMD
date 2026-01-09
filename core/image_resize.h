#pragma once

#include <cstdint>

void ResizeRGB565Image(uint16_t* pdImage, unsigned int dwidth, unsigned int dheight,
                       uint16_t* psImage, unsigned int swidth, unsigned int sheight, int filter);
void ResizeRGB565Sprite(uint16_t* pdSprite, uint8_t* pdSprMask,
                        uint16_t* psSprite, uint8_t* psSprMask,
                        bool shrink, int filter);
