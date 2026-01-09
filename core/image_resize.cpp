#include "image_resize.h"

#include <algorithm>
#include <opencv2/opencv.hpp>

#include "color_utils.h"
#include "serum_constants.h"

void ResizeRGB565Image(uint16_t* pdImage, unsigned int dwidth, unsigned int dheight,
                       uint16_t* psImage, unsigned int swidth, unsigned int sheight, int filter)
{
    cv::Mat imageMat(sheight, swidth, CV_8UC3);
    uint8_t rgb888[3];

    for (unsigned int y = 0; y < sheight; ++y)
    {
        for (unsigned int x = 0; x < swidth; ++x)
        {
            rgb565_to_rgb888(psImage[y * swidth + x], rgb888);
            imageMat.at<cv::Vec3b>(y, x) = cv::Vec3b(rgb888[2], rgb888[1], rgb888[0]);
        }
    }

    cv::Mat destmat;
    cv::resize(imageMat, destmat, cv::Size(dwidth, dheight), 0, 0, filter);

    for (unsigned int y = 0; y < dheight; ++y)
    {
        for (unsigned int x = 0; x < dwidth; ++x)
        {
            cv::Vec3b pixel = destmat.at<cv::Vec3b>(y, x);
            pdImage[y * dwidth + x] = rgb888_to_rgb565(pixel[2], pixel[1], pixel[0]);
        }
    }
}

void ResizeRGB565Sprite(uint16_t* pdSprite, uint8_t* pdSprMask,
                        uint16_t* psSprite, uint8_t* psSprMask,
                        bool shrink, int filter)
{
    int ssprw = 0, ssprh = 0;
    for (int tj = 0; tj < MAX_SPRITE_HEIGHT; tj++)
    {
        for (int ti = 0; ti < MAX_SPRITE_WIDTH; ti++)
        {
            if (psSprMask[tj * MAX_SPRITE_WIDTH + ti] < 255)
            {
                if (tj > ssprh) ssprh = tj;
                if (ti > ssprw) ssprw = ti;
            }
        }
    }
    ssprh++;
    ssprw++;

    cv::Mat imageMat(ssprh, ssprw, CV_8UC3);
    uint8_t rgb888[3];

    for (int y = 0; y < ssprh; ++y)
    {
        for (int x = 0; x < ssprw; ++x)
        {
            rgb565_to_rgb888(psSprite[y * MAX_SPRITE_WIDTH + x], rgb888);
            imageMat.at<cv::Vec3b>(y, x) = cv::Vec3b(rgb888[2], rgb888[1], rgb888[0]);
        }
    }

    int dsprw, dsprh;
    if (shrink)
    {
        dsprw = ssprw / 2;
        dsprh = ssprh / 2;
    }
    else
    {
        dsprw = std::min(ssprw * 2, MAX_SPRITE_WIDTH);
        dsprh = std::min(ssprh * 2, MAX_SPRITE_HEIGHT);
    }

    cv::Mat destmat;
    cv::resize(imageMat, destmat, cv::Size(dsprw, dsprh), 0, 0, filter);

    for (int y = 0; y < dsprh; ++y)
    {
        for (int x = 0; x < dsprw; ++x)
        {
            cv::Vec3b pixel = destmat.at<cv::Vec3b>(y, x);
            pdSprite[y * MAX_SPRITE_WIDTH + x] = rgb888_to_rgb565(pixel[2], pixel[1], pixel[0]);
            uint8_t* finmsk = &pdSprMask[y * MAX_SPRITE_WIDTH + x];
            if (shrink)
            {
                uint8_t* tmsk = &psSprMask[y * 2 * MAX_SPRITE_WIDTH + x * 2];
                if (tmsk[0] == tmsk[1] || tmsk[0] == tmsk[MAX_SPRITE_WIDTH] || tmsk[0] == tmsk[MAX_SPRITE_WIDTH + 1]) *finmsk = tmsk[0];
                else if (tmsk[1] == tmsk[MAX_SPRITE_WIDTH] || tmsk[1] == tmsk[MAX_SPRITE_WIDTH + 1]) *finmsk = tmsk[1];
                else if (tmsk[MAX_SPRITE_WIDTH] == tmsk[MAX_SPRITE_WIDTH + 1]) *finmsk = tmsk[MAX_SPRITE_WIDTH];
                else *finmsk = tmsk[0];
            }
            else *finmsk = psSprMask[y / 2 * MAX_SPRITE_WIDTH + x / 2];
        }
    }
}
