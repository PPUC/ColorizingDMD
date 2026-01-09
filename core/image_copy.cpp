#include "image_copy.h"

#include "color_utils.h"

void ApplyImageToFrames(const cv::Mat& tmp32, const cv::Mat& tmp64, const ImageCopyParams& params)
{
    for (unsigned int tj = 0; tj < params.sel_h; tj++)
    {
        for (unsigned int tk = 0; tk < params.sel_w; tk++)
        {
            if (params.copy_to_64 && params.frame64)
            {
                bool allow_copy = true;
                if (params.use_mask && params.mask64)
                {
                    unsigned int mask_index = (params.sel_y + tj) * params.mask_stride64 + params.sel_x + tk;
                    allow_copy = params.mask64[mask_index] != 0;
                }
                if (allow_copy)
                {
                    cv::Vec3b color = tmp64.at<cv::Vec3b>(tj, tk);
                    params.frame64[(params.sel_y + tj) * params.width64 + (params.sel_x + tk)] =
                        rgb888_to_rgb565(color[2], color[1], color[0]);
                }
            }
            if (params.copy_to_32 && params.frame32)
            {
                bool allow_copy = true;
                if (params.use_mask && params.mask32)
                {
                    unsigned int mask_index = (params.sel_y + tj) * params.mask_stride32 + params.sel_x + tk;
                    allow_copy = params.mask32[mask_index] != 0;
                }
                if (allow_copy)
                {
                    cv::Vec3b color = tmp32.at<cv::Vec3b>(tj / 2, tk / 2);
                    params.frame32[(params.sel_y + tj) / 2 * params.width32 + (params.sel_x + tk) / 2] =
                        rgb888_to_rgb565(color[2], color[1], color[0]);
                }
            }
        }
    }
}
