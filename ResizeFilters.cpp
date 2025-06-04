#include "ResizeFilters.h"

#include <stdint.h>

/// <summary>
/// Upscale only filtering of frame functions (convert a 128x32 to a 256x64 frame)
/// The buffers have been created before calling these functions
/// </summary>
/// <param name="inframe">inbound 128x32 frame</param>
/// <param name="outframe">outbound 256x64 frame</param>
/// 
void Filter_Scale2X(UINT16* inframe, UINT16* outframe, UINT32 width, UINT32 height)
{
    for (UINT32 tj = 0; tj < height; tj++)
    {
        for (UINT32 ti = 0; ti < width; ti++)
        {
            UINT16 A, B, C, D, E, F, G, H, I;
            if (ti == 0 && tj == 0)
            {
                A = B = D = E = inframe[0];
                C = F = inframe[1];
                G = H = inframe[width];
                I = inframe[width + 1];
            }
            else if (ti == 0 && tj == (height - 1))
            {
                D = E = G = H = inframe[(height - 1) * width];
                A = B = inframe[(height - 2) * width];
                I = F = inframe[(height - 1) * width + 1];
                C = inframe[(height - 2) * width + 1];
            }
            else if (ti == (width - 1) && tj == 0)
            {
                B = C = E = F = inframe[width - 1];
                A = D = inframe[width - 2];
                H = I = inframe[2 * width - 1];
                G = inframe[2 * width - 2];
            }
            else if (ti == (width - 1) && tj == (height - 1))
            {
                E = F = H = I = inframe[height * width - 1];
                G = D = inframe[height * width - 2];
                B = C = inframe[(height - 1) * width - 1];
                A = inframe[(height - 1) * width - 2];
            }
            else if (ti == 0)
            {
                A = B = inframe[(tj - 1) * width];
                D = E = inframe[tj * width];
                G = H = inframe[(tj + 1) * width];
                C = inframe[(tj - 1) * width + 1];
                F = inframe[tj * width + 1];
                I = inframe[(tj + 1) * width + 1];
            }
            else if (tj == 0)
            {
                A = D = inframe[ti - 1];
                B = E = inframe[ti];
                C = F = inframe[ti + 1];
                G = inframe[width + ti - 1];
                H = inframe[width + ti];
                I = inframe[width + ti + 1];
            }
            else if (ti == (width - 1))
            {
                B = C = inframe[tj * width - 1];
                E = F = inframe[(tj + 1) * width - 1];
                H = I = inframe[(tj + 2) * width - 1];
                A = inframe[tj * width - 2];
                D = inframe[(tj + 1) * width - 2];
                G = inframe[(tj + 2) * width - 2];
            }
            else if (tj == (height - 1))
            {
                D = G = inframe[(height - 1) * width + ti - 1];
                E = H = inframe[(height - 1) * width + ti];
                F = I = inframe[(height - 1) * width + ti + 1];
                A = inframe[(height - 2) * width + ti - 1];
                B = inframe[(height - 2) * width + ti];
                C = inframe[(height - 2) * width + ti + 1];
            }
            else
            {
                A = inframe[(tj - 1) * width + ti - 1];
                B = inframe[(tj - 1) * width + ti];
                C = inframe[(tj - 1) * width + ti + 1];
                D = inframe[tj * width + ti - 1];
                E = inframe[tj * width + ti];
                F = inframe[tj * width + ti + 1];
                G = inframe[(tj + 1) * width + ti - 1];
                H = inframe[(tj + 1) * width + ti];
                I = inframe[(tj + 1) * width + ti + 1];
            }
            if (B != H && D != F)
            {
                outframe[tj * 2 * (2 * width) + ti * 2] = D == B ? D : E;
                outframe[tj * 2 * (2 * width) + ti * 2 + 1] = B == F ? F : E;
                outframe[(tj * 2 + 1) * (2 * width) + ti * 2] = D == H ? D : E;
                outframe[(tj * 2 + 1) * (2 * width) + ti * 2 + 1] = H == F ? F : E;
            }
            else
            {
                outframe[tj * 2 * (2 * width) + ti * 2] = E;
                outframe[tj * 2 * (2 * width) + ti * 2 + 1] = E;
                outframe[(tj * 2 + 1) * (2 * width) + ti * 2] = E;
                outframe[(tj * 2 + 1) * (2 * width) + ti * 2 + 1] = E;
            }
        }
    }
}

/// <summary>
/// Upscale only filtering of frame functions (convert a 128x32 to a 256x64 frame)
/// The buffers have been created before calling these functions
/// </summary>
/// <param name="inframe">inbound 128x32 frame</param>
/// <param name="outframe">outbound 256x64 frame</param>
/// 
void Filter_Scale2Xmask(UINT16* inframe, UINT16* outframe, UINT8* masks, UINT8* maskd, UINT32 width, UINT32 height)
{
    for (UINT32 tj = 0; tj < height / 2; tj++)
    {
        for (UINT32 ti = 0; ti < width / 2; ti++)
        {
            UINT16 A, B, C, D, E, F, G, H, I;
            UINT8 Am, Bm, Cm, Dm, Em, Fm, Gm, Hm, Im;
            if (ti == 0 && tj == 0)
            {
                A = B = D = E = inframe[0];
                Am = Bm = Dm = Em = masks[0];
                C = F = inframe[1];
                Cm = Fm = masks[1];
                G = H = inframe[width];
                Gm = Hm = masks[width];
                I = inframe[width + 1];
                Im = masks[width + 1];
            }
            else if (ti == 0 && tj == (height - 1))
            {
                D = E = G = H = inframe[(height - 1) * width];
                Dm = Em = Gm = Hm = masks[(height - 1) * width];
                A = B = inframe[(height - 2) * width];
                Am = Bm = masks[(height - 2) * width];
                I = F = inframe[(height - 1) * width + 1];
                Im = Fm = masks[(height - 1) * width + 1];
                C = inframe[(height - 2) * width + 1];
                Cm = masks[(height - 2) * width + 1];
            }
            else if (ti == (width - 1) && tj == 0)
            {
                B = C = E = F = inframe[width - 1];
                Bm = Cm = Em = Fm = masks[width - 1];
                A = D = inframe[width - 2];
                Am = Dm = masks[width - 2];
                H = I = inframe[2 * width - 1];
                Hm = Im = masks[2 * width - 1];
                G = inframe[2 * width - 2];
                Gm = masks[2 * width - 2];
            }
            else if (ti == (width - 1) && tj == (height - 1))
            {
                E = F = H = I = inframe[height * width - 1];
                Em = Fm = Hm = Im = masks[height * width - 1];
                G = D = inframe[height * width - 2];
                Gm = Dm = masks[height * width - 2];
                B = C = inframe[(height - 1) * width - 1];
                Bm = Cm = masks[(height - 1) * width - 1];
                A = inframe[(height - 1) * width - 2];
                Am = masks[(height - 1) * width - 2];
            }
            else if (ti == 0)
            {
                A = B = inframe[(tj - 1) * width];
                Am = Bm = masks[(tj - 1) * width];
                D = E = inframe[tj * width];
                Dm = Em = masks[tj * width];
                G = H = inframe[(tj + 1) * width];
                Gm = Hm = masks[(tj + 1) * width];
                C = inframe[(tj - 1) * width + 1];
                Cm = masks[(tj - 1) * width + 1];
                F = inframe[tj * width + 1];
                Fm = masks[tj * width + 1];
                I = inframe[(tj + 1) * width + 1];
                Im = masks[(tj + 1) * width + 1];
            }
            else if (tj == 0)
            {
                A = D = inframe[ti - 1];
                Am = Dm = masks[ti - 1];
                B = E = inframe[ti];
                Bm = Em = masks[ti];
                C = F = inframe[ti + 1];
                Cm = Fm = masks[ti + 1];
                G = inframe[width + ti - 1];
                Gm = masks[width + ti - 1];
                H = inframe[width + ti];
                Hm = masks[width + ti];
                I = inframe[width + ti + 1];
                Im = masks[width + ti + 1];
            }
            else if (ti == (width - 1))
            {
                B = C = inframe[tj * width - 1];
                Bm = Cm = masks[tj * width - 1];
                E = F = inframe[(tj + 1) * width - 1];
                Em = Fm = masks[(tj + 1) * width - 1];
                H = I = inframe[(tj + 2) * width - 1];
                Hm = Im = masks[(tj + 2) * width - 1];
                A = inframe[tj * width - 2];
                Am = masks[tj * width - 2];
                D = inframe[(tj + 1) * width - 2];
                Dm = masks[(tj + 1) * width - 2];
                G = inframe[(tj + 2) * width - 2];
                Gm = masks[(tj + 2) * width - 2];
            }
            else if (tj == (height - 1))
            {
                D = G = inframe[(height - 1) * width + ti - 1];
                Dm = Gm = masks[(height - 1) * width + ti - 1];
                E = H = inframe[(height - 1) * width + ti];
                Em = Hm = masks[(height - 1) * width + ti];
                F = I = inframe[(height - 1) * width + ti + 1];
                Fm = Im = masks[(height - 1) * width + ti + 1];
                A = inframe[(height - 2) * width + ti - 1];
                Am = masks[(height - 2) * width + ti - 1];
                B = inframe[(height - 2) * width + ti];
                Bm = masks[(height - 2) * width + ti];
                C = inframe[(height - 2) * width + ti + 1];
                Cm = masks[(height - 2) * width + ti + 1];
            }
            else
            {
                A = inframe[(tj - 1) * width + ti - 1];
                Am = masks[(tj - 1) * width + ti - 1];
                B = inframe[(tj - 1) * width + ti];
                Bm = masks[(tj - 1) * width + ti];
                C = inframe[(tj - 1) * width + ti + 1];
                Cm = masks[(tj - 1) * width + ti + 1];
                D = inframe[tj * width + ti - 1];
                Dm = masks[tj * width + ti - 1];
                E = inframe[tj * width + ti];
                Em = masks[tj * width + ti];
                F = inframe[tj * width + ti + 1];
                Fm = masks[tj * width + ti + 1];
                G = inframe[(tj + 1) * width + ti - 1];
                Gm = masks[(tj + 1) * width + ti - 1];
                H = inframe[(tj + 1) * width + ti];
                Hm = masks[(tj + 1) * width + ti];
                I = inframe[(tj + 1) * width + ti + 1];
                Im = masks[(tj + 1) * width + ti + 1];
            }
            // !!!!! for sprites, the resolutions in SD and HD are the same, so no (2 * width) for a line
            if (B != H && D != F)
            {
                outframe[tj * 2 * width + ti * 2] = D == B ? D : E;
                maskd[tj * 2 * width + ti * 2] = Dm == Bm ? Dm : Em;
                outframe[tj * 2 * width + ti * 2 + 1] = B == F ? F : E;
                maskd[tj * 2 * width + ti * 2 + 1] = Bm == Fm ? Fm : Em;
                outframe[(tj * 2 + 1) * width + ti * 2] = D == H ? D : E;
                maskd[(tj * 2 + 1) * width + ti * 2] = Dm == Hm ? Dm : Em;
                outframe[(tj * 2 + 1) * width + ti * 2 + 1] = H == F ? F : E;
                maskd[(tj * 2 + 1) * width + ti * 2 + 1] = Hm == Fm ? Fm : Em;
            }
            else
            {
                outframe[tj * 2 * width + ti * 2] = E;
                maskd[tj * 2 * width + ti * 2] = Em;
                outframe[tj * 2 * width + ti * 2 + 1] = E;
                maskd[tj * 2 * width + ti * 2 + 1] = Em;
                outframe[(tj * 2 + 1) * width + ti * 2] = E;
                maskd[(tj * 2 + 1) * width + ti * 2] = Em;
                outframe[(tj * 2 + 1) * width + ti * 2 + 1] = E;
                maskd[(tj * 2 + 1) * width + ti * 2 + 1] = Em;
            }
        }
    }
}
