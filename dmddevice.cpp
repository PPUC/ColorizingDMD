#include "dmddevice.h"

HMODULE DmdDev_hModule;

DmdDev_Open_t DmdDev_Open;
DmdDev_Close_t DmdDev_Close;
DmdDev_PM_GameSettings_t DmdDev_PM_GameSettings;
DmdDev_Set_4_Colors_Palette_t DmdDev_Set_4_Colors_Palette;
DmdDev_Console_Data_t DmdDev_Console_Data;
DmdDev_Console_Input_Ptr_t DmdDev_Console_Input_Ptr;
DmdDev_Render_16_Shades_t DmdDev_Render_16_Shades;
DmdDev_Render_4_Shades_t DmdDev_Render_4_Shades;
DmdDev_Render_16_Shades_with_Raw_t DmdDev_Render_16_Shades_with_Raw;
DmdDev_Render_4_Shades_with_Raw_t DmdDev_Render_4_Shades_with_Raw;
DmdDev_render_PM_Alphanumeric_Frame_t DmdDev_render_PM_Alphanumeric_Frame;
DmdDev_render_PM_Alphanumeric_Dim_Frame_t DmdDev_render_PM_Alphanumeric_Dim_Frame;

/// <summary>
/// Initialize a DMDext DLL instance getting its function procs
/// </summary>
/// <param name="VPpath">VPinMame path where DmdDevice64.dll should be</param>
/// <param name="romname">name of the rom (without the ".zip" extension)</param>
/// <returns></returns>
const char* InitDmdDevice(char* VPpath,char* romname)
{
    bool DmdDev = false;
    char DmdDev_filename[MAX_PATH];
    sprintf_s(DmdDev_filename, MAX_PATH, "%sDmdDevice64.dll", VPpath);
    DmdDev_hModule = LoadLibraryExA(DmdDev_filename, NULL, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (!DmdDev_hModule) DmdDev_hModule = LoadLibraryA(DmdDev_filename);
    if (DmdDev_hModule) {
        DmdDev_Open = (DmdDev_Open_t)GetProcAddress(DmdDev_hModule, "Open");
        DmdDev_PM_GameSettings = (DmdDev_PM_GameSettings_t)GetProcAddress(DmdDev_hModule, "PM_GameSettings");
        DmdDev_Close = (DmdDev_Close_t)GetProcAddress(DmdDev_hModule, "Close");
        DmdDev_Render_4_Shades = (DmdDev_Render_4_Shades_t)GetProcAddress(DmdDev_hModule, "Render_4_Shades");
        DmdDev_Render_16_Shades = (DmdDev_Render_16_Shades_t)GetProcAddress(DmdDev_hModule, "Render_16_Shades");
        DmdDev_Render_4_Shades_with_Raw = (DmdDev_Render_4_Shades_with_Raw_t)GetProcAddress(DmdDev_hModule, "Render_4_Shades_with_Raw");
        DmdDev_Render_16_Shades_with_Raw = (DmdDev_Render_16_Shades_with_Raw_t)GetProcAddress(DmdDev_hModule, "Render_16_Shades_with_Raw");
        DmdDev_render_PM_Alphanumeric_Frame = (DmdDev_render_PM_Alphanumeric_Frame_t)GetProcAddress(DmdDev_hModule, "Render_PM_Alphanumeric_Frame");
        DmdDev_render_PM_Alphanumeric_Dim_Frame = (DmdDev_render_PM_Alphanumeric_Dim_Frame_t)GetProcAddress(DmdDev_hModule, "Render_PM_Alphanumeric_Dim_Frame");
        DmdDev_Set_4_Colors_Palette = (DmdDev_Set_4_Colors_Palette_t)GetProcAddress(DmdDev_hModule, "Set_4_Colors_Palette");
        DmdDev_Console_Data = (DmdDev_Console_Data_t)GetProcAddress(DmdDev_hModule, "Console_Data");
        DmdDev_Console_Input_Ptr = (DmdDev_Console_Input_Ptr_t)GetProcAddress(DmdDev_hModule, "Console_Input_Ptr");

        if (!DmdDev_Open || !DmdDev_Close || !DmdDev_PM_GameSettings || !DmdDev_Render_4_Shades || !DmdDev_Render_16_Shades || !DmdDev_render_PM_Alphanumeric_Frame)
        {
            DmdDev = false;
            FreeLibrary(DmdDev_hModule);
            return "Error while looking for DmdDevice64.dll functions";
        }
        else DmdDev = true;
    }
    else
    {
        return "Error while loading DmdDevice64.dll in VPinMame directory, check that the file is there and unblocked!";
    }
#define rStart 0xff
#define gStart 0xe0
#define bStart  0x20
    rgb24 color0 = { 0,0,0 };
    rgb24 color33 = { rStart * 33 / 100, gStart * 33 / 100, bStart * 33 / 100 };
    rgb24 color66 = { rStart * 66 / 100, gStart * 66 / 100, bStart * 66 / 100 };
    rgb24 color100 = { rStart , gStart, bStart };
    if (DmdDev) {
        DmdDev_Open();
        // structure to fill for dmd-extensions PinMameDevice\DmdDevice.cs::InternalGameSettingsDevice
        tPMoptions Options;
        const tPMoptions* pOptions = &Options;
        Options.dmd_red = rStart; // we just need to fill the full color
        Options.dmd_green = gStart;
        Options.dmd_blue = bStart;
        Options.dmd_colorize = 1;// and that we want it colorized
        DmdDev_Set_4_Colors_Palette(color0, color33, color66, color100);
        DmdDev_PM_GameSettings(romname, 0, *pOptions);
    }
    return "";
}
/// <summary>
/// Close the DMDext DLL instance
/// </summary>
void StopDmdDevice(void)
{
    DmdDev_Close();
    FreeLibrary(DmdDev_hModule);
}
/// <summary>
/// Sends a frame to DMDext for it to be displayed colorized through the serum dll. Precalculates and returns the original frame display.
/// </summary>
/// <param name="nofr">number of the frame</param>
/// <param name="nocolors">how many colors: 4 or 16</param>
/// <param name="pframes">original frames pointer</param>
/// <param name="(width,height)">size of original frames</param>
/// <param name="testerOriginalFrame">pointer that will receive the image in RGBA8888 of the original frame</param>
void SendFrameToTester(unsigned int nofr, unsigned int nocolors, UINT8* pframes, unsigned int width, unsigned int height, unsigned char* testerOriginalFrame)
{
    int pixsize = 128 / height;
    int offsetx = 0;
    if (width == 192) offsetx = 64;
    for (int tj = 0; tj < 128; tj++)
    {
        for (int ti = 0; ti < 512; ti++)
        {
            if ((ti < offsetx) || (ti >= 512 - offsetx)) memset(&testerOriginalFrame[(tj * 512 + ti) * 4], 0, 4);
            else
            {
                float coefcol = (float)pframes[nofr * width * height + (ti - offsetx) / pixsize + tj / pixsize * width] / (float)nocolors;
                testerOriginalFrame[(tj * 512 + ti) * 4 + 2] = (unsigned char)(255 * coefcol);
                testerOriginalFrame[(tj * 512 + ti) * 4 + 1] = (unsigned char)(127 * coefcol);
                testerOriginalFrame[(tj * 512 + ti) * 4] = 0;
                testerOriginalFrame[(tj * 512 + ti) * 4 + 3] = 255;
            }
        }
    }
    // send the original frame to dmdext so that it colorizes and displays it
    if (nocolors == 4) DmdDev_Render_4_Shades(width, height, &pframes[nofr * width * height]);
    else DmdDev_Render_16_Shades(width, height, &pframes[nofr * width * height]);
}
/// <summary>
/// Load a txt dump to test it with a Serum file
/// </summary>
/// <param name="path">path of the file without the filename</param>
/// <param name="_gamename">name of the rom (=file) without the "zip" extension</param>
/// <param name="ppframes">pointer to a UINT8* that will be malloc'ed and filled with the frames (don't forget to free)</param>
/// <param name="pptimecodes">pointer to a UINT* that will be malloc'ed and filled with frame durations (don't forget to free)</param>
/// <param name="nframes">pointer to an int that will receive the number of frames</param>
/// <param name="nocolors">number of colors that should be in the dump (returns an error if doesn't correspond to the rom)</param>
/// <param name="width">width of the rom (returns an error if doesn't correspond to the rom)</param>
/// <param name="height">height of the rom (returns an error if doesn't correspond to the rom)</param>
/// <returns>"" if no error, an error message if an error occurred</returns>
const char* ImportDump(char* path, char* _gamename, UINT8** ppframes, UINT** pptimecodes, UINT* nframes, UINT nocolors, UINT width, UINT height)
{
    // don't forget to free *ppframes and *pptimecodes when the dump is not needed anymore
    int frameIndex = 0;
    int lineIndex = 0;
    int is16Colors = 0;
    *ppframes = NULL;
    *pptimecodes = NULL;
    FILE* fp;
    //char _filename[MAX_PATH];
    //sprintf_s(_filename, MAX_PATH, "%s%s.txt", path, _gamename);
    fopen_s(&fp, path, "r");
    if (fp == NULL) return "Error opening dump file for testing";
    fseek(fp, 0L, SEEK_END);
    int file_size = ftell(fp);
    fseek(fp, 0L, SEEK_SET);
    char* stemp = (char*)malloc(file_size * sizeof(char));
    if (!stemp)
    {
        fclose(fp);
        return "Unable to get memory to read the dump file for testing";
    }
    fread(stemp, sizeof(char), file_size, fp);
    fclose(fp);
    int count = 0;
    for (int i = 0; i < file_size; i++) 
    {
        if (stemp[i] == 'x') count++;
    }
    free(stemp);

    //on doit compter les lignes comme dans Get_Frames_Ptr_Size_And_Number_Of_Colors() pour v�rifier si on a des frames de 128x16


    if (count > 32768)
    {
        MessageBoxA(NULL, "The tester can only display 32768 frames, the first 32768 frames will be loaded", "Caution", MB_OK);
        count = 32768;
    }
    *ppframes = (unsigned char*)malloc(count * sizeof(unsigned char*) * width * height);
    memset(*ppframes, 0, count * sizeof(unsigned char*) * width * height);
    if (!*ppframes) return "Unable to get memory to store the dump file frames for testing";
    *pptimecodes = (unsigned int*)malloc(count * sizeof(unsigned int));
    if (!*pptimecodes)
    {
        free(*ppframes);
        *ppframes = NULL;
        return "Unable to get memory to store the dump file timecodes for testing";
    }
    char line[200];
    int twidth = 0, theight = 0;
    fopen_s(&fp, path, "r");
    bool mustpad128x16 = false;
    while (fgets(line, sizeof(line), fp)) {
        lineIndex++;
        // read timecode
        char timecodeHex[20];
        sscanf_s(line, "%s", timecodeHex,20);
        (*pptimecodes)[frameIndex] = (unsigned int)strtoul(timecodeHex, NULL, 16);
        // read image data
        if (frameIndex == 0)
        {
            fgets(line, sizeof(line), fp);
            // first frame: calculate width and height
            twidth = (int)strlen(line) - 1;
            theight = 1;
            int tlineIndex = lineIndex;
            while (fgets(line, sizeof(line), fp) != NULL && strlen(line) == twidth + 1) {
                theight++;
                tlineIndex++;
            }
            fseek(fp, 0L, SEEK_SET);
            fgets(line, sizeof(line), fp);
        }
        else
        {
            (*pptimecodes)[frameIndex - 1] = (*pptimecodes)[frameIndex] - (*pptimecodes)[frameIndex - 1];
            if ((*pptimecodes)[frameIndex - 1] < 5) (*pptimecodes)[frameIndex - 1] = 5;
            if ((*pptimecodes)[frameIndex - 1] > 5000) (*pptimecodes)[frameIndex - 1] = 5000;
        }
        if (theight == 16 && twidth == 128) mustpad128x16 = true;
        if ((mustpad128x16 && (width != 128 || height != 32)) ||
            (!mustpad128x16 && (twidth != width || theight != height)))
        {
            free(*pptimecodes);
            free(*ppframes);
            *ppframes = NULL;
            *pptimecodes = NULL;
            return "The width and/or height of the dump file to test are not the same as the ones of the project";
        }
        if (frameIndex == count - 1) (*pptimecodes)[frameIndex] = 100;
        UINT offsypad = 0;
        if (mustpad128x16) offsypad = 8;
        for (UINT y = 0; y < (UINT)theight; y++)
        {
            fgets(line, sizeof(line), fp);
            if (strlen(line) != width + 1)
            {
                free(*pptimecodes);
                free(*ppframes);
                *ppframes = NULL;
                *pptimecodes = NULL;
                return "Invalid frame found in the dump file to test";
            }
            for (UINT x = 0; x < width; x++)
            {
                char c = line[x];
                unsigned char value;
                if (c >= '0' && c <= '3')
                {
                    value = (unsigned char)(c - '0');
                }
                else if (c >= '0' && c <= '9')
                {
                    value = (unsigned char)(c - '0');
                    is16Colors = 1;
                }
                else if (c >= 'a' && c <= 'f')
                {
                    value = (unsigned char)(c - 'a' + 10);
                    is16Colors = 1;
                }
                else if (c >= 'A' && c <= 'F')
                {
                    value = (unsigned char)(c - 'A' + 10);
                    is16Colors = 1;
                }
                else
                {
                    free(*pptimecodes);
                    free(*ppframes);
                    *ppframes = NULL;
                    *pptimecodes = NULL;
                    return "Invalid character found in the dump file to test";
                }
                (*ppframes)[frameIndex * width * height + (y + offsypad) * width + x] = value;
            }
        }
        fgets(line, sizeof(line), fp);
        frameIndex++;
        if (frameIndex == 32768) break;
    }
    fclose(fp);
    *nframes = frameIndex;
    if (is16Colors && (nocolors!=16))
    {
        free(*pptimecodes);
        free(*ppframes);
        *ppframes = NULL;
        *pptimecodes = NULL;
        return "Invalid number of different colors found in the dump file to test";
    }
    return "";
}
