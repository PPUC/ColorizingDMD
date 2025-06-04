#pragma once

// Initilize the OLE object to start the Pup pack
const char* pt_Init();
// Start the Pup session with the name of the Pup Pack (the name of the directory in "PinUPSystem\PUPVideos")
const char* pt_B2sInit(wchar_t* puppackname);
// Send a trigger ID to the Pup pack
const char* pt_B2sData(int IdNum);
// Close the Ole object
void pt_Close();
