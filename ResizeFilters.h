#pragma once
#include "resource.h"
#include <BaseTsd.h>

void Filter_Scale2X(UINT16* inframe, UINT16* outframe, UINT32 width, UINT32 height);
void Filter_Scale2Xmask(UINT16* inframe, UINT16* outframe, UINT8* masks, UINT8* maskd, UINT32 width, UINT32 height);

