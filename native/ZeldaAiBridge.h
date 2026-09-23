#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// Installed in PadMgr_RequestPadData, after the normal controller copy.
void ZeldaAiBridge_ConsumeInput(int32_t controller, void* input, int32_t mode);
#ifdef __cplusplus
}
#endif
