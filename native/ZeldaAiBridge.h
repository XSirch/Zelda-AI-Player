#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
void ZeldaAiBridge_OverrideInput(int32_t controller, uint16_t* buttons, int8_t* stickX, int8_t* stickY);
#ifdef __cplusplus
}
#endif
