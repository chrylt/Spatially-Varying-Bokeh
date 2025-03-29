///////////////////////////////////////////////////////////////////////////////
//     Filter-Adapted Spatio-Temporal Sampling With General Distributions    //
//        Copyright (c) 2025 Electronic Arts Inc. All rights reserved.       //
///////////////////////////////////////////////////////////////////////////////

#pragma once

enum class LogLevel : int
{
    Info,
    Warn,
    Error
};
using TLogFn = void (*)(LogLevel level, const char* msg, ...);
