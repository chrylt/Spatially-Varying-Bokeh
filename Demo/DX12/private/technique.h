///////////////////////////////////////////////////////////////////////////////
//     Filter-Adapted Spatio-Temporal Sampling With General Distributions    //
//        Copyright (c) 2025 Electronic Arts Inc. All rights reserved.       //
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <d3d12.h>
#include <array>
#include <vector>
#include <unordered_map>
#include "DX12Utils/dxutils.h"

namespace FastBokeh
{
    using uint = unsigned int;
    using uint2 = std::array<uint, 2>;
    using uint3 = std::array<uint, 3>;
    using uint4 = std::array<uint, 4>;

    using int2 = std::array<int, 2>;
    using int3 = std::array<int, 3>;
    using int4 = std::array<int, 4>;
    using float2 = std::array<float, 2>;
    using float3 = std::array<float, 3>;
    using float4 = std::array<float, 4>;
    using float4x4 = std::array<std::array<float, 4>, 4>;

    enum class LensRNG: int
    {
        UniformCircleWhite_PCG,
        UniformCircleWhite,
        UniformCircleBlue,
        UniformHexagonWhite,
        UniformHexagonBlue,
        UniformHexagonICDF_White,
        UniformHexagonICDF_Blue,
        UniformStarWhite,
        UniformStarBlue,
        UniformStarICDF_White,
        UniformStarICDF_Blue,
        NonUniformStarWhite,
        NonUniformStarBlue,
        NonUniformStar2White,
        NonUniformStar2Blue,
        LKCP6White,
        LKCP6Blue,
        LKCP204White,
        LKCP204Blue,
        LKCP204ICDF_White,
        LKCP204ICDF_Blue,
    };

    enum class DOFMode: int
    {
        Off,
        PathTraced,
        PostProcessing,
        Realistic,
    };

    enum class PixelJitterType: int
    {
        None,
        PerPixel,
        Global,
    };

    enum class MaterialSets: int
    {
        None,
        Interior,
        Exterior,
    };

    enum class NoiseTexExtends: int
    {
        None,
        White,
        Shuffle1D,
        Shuffle1DHilbert,
    };

    enum class GatherDOF_LensRNG: int
    {
        UniformCircleWhite_PCG,
        UniformCircleWhite,
        UniformCircleBlue,
        UniformHexagonWhite,
        UniformHexagonBlue,
        UniformHexagonICDF_White,
        UniformHexagonICDF_Blue,
        UniformStarWhite,
        UniformStarBlue,
        UniformStarICDF_White,
        UniformStarICDF_Blue,
        NonUniformStarWhite,
        NonUniformStarBlue,
        NonUniformStar2White,
        NonUniformStar2Blue,
        LKCP6White,
        LKCP6Blue,
        LKCP204White,
        LKCP204Blue,
        LKCP204ICDF_White,
        LKCP204ICDF_Blue,
    };

    enum class GatherDOF_NoiseTexExtends: int
    {
        None,
        White,
        Shuffle1D,
        Shuffle1DHilbert,
    };

    enum class ToneMap_ToneMappingOperation: int
    {
        None, // Only exposure is applied
        Reinhard_Simple, // Portions of this are based on https://64.github.io/tonemapping/
        ACES_Luminance, // Portions of this are based on https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
        ACES, // Portions of this are based on https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl
    };

    inline const char* EnumToString(LensRNG value, bool displayString = false)
    {
        switch(value)
        {
            case LensRNG::UniformCircleWhite_PCG: return displayString ? "UniformCircleWhite_PCG" : "UniformCircleWhite_PCG";
            case LensRNG::UniformCircleWhite: return displayString ? "UniformCircleWhite" : "UniformCircleWhite";
            case LensRNG::UniformCircleBlue: return displayString ? "UniformCircleBlue" : "UniformCircleBlue";
            case LensRNG::UniformHexagonWhite: return displayString ? "UniformHexagonWhite" : "UniformHexagonWhite";
            case LensRNG::UniformHexagonBlue: return displayString ? "UniformHexagonBlue" : "UniformHexagonBlue";
            case LensRNG::UniformHexagonICDF_White: return displayString ? "UniformHexagonICDF_White" : "UniformHexagonICDF_White";
            case LensRNG::UniformHexagonICDF_Blue: return displayString ? "UniformHexagonICDF_Blue" : "UniformHexagonICDF_Blue";
            case LensRNG::UniformStarWhite: return displayString ? "UniformStarWhite" : "UniformStarWhite";
            case LensRNG::UniformStarBlue: return displayString ? "UniformStarBlue" : "UniformStarBlue";
            case LensRNG::UniformStarICDF_White: return displayString ? "UniformStarICDF_White" : "UniformStarICDF_White";
            case LensRNG::UniformStarICDF_Blue: return displayString ? "UniformStarICDF_Blue" : "UniformStarICDF_Blue";
            case LensRNG::NonUniformStarWhite: return displayString ? "NonUniformStarWhite" : "NonUniformStarWhite";
            case LensRNG::NonUniformStarBlue: return displayString ? "NonUniformStarBlue" : "NonUniformStarBlue";
            case LensRNG::NonUniformStar2White: return displayString ? "NonUniformStar2White" : "NonUniformStar2White";
            case LensRNG::NonUniformStar2Blue: return displayString ? "NonUniformStar2Blue" : "NonUniformStar2Blue";
            case LensRNG::LKCP6White: return displayString ? "LKCP6White" : "LKCP6White";
            case LensRNG::LKCP6Blue: return displayString ? "LKCP6Blue" : "LKCP6Blue";
            case LensRNG::LKCP204White: return displayString ? "LKCP204White" : "LKCP204White";
            case LensRNG::LKCP204Blue: return displayString ? "LKCP204Blue" : "LKCP204Blue";
            case LensRNG::LKCP204ICDF_White: return displayString ? "LKCP204ICDF_White" : "LKCP204ICDF_White";
            case LensRNG::LKCP204ICDF_Blue: return displayString ? "LKCP204ICDF_Blue" : "LKCP204ICDF_Blue";
            default: return nullptr;
        }
    }

    inline const char* EnumToString(DOFMode value, bool displayString = false)
    {
        switch(value)
        {
            case DOFMode::Off: return displayString ? "Off" : "Off";
            case DOFMode::PathTraced: return displayString ? "PathTraced" : "PathTraced";
            case DOFMode::PostProcessing: return displayString ? "PostProcessing" : "PostProcessing";
            case DOFMode::Realistic: return displayString ? "Realistic" : "Realistic";
            default: return nullptr;
        }
    }

    inline const char* EnumToString(PixelJitterType value, bool displayString = false)
    {
        switch(value)
        {
            case PixelJitterType::None: return displayString ? "None" : "None";
            case PixelJitterType::PerPixel: return displayString ? "PerPixel" : "PerPixel";
            case PixelJitterType::Global: return displayString ? "Global" : "Global";
            default: return nullptr;
        }
    }

    inline const char* EnumToString(MaterialSets value, bool displayString = false)
    {
        switch(value)
        {
            case MaterialSets::None: return displayString ? "None" : "None";
            case MaterialSets::Interior: return displayString ? "Interior" : "Interior";
            case MaterialSets::Exterior: return displayString ? "Exterior" : "Exterior";
            default: return nullptr;
        }
    }

    inline const char* EnumToString(NoiseTexExtends value, bool displayString = false)
    {
        switch(value)
        {
            case NoiseTexExtends::None: return displayString ? "None" : "None";
            case NoiseTexExtends::White: return displayString ? "White" : "White";
            case NoiseTexExtends::Shuffle1D: return displayString ? "Shuffle1D" : "Shuffle1D";
            case NoiseTexExtends::Shuffle1DHilbert: return displayString ? "Shuffle1DHilbert" : "Shuffle1DHilbert";
            default: return nullptr;
        }
    }

    inline const char* EnumToString(GatherDOF_LensRNG value, bool displayString = false)
    {
        switch(value)
        {
            case GatherDOF_LensRNG::UniformCircleWhite_PCG: return displayString ? "UniformCircleWhite_PCG" : "UniformCircleWhite_PCG";
            case GatherDOF_LensRNG::UniformCircleWhite: return displayString ? "UniformCircleWhite" : "UniformCircleWhite";
            case GatherDOF_LensRNG::UniformCircleBlue: return displayString ? "UniformCircleBlue" : "UniformCircleBlue";
            case GatherDOF_LensRNG::UniformHexagonWhite: return displayString ? "UniformHexagonWhite" : "UniformHexagonWhite";
            case GatherDOF_LensRNG::UniformHexagonBlue: return displayString ? "UniformHexagonBlue" : "UniformHexagonBlue";
            case GatherDOF_LensRNG::UniformHexagonICDF_White: return displayString ? "UniformHexagonICDF_White" : "UniformHexagonICDF_White";
            case GatherDOF_LensRNG::UniformHexagonICDF_Blue: return displayString ? "UniformHexagonICDF_Blue" : "UniformHexagonICDF_Blue";
            case GatherDOF_LensRNG::UniformStarWhite: return displayString ? "UniformStarWhite" : "UniformStarWhite";
            case GatherDOF_LensRNG::UniformStarBlue: return displayString ? "UniformStarBlue" : "UniformStarBlue";
            case GatherDOF_LensRNG::UniformStarICDF_White: return displayString ? "UniformStarICDF_White" : "UniformStarICDF_White";
            case GatherDOF_LensRNG::UniformStarICDF_Blue: return displayString ? "UniformStarICDF_Blue" : "UniformStarICDF_Blue";
            case GatherDOF_LensRNG::NonUniformStarWhite: return displayString ? "NonUniformStarWhite" : "NonUniformStarWhite";
            case GatherDOF_LensRNG::NonUniformStarBlue: return displayString ? "NonUniformStarBlue" : "NonUniformStarBlue";
            case GatherDOF_LensRNG::NonUniformStar2White: return displayString ? "NonUniformStar2White" : "NonUniformStar2White";
            case GatherDOF_LensRNG::NonUniformStar2Blue: return displayString ? "NonUniformStar2Blue" : "NonUniformStar2Blue";
            case GatherDOF_LensRNG::LKCP6White: return displayString ? "LKCP6White" : "LKCP6White";
            case GatherDOF_LensRNG::LKCP6Blue: return displayString ? "LKCP6Blue" : "LKCP6Blue";
            case GatherDOF_LensRNG::LKCP204White: return displayString ? "LKCP204White" : "LKCP204White";
            case GatherDOF_LensRNG::LKCP204Blue: return displayString ? "LKCP204Blue" : "LKCP204Blue";
            case GatherDOF_LensRNG::LKCP204ICDF_White: return displayString ? "LKCP204ICDF_White" : "LKCP204ICDF_White";
            case GatherDOF_LensRNG::LKCP204ICDF_Blue: return displayString ? "LKCP204ICDF_Blue" : "LKCP204ICDF_Blue";
            default: return nullptr;
        }
    }

    inline const char* EnumToString(GatherDOF_NoiseTexExtends value, bool displayString = false)
    {
        switch(value)
        {
            case GatherDOF_NoiseTexExtends::None: return displayString ? "None" : "None";
            case GatherDOF_NoiseTexExtends::White: return displayString ? "White" : "White";
            case GatherDOF_NoiseTexExtends::Shuffle1D: return displayString ? "Shuffle1D" : "Shuffle1D";
            case GatherDOF_NoiseTexExtends::Shuffle1DHilbert: return displayString ? "Shuffle1DHilbert" : "Shuffle1DHilbert";
            default: return nullptr;
        }
    }

    inline const char* EnumToString(ToneMap_ToneMappingOperation value, bool displayString = false)
    {
        switch(value)
        {
            case ToneMap_ToneMappingOperation::None: return displayString ? "None" : "None";
            case ToneMap_ToneMappingOperation::Reinhard_Simple: return displayString ? "Reinhard_Simple" : "Reinhard_Simple";
            case ToneMap_ToneMappingOperation::ACES_Luminance: return displayString ? "ACES_Luminance" : "ACES_Luminance";
            case ToneMap_ToneMappingOperation::ACES: return displayString ? "ACES" : "ACES";
            default: return nullptr;
        }
    }

    struct ContextInternal
    {
        ID3D12QueryHeap* m_TimestampQueryHeap = nullptr;
        ID3D12Resource* m_TimestampReadbackBuffer = nullptr;

        static ID3D12CommandSignature* s_commandSignatureDispatch;

        struct Struct_PixelDebugStruct
        {
            float2 MousePos = {0.000000f, 0.000000f};
            uint MaterialID = 0;
            float HitT = 0.000000f;
            float3 WorldPos = {0.000000f, 0.000000f, 0.000000f};
            float centerDepth = 0.000000f;
            float centerSize = 0.000000f;
            float tot = 0.000000f;
        };

        struct Struct__RayGenCB
        {
            unsigned int Accumulate = true;
            unsigned int AlbedoMode = 1.0f;  // if true, returns albedo * AlbedoModeAlbedoMultiplier + emissive at primary hit
            float AlbedoModeAlbedoMultiplier = 0.500000f;  // How much to multiply albedo by in albedo mode, to darken it or lighten it
            float _padding0 = 0.000000f;  // Padding
            float2 AnamorphicScaling = {1.000000f, 1.000000f};  // Defaults to 1.0, 1.0 for no anamorphic effects. Elongates the aperture, does not simulate anamorphic elements.
            unsigned int Animate = true;
            float ApertureRadius = 1.000000f;
            float3 CameraPos = {0.000000f, 0.000000f, 0.000000f};
            int DOF = (int)DOFMode::Off;
            float DepthNearPlane = 0.100000f;
            float FocalLength = 28.000000f;
            uint FrameIndex = 0;
            float _padding1 = 0.000000f;  // Padding
            float4x4 InvViewMtx = {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f};
            float4x4 InvViewProjMtx = {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f};
            unsigned int JitterNoiseTextures = false;  // The noise textures are 8 bit unorms. This adds a random value between -0.5/255 and +0.5/255 to fill in the unset bits with white noise.
            int JitterPixels = (int)PixelJitterType::PerPixel;  // Provides Antialiasing
            int LensRNGExtend = (int)NoiseTexExtends::None;  // How to extend the noise textures
            int LensRNGSource = (int)LensRNG::UniformCircleWhite;
            float MaterialEmissiveMultiplier = 1.000000f;
            int MaterialSet = (int)MaterialSets::None;
            float2 _padding2 = {0.000000f, 0.000000f};  // Padding
            float4 MouseState = {0.000000f, 0.000000f, 0.000000f, 0.000000f};
            unsigned int NoImportanceSampling = false;  // If true, the FAST noise textures will not be used, and 
            uint NumBounces = 4;  // How many bounces the rays are allowed
            float2 _padding3 = {0.000000f, 0.000000f};  // Padding
            float3 OcclusionSettings = {1.000000f, 1.000000f, 1.000000f};  // Pushes the bounding square of the lens outwards and clips against a unit circle. 1,1,1 means no occlusion. x is how far from the center of the screen to start moving the square. 0 is center, 1 is the corner.  y is how much to scale the lens bounding square by.  z is how far to move the square, as the pixel is farther from where the occlusion begins. Reasonable settings are 0, 0.1, 1.25.
            float _padding4 = 0.000000f;  // Padding
            float2 PetzvalScaling = {1.000000f, 1.000000f};  // Scales bokeh on each axis depending on screen position. Fakes the effect. Defaults to 1.0, 1.0 for no elongation.
            float RayPosNormalNudge = 0.100000f;
            uint SamplesPerPixelPerFrame = 1;
            float SkyBrightness = 10.000000f;
            float3 SkyColor = {1.000000f, 1.000000f, 1.000000f};
            float SmallLightBrightness = 1.000000f;
            float SmallLightRadius = 1.000000f;
            float2 _padding5 = {0.000000f, 0.000000f};  // Padding
            float3 SmallLightsColor = {1.000000f, 1.000000f, 1.000000f};
            unsigned int SmallLightsColorful = false;  // If true, makes the small lights colorful, else makes them all the same color
        };

        struct Struct__GatherDOF_SetupCSCB
        {
            float GatherDOF_FarTransitionRegion = 200.000000f;  // Fade distance in world units
            float GatherDOF_FocalDistance = 500.000000f;  // Anything closer than this is considered near field
            float GatherDOF_FocalLength = 75.000000f;  // Focal length in mm (Camera property e.g. 75mm)
            float GatherDOF_FocalRegion = 100.000000f;  // The size in world units of the middle range which is in focus
            float GatherDOF_NearTransitionRegion = 50.000000f;  // Fade distance in world units
            float GatherDOF_Scale = 0.500000f;  // Camera property e.g. 0.5f, like aperture
            unsigned int GatherDOF_SuppressBokeh = false;  // If true, blurs out of focus areas, but reduces the Bokeh effect of small bright lights
            float _padding0 = 0.000000f;  // Padding
        };

        struct Struct__GatherDOF_BlurFarCSCB
        {
            unsigned int GatherDOF_AnimateNoiseTextures = true;
            uint GatherDOF_BlurTapCount = 8;  // 8 for high quality, 6 for low quality. Used in a double for loop, so it's this number squared.
            uint GatherDOF_FrameIndex = 0;
            float _padding0 = 0.000000f;  // Padding
            float4 GatherDOF_KernelSize = {10.000000f, 15.000000f, 5.000000f, 0.000000f};  // x = size of the bokeh blur radius in texel space. y = rotation in radians to apply to the bokeh shape. z = Number of edge of the polygon (number of blades). 0: circle. 4: square, 6: hexagon...
            unsigned int GatherDOF_UseNoiseTextures = false;
            unsigned int JitterNoiseTextures = false;  // The noise textures are 8 bit unorms. This adds a random value between -0.5/255 and +0.5/255 to fill in the unset bits with white noise.
            int LensRNGExtend = (int)NoiseTexExtends::None;  // How to extend the noise textures
            int LensRNGSource = (int)LensRNG::UniformCircleWhite;
        };

        struct Struct__GatherDOF_NearBlurCB
        {
            unsigned int GatherDOF_AnimateNoiseTextures = true;
            uint GatherDOF_BlurTapCount = 8;  // 8 for high quality, 6 for low quality. Used in a double for loop, so it's this number squared.
            uint GatherDOF_FrameIndex = 0;
            float _padding0 = 0.000000f;  // Padding
            float4 GatherDOF_KernelSize = {10.000000f, 15.000000f, 5.000000f, 0.000000f};  // x = size of the bokeh blur radius in texel space. y = rotation in radians to apply to the bokeh shape. z = Number of edge of the polygon (number of blades). 0: circle. 4: square, 6: hexagon...
            unsigned int GatherDOF_UseNoiseTextures = false;
            unsigned int JitterNoiseTextures = false;  // The noise textures are 8 bit unorms. This adds a random value between -0.5/255 and +0.5/255 to fill in the unset bits with white noise.
            int LensRNGExtend = (int)NoiseTexExtends::None;  // How to extend the noise textures
            int LensRNGSource = (int)LensRNG::UniformCircleWhite;
        };

        struct Struct__GatherDOF_RecombineCSCB
        {
            unsigned int GatherDOF_DoFarField = true;  // Whether or not to do the far field
            unsigned int GatherDOF_DoNearField = true;  // Whether or not to do the near field
            float GatherDOF_FarTransitionRegion = 200.000000f;  // Fade distance in world units
            float GatherDOF_FocalDistance = 500.000000f;  // Anything closer than this is considered near field
            float GatherDOF_FocalLength = 75.000000f;  // Focal length in mm (Camera property e.g. 75mm)
            float GatherDOF_FocalRegion = 100.000000f;  // The size in world units of the middle range which is in focus
            float GatherDOF_NearTransitionRegion = 50.000000f;  // Fade distance in world units
            float GatherDOF_Scale = 0.500000f;  // Camera property e.g. 0.5f, like aperture
        };

        struct Struct__GaussBlur_GaussBlurCSCB
        {
            float GaussBlur_Sigma = 1.000000f;  // Strength of blur. Standard deviation of gaussian distribution.
            float3 _padding0 = {0.000000f, 0.000000f, 0.000000f};  // Padding
        };

        struct Struct__TemporalAccumulation_AccumulateCB
        {
            unsigned int CameraChanged = false;
            float TemporalAccumulation_Alpha = 0.100000f;  // For exponential moving average. From 0 to 1. TAA commonly uses 0.1.
            unsigned int TemporalAccumulation_Enabled = true;
            float _padding0 = 0.000000f;  // Padding
        };

        struct Struct__ToneMap_TonemapCB
        {
            float ToneMap_ExposureFStops = 0.000000f;
            int ToneMap_ToneMapper = (int)ToneMap_ToneMappingOperation::None;
            float2 _padding0 = {0.000000f, 0.000000f};  // Padding
        };

        struct Struct__GatherDOF_FloodFillFarCS_0CB
        {
            unsigned int GatherDOF_AnimateNoiseTextures = true;
            unsigned int GatherDOF_DoFarFieldFloodFill = true;  // Whether to do flood fill on the far field
            unsigned int GatherDOF_DoNearFieldFloodFill = true;  // Whether to do flood fill on the near field
            uint GatherDOF_FloodFillTapCount = 4;  // 4 for high quality, 3 for low quality. Used in a double for loop, so it's this number squared.
            uint GatherDOF_FrameIndex = 0;
            float3 _padding0 = {0.000000f, 0.000000f, 0.000000f};  // Padding
            float4 GatherDOF_KernelSize = {10.000000f, 15.000000f, 5.000000f, 0.000000f};  // x = size of the bokeh blur radius in texel space. y = rotation in radians to apply to the bokeh shape. z = Number of edge of the polygon (number of blades). 0: circle. 4: square, 6: hexagon...
            unsigned int GatherDOF_UseNoiseTextures = false;
            unsigned int JitterNoiseTextures = false;  // The noise textures are 8 bit unorms. This adds a random value between -0.5/255 and +0.5/255 to fill in the unset bits with white noise.
            int LensRNGExtend = (int)NoiseTexExtends::None;  // How to extend the noise textures
            int LensRNGSource = (int)LensRNG::UniformCircleWhite;
        };

        struct Struct__GatherDOF_FloodFillFarCS_1CB
        {
            unsigned int GatherDOF_AnimateNoiseTextures = true;
            unsigned int GatherDOF_DoFarFieldFloodFill = true;  // Whether to do flood fill on the far field
            unsigned int GatherDOF_DoNearFieldFloodFill = true;  // Whether to do flood fill on the near field
            uint GatherDOF_FloodFillTapCount = 4;  // 4 for high quality, 3 for low quality. Used in a double for loop, so it's this number squared.
            uint GatherDOF_FrameIndex = 0;
            float3 _padding0 = {0.000000f, 0.000000f, 0.000000f};  // Padding
            float4 GatherDOF_KernelSize = {10.000000f, 15.000000f, 5.000000f, 0.000000f};  // x = size of the bokeh blur radius in texel space. y = rotation in radians to apply to the bokeh shape. z = Number of edge of the polygon (number of blades). 0: circle. 4: square, 6: hexagon...
            unsigned int GatherDOF_UseNoiseTextures = false;
            unsigned int JitterNoiseTextures = false;  // The noise textures are 8 bit unorms. This adds a random value between -0.5/255 and +0.5/255 to fill in the unset bits with white noise.
            int LensRNGExtend = (int)NoiseTexExtends::None;  // How to extend the noise textures
            int LensRNGSource = (int)LensRNG::UniformCircleWhite;
        };

        // Variables
        uint variable_FrameIndex = 0;
        const bool variable___literal_0 = false;  // Made to replace variable "sRGB" with a constant value in subgraph node "GatherDOF"
        uint variable_GatherDOF_FrameIndex = 0;
        const float variable___literal_1 = 0.995000f;  // Made to replace variable "Support" with a constant value in subgraph node "GaussBlur"
        const bool variable___literal_2 = false;  // Made to replace variable "sRGB" with a constant value in subgraph node "GaussBlur"

        ID3D12Resource* texture_ColorHDR = nullptr;
        unsigned int texture_ColorHDR_size[3] = { 0, 0, 0 };
        unsigned int texture_ColorHDR_numMips = 0;
        DXGI_FORMAT texture_ColorHDR_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture_ColorHDR_flags =  D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        const D3D12_RESOURCE_STATES c_texture_ColorHDR_endingState = D3D12_RESOURCE_STATE_COPY_SOURCE;

        ID3D12Resource* buffer_PixelDebug = nullptr;
        DXGI_FORMAT buffer_PixelDebug_format = DXGI_FORMAT_UNKNOWN; // For typed buffers, the type of the buffer
        unsigned int buffer_PixelDebug_stride = 0; // For structured buffers, the size of the structure
        unsigned int buffer_PixelDebug_count = 0; // How many items there are
        const D3D12_RESOURCE_STATES c_buffer_PixelDebug_endingState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;

        static const D3D12_RESOURCE_FLAGS c_buffer_PixelDebug_flags =  D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS; // Flags the buffer needs to have been created with

        ID3D12Resource* texture_LinearDepth = nullptr;
        unsigned int texture_LinearDepth_size[3] = { 0, 0, 0 };
        unsigned int texture_LinearDepth_numMips = 0;
        DXGI_FORMAT texture_LinearDepth_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture_LinearDepth_flags =  D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        const D3D12_RESOURCE_STATES c_texture_LinearDepth_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture_DebugTex = nullptr;
        unsigned int texture_DebugTex_size[3] = { 0, 0, 0 };
        unsigned int texture_DebugTex_numMips = 0;
        DXGI_FORMAT texture_DebugTex_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture_DebugTex_flags =  D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        const D3D12_RESOURCE_STATES c_texture_DebugTex_endingState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;

        ID3D12Resource* texture_GatherDOF_FarFieldColorCoC = nullptr;
        unsigned int texture_GatherDOF_FarFieldColorCoC_size[3] = { 0, 0, 0 };
        unsigned int texture_GatherDOF_FarFieldColorCoC_numMips = 0;
        DXGI_FORMAT texture_GatherDOF_FarFieldColorCoC_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture_GatherDOF_FarFieldColorCoC_flags =  D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        const D3D12_RESOURCE_STATES c_texture_GatherDOF_FarFieldColorCoC_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture_GatherDOF_NearFieldColorCoC = nullptr;
        unsigned int texture_GatherDOF_NearFieldColorCoC_size[3] = { 0, 0, 0 };
        unsigned int texture_GatherDOF_NearFieldColorCoC_numMips = 0;
        DXGI_FORMAT texture_GatherDOF_NearFieldColorCoC_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture_GatherDOF_NearFieldColorCoC_flags =  D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        const D3D12_RESOURCE_STATES c_texture_GatherDOF_NearFieldColorCoC_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture_GatherDOF_NearMaxCocTilemap = nullptr;
        unsigned int texture_GatherDOF_NearMaxCocTilemap_size[3] = { 0, 0, 0 };
        unsigned int texture_GatherDOF_NearMaxCocTilemap_numMips = 0;
        DXGI_FORMAT texture_GatherDOF_NearMaxCocTilemap_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture_GatherDOF_NearMaxCocTilemap_flags =  D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        const D3D12_RESOURCE_STATES c_texture_GatherDOF_NearMaxCocTilemap_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture_GatherDOF_BlurredFarFieldColorAlpha = nullptr;
        unsigned int texture_GatherDOF_BlurredFarFieldColorAlpha_size[3] = { 0, 0, 0 };
        unsigned int texture_GatherDOF_BlurredFarFieldColorAlpha_numMips = 0;
        DXGI_FORMAT texture_GatherDOF_BlurredFarFieldColorAlpha_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture_GatherDOF_BlurredFarFieldColorAlpha_flags =  D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        const D3D12_RESOURCE_STATES c_texture_GatherDOF_BlurredFarFieldColorAlpha_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture_GatherDOF_FloodFilledBlurredFarFieldColorAlpha = nullptr;
        unsigned int texture_GatherDOF_FloodFilledBlurredFarFieldColorAlpha_size[3] = { 0, 0, 0 };
        unsigned int texture_GatherDOF_FloodFilledBlurredFarFieldColorAlpha_numMips = 0;
        DXGI_FORMAT texture_GatherDOF_FloodFilledBlurredFarFieldColorAlpha_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture_GatherDOF_FloodFilledBlurredFarFieldColorAlpha_flags =  D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        const D3D12_RESOURCE_STATES c_texture_GatherDOF_FloodFilledBlurredFarFieldColorAlpha_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture_GatherDOF_NearMaxCocTilemap_1_4 = nullptr;
        unsigned int texture_GatherDOF_NearMaxCocTilemap_1_4_size[3] = { 0, 0, 0 };
        unsigned int texture_GatherDOF_NearMaxCocTilemap_1_4_numMips = 0;
        DXGI_FORMAT texture_GatherDOF_NearMaxCocTilemap_1_4_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture_GatherDOF_NearMaxCocTilemap_1_4_flags =  D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        const D3D12_RESOURCE_STATES c_texture_GatherDOF_NearMaxCocTilemap_1_4_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture_GatherDOF_NearMaxCocTilemap_1_8 = nullptr;
        unsigned int texture_GatherDOF_NearMaxCocTilemap_1_8_size[3] = { 0, 0, 0 };
        unsigned int texture_GatherDOF_NearMaxCocTilemap_1_8_numMips = 0;
        DXGI_FORMAT texture_GatherDOF_NearMaxCocTilemap_1_8_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture_GatherDOF_NearMaxCocTilemap_1_8_flags =  D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        const D3D12_RESOURCE_STATES c_texture_GatherDOF_NearMaxCocTilemap_1_8_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture_GatherDOF_NearMaxCocTilemap_1_8_Halo = nullptr;
        unsigned int texture_GatherDOF_NearMaxCocTilemap_1_8_Halo_size[3] = { 0, 0, 0 };
        unsigned int texture_GatherDOF_NearMaxCocTilemap_1_8_Halo_numMips = 0;
        DXGI_FORMAT texture_GatherDOF_NearMaxCocTilemap_1_8_Halo_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture_GatherDOF_NearMaxCocTilemap_1_8_Halo_flags =  D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        const D3D12_RESOURCE_STATES c_texture_GatherDOF_NearMaxCocTilemap_1_8_Halo_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture_GatherDOF_NearFieldColorCoCBorder = nullptr;
        unsigned int texture_GatherDOF_NearFieldColorCoCBorder_size[3] = { 0, 0, 0 };
        unsigned int texture_GatherDOF_NearFieldColorCoCBorder_numMips = 0;
        DXGI_FORMAT texture_GatherDOF_NearFieldColorCoCBorder_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture_GatherDOF_NearFieldColorCoCBorder_flags =  D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        const D3D12_RESOURCE_STATES c_texture_GatherDOF_NearFieldColorCoCBorder_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture_GatherDOF_NearFieldColorCoCBorderBlurred = nullptr;
        unsigned int texture_GatherDOF_NearFieldColorCoCBorderBlurred_size[3] = { 0, 0, 0 };
        unsigned int texture_GatherDOF_NearFieldColorCoCBorderBlurred_numMips = 0;
        DXGI_FORMAT texture_GatherDOF_NearFieldColorCoCBorderBlurred_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture_GatherDOF_NearFieldColorCoCBorderBlurred_flags =  D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        const D3D12_RESOURCE_STATES c_texture_GatherDOF_NearFieldColorCoCBorderBlurred_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture_GatherDOF_FloodFilledBlurredNearFieldColorAlpha = nullptr;
        unsigned int texture_GatherDOF_FloodFilledBlurredNearFieldColorAlpha_size[3] = { 0, 0, 0 };
        unsigned int texture_GatherDOF_FloodFilledBlurredNearFieldColorAlpha_numMips = 0;
        DXGI_FORMAT texture_GatherDOF_FloodFilledBlurredNearFieldColorAlpha_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture_GatherDOF_FloodFilledBlurredNearFieldColorAlpha_flags =  D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        const D3D12_RESOURCE_STATES c_texture_GatherDOF_FloodFilledBlurredNearFieldColorAlpha_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_0 = nullptr;
        unsigned int texture__loadedTexture_0_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_0_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_0_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_0_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_0_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_1 = nullptr;
        unsigned int texture__loadedTexture_1_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_1_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_1_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_1_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_1_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_2 = nullptr;
        unsigned int texture__loadedTexture_2_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_2_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_2_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_2_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_2_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_3 = nullptr;
        unsigned int texture__loadedTexture_3_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_3_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_3_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_3_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_3_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_4 = nullptr;
        unsigned int texture__loadedTexture_4_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_4_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_4_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_4_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_4_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_5 = nullptr;
        unsigned int texture__loadedTexture_5_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_5_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_5_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_5_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_5_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_6 = nullptr;
        unsigned int texture__loadedTexture_6_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_6_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_6_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_6_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_6_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_7 = nullptr;
        unsigned int texture__loadedTexture_7_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_7_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_7_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_7_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_7_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_8 = nullptr;
        unsigned int texture__loadedTexture_8_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_8_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_8_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_8_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_8_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_9 = nullptr;
        unsigned int texture__loadedTexture_9_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_9_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_9_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_9_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_9_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_10 = nullptr;
        unsigned int texture__loadedTexture_10_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_10_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_10_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_10_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_10_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_11 = nullptr;
        unsigned int texture__loadedTexture_11_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_11_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_11_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_11_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_11_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_12 = nullptr;
        unsigned int texture__loadedTexture_12_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_12_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_12_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_12_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_12_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_13 = nullptr;
        unsigned int texture__loadedTexture_13_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_13_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_13_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_13_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_13_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_14 = nullptr;
        unsigned int texture__loadedTexture_14_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_14_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_14_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_14_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_14_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_15 = nullptr;
        unsigned int texture__loadedTexture_15_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_15_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_15_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_15_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_15_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_16 = nullptr;
        unsigned int texture__loadedTexture_16_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_16_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_16_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_16_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_16_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_17 = nullptr;
        unsigned int texture__loadedTexture_17_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_17_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_17_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_17_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_17_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_18 = nullptr;
        unsigned int texture__loadedTexture_18_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_18_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_18_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_18_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_18_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_19 = nullptr;
        unsigned int texture__loadedTexture_19_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_19_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_19_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_19_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_19_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_20 = nullptr;
        unsigned int texture__loadedTexture_20_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_20_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_20_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_20_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_20_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_21 = nullptr;
        unsigned int texture__loadedTexture_21_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_21_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_21_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_21_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_21_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_22 = nullptr;
        unsigned int texture__loadedTexture_22_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_22_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_22_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_22_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_22_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_23 = nullptr;
        unsigned int texture__loadedTexture_23_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_23_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_23_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_23_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_23_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_24 = nullptr;
        unsigned int texture__loadedTexture_24_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_24_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_24_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_24_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_24_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_25 = nullptr;
        unsigned int texture__loadedTexture_25_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_25_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_25_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_25_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_25_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_26 = nullptr;
        unsigned int texture__loadedTexture_26_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_26_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_26_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_26_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_26_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_27 = nullptr;
        unsigned int texture__loadedTexture_27_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_27_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_27_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_27_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_27_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_28 = nullptr;
        unsigned int texture__loadedTexture_28_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_28_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_28_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_28_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_28_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_29 = nullptr;
        unsigned int texture__loadedTexture_29_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_29_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_29_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_29_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_29_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_30 = nullptr;
        unsigned int texture__loadedTexture_30_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_30_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_30_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_30_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_30_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_31 = nullptr;
        unsigned int texture__loadedTexture_31_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_31_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_31_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_31_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_31_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_32 = nullptr;
        unsigned int texture__loadedTexture_32_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_32_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_32_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_32_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_32_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_33 = nullptr;
        unsigned int texture__loadedTexture_33_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_33_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_33_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_33_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_33_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_34 = nullptr;
        unsigned int texture__loadedTexture_34_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_34_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_34_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_34_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_34_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_35 = nullptr;
        unsigned int texture__loadedTexture_35_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_35_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_35_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_35_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_35_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_36 = nullptr;
        unsigned int texture__loadedTexture_36_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_36_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_36_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_36_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_36_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_37 = nullptr;
        unsigned int texture__loadedTexture_37_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_37_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_37_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_37_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_37_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_38 = nullptr;
        unsigned int texture__loadedTexture_38_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_38_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_38_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_38_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_38_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_39 = nullptr;
        unsigned int texture__loadedTexture_39_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_39_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_39_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_39_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_39_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_40 = nullptr;
        unsigned int texture__loadedTexture_40_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_40_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_40_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_40_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_40_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_41 = nullptr;
        unsigned int texture__loadedTexture_41_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_41_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_41_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_41_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_41_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_42 = nullptr;
        unsigned int texture__loadedTexture_42_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_42_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_42_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_42_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_42_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_43 = nullptr;
        unsigned int texture__loadedTexture_43_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_43_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_43_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_43_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_43_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_44 = nullptr;
        unsigned int texture__loadedTexture_44_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_44_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_44_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_44_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_44_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_45 = nullptr;
        unsigned int texture__loadedTexture_45_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_45_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_45_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_45_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_45_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_46 = nullptr;
        unsigned int texture__loadedTexture_46_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_46_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_46_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_46_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_46_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_47 = nullptr;
        unsigned int texture__loadedTexture_47_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_47_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_47_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_47_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_47_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_48 = nullptr;
        unsigned int texture__loadedTexture_48_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_48_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_48_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_48_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_48_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_49 = nullptr;
        unsigned int texture__loadedTexture_49_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_49_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_49_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_49_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_49_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_50 = nullptr;
        unsigned int texture__loadedTexture_50_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_50_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_50_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_50_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_50_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_51 = nullptr;
        unsigned int texture__loadedTexture_51_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_51_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_51_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_51_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_51_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_52 = nullptr;
        unsigned int texture__loadedTexture_52_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_52_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_52_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_52_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_52_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_53 = nullptr;
        unsigned int texture__loadedTexture_53_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_53_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_53_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_53_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_53_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_54 = nullptr;
        unsigned int texture__loadedTexture_54_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_54_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_54_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_54_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_54_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_55 = nullptr;
        unsigned int texture__loadedTexture_55_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_55_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_55_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_55_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_55_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_56 = nullptr;
        unsigned int texture__loadedTexture_56_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_56_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_56_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_56_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_56_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_57 = nullptr;
        unsigned int texture__loadedTexture_57_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_57_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_57_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_57_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_57_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_58 = nullptr;
        unsigned int texture__loadedTexture_58_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_58_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_58_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_58_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_58_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_59 = nullptr;
        unsigned int texture__loadedTexture_59_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_59_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_59_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_59_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_59_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_60 = nullptr;
        unsigned int texture__loadedTexture_60_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_60_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_60_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_60_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_60_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_61 = nullptr;
        unsigned int texture__loadedTexture_61_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_61_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_61_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_61_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_61_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_62 = nullptr;
        unsigned int texture__loadedTexture_62_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_62_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_62_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_62_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_62_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_63 = nullptr;
        unsigned int texture__loadedTexture_63_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_63_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_63_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_63_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_63_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_64 = nullptr;
        unsigned int texture__loadedTexture_64_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_64_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_64_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_64_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_64_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_65 = nullptr;
        unsigned int texture__loadedTexture_65_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_65_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_65_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_65_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_65_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_66 = nullptr;
        unsigned int texture__loadedTexture_66_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_66_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_66_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_66_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_66_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_67 = nullptr;
        unsigned int texture__loadedTexture_67_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_67_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_67_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_67_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_67_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_68 = nullptr;
        unsigned int texture__loadedTexture_68_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_68_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_68_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_68_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_68_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_69 = nullptr;
        unsigned int texture__loadedTexture_69_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_69_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_69_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_69_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_69_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_70 = nullptr;
        unsigned int texture__loadedTexture_70_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_70_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_70_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_70_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_70_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_71 = nullptr;
        unsigned int texture__loadedTexture_71_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_71_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_71_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_71_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_71_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_72 = nullptr;
        unsigned int texture__loadedTexture_72_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_72_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_72_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_72_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_72_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_73 = nullptr;
        unsigned int texture__loadedTexture_73_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_73_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_73_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_73_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_73_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_74 = nullptr;
        unsigned int texture__loadedTexture_74_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_74_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_74_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_74_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_74_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_75 = nullptr;
        unsigned int texture__loadedTexture_75_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_75_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_75_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_75_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_75_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_76 = nullptr;
        unsigned int texture__loadedTexture_76_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_76_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_76_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_76_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_76_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_77 = nullptr;
        unsigned int texture__loadedTexture_77_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_77_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_77_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_77_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_77_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_78 = nullptr;
        unsigned int texture__loadedTexture_78_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_78_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_78_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_78_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_78_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_79 = nullptr;
        unsigned int texture__loadedTexture_79_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_79_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_79_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_79_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_79_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_80 = nullptr;
        unsigned int texture__loadedTexture_80_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_80_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_80_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_80_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_80_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_81 = nullptr;
        unsigned int texture__loadedTexture_81_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_81_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_81_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_81_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_81_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_82 = nullptr;
        unsigned int texture__loadedTexture_82_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_82_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_82_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_82_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_82_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_83 = nullptr;
        unsigned int texture__loadedTexture_83_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_83_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_83_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_83_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_83_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_84 = nullptr;
        unsigned int texture__loadedTexture_84_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_84_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_84_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_84_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_84_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_85 = nullptr;
        unsigned int texture__loadedTexture_85_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_85_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_85_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_85_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_85_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_86 = nullptr;
        unsigned int texture__loadedTexture_86_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_86_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_86_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_86_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_86_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_87 = nullptr;
        unsigned int texture__loadedTexture_87_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_87_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_87_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_87_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_87_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_88 = nullptr;
        unsigned int texture__loadedTexture_88_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_88_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_88_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_88_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_88_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_89 = nullptr;
        unsigned int texture__loadedTexture_89_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_89_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_89_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_89_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_89_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_90 = nullptr;
        unsigned int texture__loadedTexture_90_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_90_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_90_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_90_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_90_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_91 = nullptr;
        unsigned int texture__loadedTexture_91_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_91_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_91_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_91_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_91_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_92 = nullptr;
        unsigned int texture__loadedTexture_92_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_92_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_92_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_92_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_92_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_93 = nullptr;
        unsigned int texture__loadedTexture_93_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_93_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_93_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_93_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_93_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_94 = nullptr;
        unsigned int texture__loadedTexture_94_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_94_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_94_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_94_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_94_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_95 = nullptr;
        unsigned int texture__loadedTexture_95_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_95_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_95_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_95_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_95_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_96 = nullptr;
        unsigned int texture__loadedTexture_96_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_96_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_96_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_96_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_96_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_97 = nullptr;
        unsigned int texture__loadedTexture_97_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_97_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_97_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_97_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_97_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_98 = nullptr;
        unsigned int texture__loadedTexture_98_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_98_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_98_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_98_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_98_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_99 = nullptr;
        unsigned int texture__loadedTexture_99_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_99_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_99_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_99_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_99_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_100 = nullptr;
        unsigned int texture__loadedTexture_100_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_100_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_100_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_100_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_100_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_101 = nullptr;
        unsigned int texture__loadedTexture_101_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_101_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_101_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_101_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_101_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_102 = nullptr;
        unsigned int texture__loadedTexture_102_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_102_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_102_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_102_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_102_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_103 = nullptr;
        unsigned int texture__loadedTexture_103_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_103_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_103_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_103_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_103_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_104 = nullptr;
        unsigned int texture__loadedTexture_104_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_104_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_104_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_104_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_104_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_105 = nullptr;
        unsigned int texture__loadedTexture_105_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_105_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_105_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_105_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_105_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_106 = nullptr;
        unsigned int texture__loadedTexture_106_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_106_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_106_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_106_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_106_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_107 = nullptr;
        unsigned int texture__loadedTexture_107_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_107_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_107_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_107_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_107_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_108 = nullptr;
        unsigned int texture__loadedTexture_108_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_108_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_108_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_108_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_108_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_109 = nullptr;
        unsigned int texture__loadedTexture_109_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_109_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_109_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_109_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_109_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_110 = nullptr;
        unsigned int texture__loadedTexture_110_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_110_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_110_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_110_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_110_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_111 = nullptr;
        unsigned int texture__loadedTexture_111_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_111_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_111_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_111_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_111_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_112 = nullptr;
        unsigned int texture__loadedTexture_112_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_112_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_112_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_112_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_112_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_113 = nullptr;
        unsigned int texture__loadedTexture_113_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_113_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_113_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_113_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_113_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_114 = nullptr;
        unsigned int texture__loadedTexture_114_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_114_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_114_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_114_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_114_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_115 = nullptr;
        unsigned int texture__loadedTexture_115_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_115_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_115_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_115_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_115_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_116 = nullptr;
        unsigned int texture__loadedTexture_116_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_116_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_116_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_116_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_116_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_117 = nullptr;
        unsigned int texture__loadedTexture_117_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_117_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_117_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_117_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_117_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_118 = nullptr;
        unsigned int texture__loadedTexture_118_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_118_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_118_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_118_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_118_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_119 = nullptr;
        unsigned int texture__loadedTexture_119_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_119_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_119_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_119_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_119_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_120 = nullptr;
        unsigned int texture__loadedTexture_120_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_120_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_120_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_120_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_120_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_121 = nullptr;
        unsigned int texture__loadedTexture_121_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_121_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_121_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_121_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_121_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_122 = nullptr;
        unsigned int texture__loadedTexture_122_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_122_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_122_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_122_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_122_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_123 = nullptr;
        unsigned int texture__loadedTexture_123_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_123_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_123_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_123_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_123_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_124 = nullptr;
        unsigned int texture__loadedTexture_124_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_124_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_124_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_124_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_124_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_125 = nullptr;
        unsigned int texture__loadedTexture_125_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_125_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_125_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_125_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_125_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_126 = nullptr;
        unsigned int texture__loadedTexture_126_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_126_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_126_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_126_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_126_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_127 = nullptr;
        unsigned int texture__loadedTexture_127_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_127_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_127_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_127_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_127_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_128 = nullptr;
        unsigned int texture__loadedTexture_128_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_128_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_128_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_128_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_128_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_129 = nullptr;
        unsigned int texture__loadedTexture_129_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_129_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_129_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_129_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_129_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_130 = nullptr;
        unsigned int texture__loadedTexture_130_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_130_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_130_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_130_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_130_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_131 = nullptr;
        unsigned int texture__loadedTexture_131_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_131_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_131_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_131_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_131_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_132 = nullptr;
        unsigned int texture__loadedTexture_132_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_132_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_132_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_132_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_132_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_133 = nullptr;
        unsigned int texture__loadedTexture_133_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_133_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_133_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_133_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_133_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_134 = nullptr;
        unsigned int texture__loadedTexture_134_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_134_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_134_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_134_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_134_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_135 = nullptr;
        unsigned int texture__loadedTexture_135_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_135_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_135_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_135_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_135_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_136 = nullptr;
        unsigned int texture__loadedTexture_136_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_136_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_136_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_136_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_136_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_137 = nullptr;
        unsigned int texture__loadedTexture_137_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_137_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_137_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_137_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_137_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_138 = nullptr;
        unsigned int texture__loadedTexture_138_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_138_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_138_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_138_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_138_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_139 = nullptr;
        unsigned int texture__loadedTexture_139_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_139_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_139_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_139_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_139_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_140 = nullptr;
        unsigned int texture__loadedTexture_140_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_140_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_140_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_140_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_140_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_141 = nullptr;
        unsigned int texture__loadedTexture_141_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_141_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_141_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_141_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_141_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_142 = nullptr;
        unsigned int texture__loadedTexture_142_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_142_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_142_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_142_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_142_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_143 = nullptr;
        unsigned int texture__loadedTexture_143_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_143_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_143_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_143_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_143_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_144 = nullptr;
        unsigned int texture__loadedTexture_144_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_144_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_144_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_144_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_144_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_145 = nullptr;
        unsigned int texture__loadedTexture_145_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_145_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_145_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_145_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_145_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_146 = nullptr;
        unsigned int texture__loadedTexture_146_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_146_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_146_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_146_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_146_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_147 = nullptr;
        unsigned int texture__loadedTexture_147_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_147_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_147_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_147_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_147_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_148 = nullptr;
        unsigned int texture__loadedTexture_148_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_148_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_148_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_148_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_148_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_149 = nullptr;
        unsigned int texture__loadedTexture_149_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_149_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_149_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_149_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_149_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_150 = nullptr;
        unsigned int texture__loadedTexture_150_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_150_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_150_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_150_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_150_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_151 = nullptr;
        unsigned int texture__loadedTexture_151_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_151_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_151_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_151_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_151_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_152 = nullptr;
        unsigned int texture__loadedTexture_152_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_152_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_152_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_152_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_152_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_153 = nullptr;
        unsigned int texture__loadedTexture_153_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_153_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_153_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_153_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_153_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_154 = nullptr;
        unsigned int texture__loadedTexture_154_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_154_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_154_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_154_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_154_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_155 = nullptr;
        unsigned int texture__loadedTexture_155_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_155_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_155_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_155_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_155_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_156 = nullptr;
        unsigned int texture__loadedTexture_156_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_156_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_156_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_156_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_156_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_157 = nullptr;
        unsigned int texture__loadedTexture_157_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_157_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_157_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_157_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_157_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_158 = nullptr;
        unsigned int texture__loadedTexture_158_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_158_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_158_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_158_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_158_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_159 = nullptr;
        unsigned int texture__loadedTexture_159_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_159_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_159_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_159_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_159_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_160 = nullptr;
        unsigned int texture__loadedTexture_160_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_160_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_160_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_160_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_160_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_161 = nullptr;
        unsigned int texture__loadedTexture_161_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_161_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_161_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_161_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_161_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_162 = nullptr;
        unsigned int texture__loadedTexture_162_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_162_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_162_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_162_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_162_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_163 = nullptr;
        unsigned int texture__loadedTexture_163_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_163_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_163_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_163_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_163_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_164 = nullptr;
        unsigned int texture__loadedTexture_164_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_164_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_164_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_164_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_164_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_165 = nullptr;
        unsigned int texture__loadedTexture_165_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_165_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_165_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_165_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_165_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_166 = nullptr;
        unsigned int texture__loadedTexture_166_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_166_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_166_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_166_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_166_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_167 = nullptr;
        unsigned int texture__loadedTexture_167_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_167_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_167_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_167_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_167_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_168 = nullptr;
        unsigned int texture__loadedTexture_168_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_168_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_168_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_168_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_168_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_169 = nullptr;
        unsigned int texture__loadedTexture_169_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_169_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_169_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_169_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_169_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_170 = nullptr;
        unsigned int texture__loadedTexture_170_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_170_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_170_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_170_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_170_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_171 = nullptr;
        unsigned int texture__loadedTexture_171_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_171_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_171_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_171_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_171_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_172 = nullptr;
        unsigned int texture__loadedTexture_172_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_172_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_172_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_172_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_172_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_173 = nullptr;
        unsigned int texture__loadedTexture_173_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_173_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_173_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_173_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_173_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_174 = nullptr;
        unsigned int texture__loadedTexture_174_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_174_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_174_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_174_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_174_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_175 = nullptr;
        unsigned int texture__loadedTexture_175_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_175_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_175_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_175_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_175_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_176 = nullptr;
        unsigned int texture__loadedTexture_176_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_176_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_176_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_176_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_176_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_177 = nullptr;
        unsigned int texture__loadedTexture_177_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_177_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_177_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_177_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_177_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_178 = nullptr;
        unsigned int texture__loadedTexture_178_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_178_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_178_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_178_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_178_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_179 = nullptr;
        unsigned int texture__loadedTexture_179_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_179_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_179_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_179_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_179_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_180 = nullptr;
        unsigned int texture__loadedTexture_180_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_180_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_180_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_180_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_180_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_181 = nullptr;
        unsigned int texture__loadedTexture_181_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_181_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_181_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_181_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_181_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_182 = nullptr;
        unsigned int texture__loadedTexture_182_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_182_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_182_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_182_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_182_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_183 = nullptr;
        unsigned int texture__loadedTexture_183_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_183_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_183_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_183_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_183_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_184 = nullptr;
        unsigned int texture__loadedTexture_184_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_184_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_184_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_184_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_184_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_185 = nullptr;
        unsigned int texture__loadedTexture_185_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_185_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_185_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_185_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_185_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_186 = nullptr;
        unsigned int texture__loadedTexture_186_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_186_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_186_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_186_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_186_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_187 = nullptr;
        unsigned int texture__loadedTexture_187_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_187_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_187_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_187_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_187_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_188 = nullptr;
        unsigned int texture__loadedTexture_188_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_188_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_188_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_188_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_188_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_189 = nullptr;
        unsigned int texture__loadedTexture_189_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_189_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_189_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_189_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_189_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_190 = nullptr;
        unsigned int texture__loadedTexture_190_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_190_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_190_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_190_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_190_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_191 = nullptr;
        unsigned int texture__loadedTexture_191_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_191_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_191_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_191_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_191_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_192 = nullptr;
        unsigned int texture__loadedTexture_192_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_192_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_192_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_192_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_192_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_193 = nullptr;
        unsigned int texture__loadedTexture_193_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_193_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_193_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_193_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_193_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_194 = nullptr;
        unsigned int texture__loadedTexture_194_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_194_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_194_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_194_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_194_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_195 = nullptr;
        unsigned int texture__loadedTexture_195_size[3] = { 0, 0, 0 };
        unsigned int texture__loadedTexture_195_numMips = 0;
        DXGI_FORMAT texture__loadedTexture_195_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_195_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_195_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        Struct__RayGenCB constantBuffer__RayGenCB_cpu;
        ID3D12Resource* constantBuffer__RayGenCB = nullptr;

        static ID3D12StateObject* rayShader_Raytrace_rtso;
        static ID3D12RootSignature* rayShader_Raytrace_rootSig;
        static ID3D12Resource* rayShader_Raytrace_shaderTableRayGen;
        static unsigned int    rayShader_Raytrace_shaderTableRayGenSize;
        static ID3D12Resource* rayShader_Raytrace_shaderTableMiss;
        static unsigned int    rayShader_Raytrace_shaderTableMissSize;
        static ID3D12Resource* rayShader_Raytrace_shaderTableHitGroup;
        static unsigned int    rayShader_Raytrace_shaderTableHitGroupSize;

        Struct__GatherDOF_SetupCSCB constantBuffer__GatherDOF_SetupCSCB_cpu;
        ID3D12Resource* constantBuffer__GatherDOF_SetupCSCB = nullptr;

        static ID3D12PipelineState* computeShader_GatherDOF_Setup_pso;
        static ID3D12RootSignature* computeShader_GatherDOF_Setup_rootSig;

        static ID3D12PipelineState* computeShader_GatherDOF_NearBorder_pso;
        static ID3D12RootSignature* computeShader_GatherDOF_NearBorder_rootSig;

        Struct__GatherDOF_BlurFarCSCB constantBuffer__GatherDOF_BlurFarCSCB_cpu;
        ID3D12Resource* constantBuffer__GatherDOF_BlurFarCSCB = nullptr;

        static ID3D12PipelineState* computeShader_GatherDOF_BlurFar_pso;
        static ID3D12RootSignature* computeShader_GatherDOF_BlurFar_rootSig;

        Struct__GatherDOF_NearBlurCB constantBuffer__GatherDOF_NearBlurCB_cpu;
        ID3D12Resource* constantBuffer__GatherDOF_NearBlurCB = nullptr;

        Struct__GatherDOF_RecombineCSCB constantBuffer__GatherDOF_RecombineCSCB_cpu;
        ID3D12Resource* constantBuffer__GatherDOF_RecombineCSCB = nullptr;

        Struct__GaussBlur_GaussBlurCSCB constantBuffer__GaussBlur_GaussBlurCSCB_cpu;
        ID3D12Resource* constantBuffer__GaussBlur_GaussBlurCSCB = nullptr;

        Struct__TemporalAccumulation_AccumulateCB constantBuffer__TemporalAccumulation_AccumulateCB_cpu;
        ID3D12Resource* constantBuffer__TemporalAccumulation_AccumulateCB = nullptr;

        Struct__ToneMap_TonemapCB constantBuffer__ToneMap_TonemapCB_cpu;
        ID3D12Resource* constantBuffer__ToneMap_TonemapCB = nullptr;

        Struct__GatherDOF_FloodFillFarCS_0CB constantBuffer__GatherDOF_FloodFillFarCS_0CB_cpu;
        ID3D12Resource* constantBuffer__GatherDOF_FloodFillFarCS_0CB = nullptr;

        static ID3D12PipelineState* computeShader_GatherDOF_FloodFillFar_pso;
        static ID3D12RootSignature* computeShader_GatherDOF_FloodFillFar_rootSig;

        static ID3D12PipelineState* computeShader_GatherDOF_DownscaleTileMap_1_4_pso;
        static ID3D12RootSignature* computeShader_GatherDOF_DownscaleTileMap_1_4_rootSig;

        static ID3D12PipelineState* computeShader_GatherDOF_DownscaleTileMap_1_8_pso;
        static ID3D12RootSignature* computeShader_GatherDOF_DownscaleTileMap_1_8_rootSig;

        static ID3D12PipelineState* computeShader_GatherDOF_NearHalo_pso;
        static ID3D12RootSignature* computeShader_GatherDOF_NearHalo_rootSig;

        static ID3D12PipelineState* computeShader_GatherDOF_NearBlur_pso;
        static ID3D12RootSignature* computeShader_GatherDOF_NearBlur_rootSig;

        Struct__GatherDOF_FloodFillFarCS_1CB constantBuffer__GatherDOF_FloodFillFarCS_1CB_cpu;
        ID3D12Resource* constantBuffer__GatherDOF_FloodFillFarCS_1CB = nullptr;

        static ID3D12PipelineState* computeShader_GatherDOF_FloodFillNear_pso;
        static ID3D12RootSignature* computeShader_GatherDOF_FloodFillNear_rootSig;

        static ID3D12PipelineState* computeShader_GatherDOF_Recombine_pso;
        static ID3D12RootSignature* computeShader_GatherDOF_Recombine_rootSig;

        static ID3D12PipelineState* computeShader_GaussBlur_DoBlur_pso;
        static ID3D12RootSignature* computeShader_GaussBlur_DoBlur_rootSig;

        static ID3D12PipelineState* computeShader_TemporalAccumulation_DoAccum_pso;
        static ID3D12RootSignature* computeShader_TemporalAccumulation_DoAccum_rootSig;

        static ID3D12PipelineState* computeShader_ToneMap_Tonemap_pso;
        static ID3D12RootSignature* computeShader_ToneMap_Tonemap_rootSig;

        std::unordered_map<DX12Utils::SubResourceHeapAllocationInfo, int, DX12Utils::SubResourceHeapAllocationInfo> m_RTVCache;
        std::unordered_map<DX12Utils::SubResourceHeapAllocationInfo, int, DX12Utils::SubResourceHeapAllocationInfo> m_DSVCache;

        // Freed on destruction of the context
        std::vector<ID3D12Resource*> m_managedResources;
    };
};
