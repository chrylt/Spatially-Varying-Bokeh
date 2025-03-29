///////////////////////////////////////////////////////////////////////////////
//     Filter-Adapted Spatio-Temporal Sampling With General Distributions    //
//        Copyright (c) 2024 Electronic Arts Inc. All rights reserved.       //
///////////////////////////////////////////////////////////////////////////////

// Unnamed technique, shader BlurFarCS


struct LensRNG
{
    static const int UniformCircleWhite_PCG = 0;
    static const int UniformCircleWhite = 1;
    static const int UniformCircleBlue = 2;
    static const int UniformHexagonWhite = 3;
    static const int UniformHexagonBlue = 4;
    static const int UniformHexagonICDF_White = 5;
    static const int UniformHexagonICDF_Blue = 6;
    static const int UniformStarWhite = 7;
    static const int UniformStarBlue = 8;
    static const int UniformStarICDF_White = 9;
    static const int UniformStarICDF_Blue = 10;
    static const int NonUniformStarWhite = 11;
    static const int NonUniformStarBlue = 12;
    static const int NonUniformStar2White = 13;
    static const int NonUniformStar2Blue = 14;
    static const int LKCP6White = 15;
    static const int LKCP6Blue = 16;
    static const int LKCP204White = 17;
    static const int LKCP204Blue = 18;
    static const int LKCP204ICDF_White = 19;
    static const int LKCP204ICDF_Blue = 20;
};

struct NoiseTexExtends
{
    static const int None = 0;
    static const int White = 1;
    static const int Shuffle1D = 2;
    static const int Shuffle1DHilbert = 3;
};

struct Struct__GatherDOF_BlurFarCSCB
{
    uint GatherDOF_AnimateNoiseTextures;
    uint GatherDOF_BlurTapCount;
    uint GatherDOF_FrameIndex;
    float _padding0;
    float4 GatherDOF_KernelSize;
    uint GatherDOF_UseNoiseTextures;
    uint JitterNoiseTextures;
    int LensRNGExtend;
    int LensRNGSource;
};

SamplerState linearClampSampler : register(s0);
Texture2D<float4> FarFieldColorCoC : register(t0);
RWTexture2D<float4> BlurredFarFieldColorAlpha : register(u0);
Texture2DArray<float2> _loadedTexture_179 : register(t1);
Texture2DArray<float2> _loadedTexture_180 : register(t2);
Texture2DArray<float2> _loadedTexture_181 : register(t3);
Texture2DArray<float2> _loadedTexture_182 : register(t4);
Texture2D<float> _loadedTexture_183 : register(t5);
Texture2DArray<float2> _loadedTexture_171 : register(t6);
Texture2DArray<float2> _loadedTexture_184 : register(t7);
Texture2DArray<float2> _loadedTexture_185 : register(t8);
Texture2DArray<float2> _loadedTexture_187 : register(t9);
Texture2DArray<float2> _loadedTexture_188 : register(t10);
Texture2D<float> _loadedTexture_186 : register(t11);
Texture2DArray<float2> _loadedTexture_189 : register(t12);
Texture2DArray<float2> _loadedTexture_190 : register(t13);
Texture2DArray<float2> _loadedTexture_191 : register(t14);
Texture2DArray<float2> _loadedTexture_192 : register(t15);
Texture2DArray<float2> _loadedTexture_193 : register(t16);
Texture2DArray<float2> _loadedTexture_194 : register(t17);
Texture2D<float> _loadedTexture_195 : register(t18);
ConstantBuffer<Struct__GatherDOF_BlurFarCSCB> _GatherDOF_BlurFarCSCB : register(b0);

#line 7


#include "Common.hlsli"
#include "PCG.hlsli"
#include "LDSShuffler.hlsli"

float2 ReadVec2STTextureRaw(in uint3 pxAndFrame, in Texture2DArray<float2> tex)
{
    uint3 dims;
    tex.GetDimensions(dims.x, dims.y, dims.z);

    // Extend the noise texture over time
	uint cycleCount = pxAndFrame.z / dims.z;
	switch(_GatherDOF_BlurFarCSCB.LensRNGExtend)
	{
		case NoiseTexExtends::None: break;
		case NoiseTexExtends::White:
		{
			uint OffsetRNG = HashInit(uint3(0x1337, 0xbeef, cycleCount));
			pxAndFrame.x += HashPCG(OffsetRNG);
			pxAndFrame.y += HashPCG(OffsetRNG);
			break;
		}
		case NoiseTexExtends::Shuffle1D:
		{
			uint shuffleIndex = LDSShuffle1D_GetValueAtIndex(cycleCount, 16384, 10127, 435);
			pxAndFrame.x += shuffleIndex % dims.x;
			pxAndFrame.y += shuffleIndex / dims.x;
			break;
		}
		case NoiseTexExtends::Shuffle1DHilbert:
		{
			uint shuffleIndex = LDSShuffle1D_GetValueAtIndex(cycleCount, 16384, 10127, 435);
			pxAndFrame.xy += Convert1DTo2D_Hilbert(shuffleIndex, 16384);
			break;
		}
	}

    return tex[pxAndFrame % dims].rg;
}

float2 ReadVec2STTexture(in uint3 pxAndFrame, in Texture2DArray<float2> tex)
{
	uint jitterRNG = HashInit(pxAndFrame);

    float2 ret = ReadVec2STTextureRaw(pxAndFrame, tex) * 2.0f - 1.0f;

	if (_GatherDOF_BlurFarCSCB.JitterNoiseTextures)
		ret += float2((RandomFloat01(jitterRNG) - 0.5f) / 255.0f, (RandomFloat01(jitterRNG) - 0.5f) / 255.0f);

	return ret;
}

float2 SampleICDF(float2 rng, in Texture2D<float> MarginalCDF)
{
    rng = clamp(rng, 0.001f, 0.999f);

    // Get the dimensions of the MarginalCDF
    uint2 MarginalCDFDims;
    MarginalCDF.GetDimensions(MarginalCDFDims.x, MarginalCDFDims.y);

    // Find the column of pixels we want to sample from by doing a branchless binary search.  This is our x axis.
    uint2 samplePos = uint2(0, 0);
    {
        uint numColumns = MarginalCDFDims.x;
        uint pow2Size = (uint)1 << uint(ceil(log2(numColumns)));

        // Do first step manually to handle the possibility of non power of 2
        uint searchSize = pow2Size / 2;
        uint ret = 0;
        ret = (MarginalCDF[uint2(ret + searchSize, 0)] <= rng.x) * (numColumns - searchSize);
        searchSize /= 2;

        // Do the rest of the steps
        while (searchSize > 0)
        {
            ret += (MarginalCDF[uint2(ret + searchSize, 0)] <= rng.x) * searchSize;
            searchSize /= 2;
        }
        samplePos.x = ret;
    }

    // Find the row we want to sample, in that column, by doing a branchless binary search. This is our y axis.
    {
        uint numRows = MarginalCDFDims.y - 1;
        uint pow2Size = (uint)1 << uint(ceil(log2(numRows)));

        // Do first step manually to handle the possibility of non power of 2
        uint searchSize = pow2Size / 2;
        uint ret = 0;
        ret = (MarginalCDF[uint2(samplePos.x, ret + searchSize + 1)] <= rng.y) * (numRows - searchSize);
        searchSize /= 2;

        // Do the rest of the steps
        while (searchSize > 0)
        {
            ret += (MarginalCDF[uint2(samplePos.x, ret + searchSize + 1)] <= rng.y) * searchSize;
            searchSize /= 2;
        }
        samplePos.y = ret;
    }

    // Convert to [-1,+1]^2 coordinates
    float2 uv = (float2(samplePos) + float2(0.5f, 0.5f)) / float2(MarginalCDFDims);
    return uv * 2.0f - 1.0f;
}

float2 GetApertureSamplePoint(uint3 pxAndFrame, int u, int v, int maxuv, in float4 KernelSize)
{
	if (!_GatherDOF_BlurFarCSCB.GatherDOF_UseNoiseTextures)
	{
		float2 uv = float2(u, v) / (maxuv.xx - 1); // map to [0, 1]
		return SquareToPolygonMapping( uv, KernelSize );
	}

	// calculate what sample index we are on
	uint sampleIndex = pxAndFrame.z * maxuv * maxuv;
	sampleIndex += v * maxuv + u;
	uint3 pxAndSampleIndex = uint3(pxAndFrame.xy, sampleIndex);

	float2 offset = float2(0.0f, 0.0f);
	switch(_GatherDOF_BlurFarCSCB.LensRNGSource)
	{
		case LensRNG::UniformCircleWhite_PCG:
		{
			uint RNG = HashInit(pxAndSampleIndex);
			float angle = RandomFloat01(RNG) * 2.0f * c_pi;
			float radius = sqrt(RandomFloat01(RNG));
			return float2(cos(angle), sin(angle)) * radius;
		}
		case LensRNG::UniformCircleWhite:
		{
			return ReadVec2STTexture(pxAndSampleIndex, _loadedTexture_179);
		}
		case LensRNG::UniformCircleBlue:
		{
			return ReadVec2STTexture(pxAndSampleIndex, _loadedTexture_180);
		}
		case LensRNG::UniformHexagonWhite:
		{
			return ReadVec2STTexture(pxAndSampleIndex, _loadedTexture_181);
		}
		case LensRNG::UniformHexagonBlue:
		{
			return ReadVec2STTexture(pxAndSampleIndex, _loadedTexture_182);
        }
        case LensRNG::UniformHexagonICDF_White:
        {
            uint RNG = HashInit(pxAndSampleIndex);
            float2 rng = float2(RandomFloat01(RNG), RandomFloat01(RNG));
            return SampleICDF(rng, _loadedTexture_183);
        }
        case LensRNG::UniformHexagonICDF_Blue:
        {
            float2 rng = ReadVec2STTextureRaw(pxAndSampleIndex, _loadedTexture_171);
            return SampleICDF(rng, _loadedTexture_183);
        }
		case LensRNG::UniformStarWhite:
		{
			return ReadVec2STTexture(pxAndSampleIndex, _loadedTexture_184);
		}
		case LensRNG::UniformStarBlue:
		{
			return ReadVec2STTexture(pxAndSampleIndex, _loadedTexture_185);
		}
		case LensRNG::NonUniformStarWhite:
		{
			return ReadVec2STTexture(pxAndSampleIndex, _loadedTexture_187);
		}
		case LensRNG::NonUniformStarBlue:
		{
			return ReadVec2STTexture(pxAndSampleIndex, _loadedTexture_188);
        }
        case LensRNG::UniformStarICDF_White:
        {
            uint RNG = HashInit(pxAndSampleIndex);
            float2 rng = float2(RandomFloat01(RNG), RandomFloat01(RNG));
            return SampleICDF(rng, _loadedTexture_186);
        }
        case LensRNG::UniformStarICDF_Blue:
        {
            float2 rng = ReadVec2STTextureRaw(pxAndSampleIndex, _loadedTexture_171);
            return SampleICDF(rng, _loadedTexture_186);
        }
		case LensRNG::NonUniformStar2White:
		{
			return ReadVec2STTexture(pxAndSampleIndex, _loadedTexture_189);
		}
		case LensRNG::NonUniformStar2Blue:
		{
			return ReadVec2STTexture(pxAndSampleIndex, _loadedTexture_190);
		}
		case LensRNG::LKCP6White:
		{
			return ReadVec2STTexture(pxAndSampleIndex, _loadedTexture_191);
		}
		case LensRNG::LKCP6Blue:
		{
			return ReadVec2STTexture(pxAndSampleIndex, _loadedTexture_192);
		}
		case LensRNG::LKCP204White:
		{
			return ReadVec2STTexture(pxAndSampleIndex, _loadedTexture_193);
		}
		case LensRNG::LKCP204Blue:
		{
			return ReadVec2STTexture(pxAndSampleIndex, _loadedTexture_194);
        }
        case LensRNG::LKCP204ICDF_White:
        {
            uint RNG = HashInit(pxAndSampleIndex);
            float2 rng = float2(RandomFloat01(RNG), RandomFloat01(RNG));
            return SampleICDF(rng, _loadedTexture_195);
        }
        case LensRNG::LKCP204ICDF_Blue:
        {
            float2 rng = ReadVec2STTextureRaw(pxAndSampleIndex, _loadedTexture_171);
            return SampleICDF(rng, _loadedTexture_195);
        }
	}

	return float2(0.0f, 0.0f);
}

#define BLUR_TAP_COUNT _GatherDOF_BlurFarCSCB.GatherDOF_BlurTapCount

// .x : size of the bokeh blur radius in texel space
// .y : rotation in radius to apply to the bokeh shape
// .z : Number of edge of the polygon (number of blades). 0: circle. 4: square, 6: hexagon...
#define KernelSize _GatherDOF_BlurFarCSCB.GatherDOF_KernelSize

[numthreads(8, 8, 1)]
#line 238
void csmain(uint3 DTid : SV_DispatchThreadID)
{
	uint2 px = DTid.xy;

	uint2 FarFieldColorCoCSize;
	FarFieldColorCoC.GetDimensions(FarFieldColorCoCSize.x, FarFieldColorCoCSize.y);

	float2 UVAndScreenPos = (float2(px) + float2(0.5f, 0.5f)) / float2(FarFieldColorCoCSize);

	float4 PixelColor = FarFieldColorCoC[px];
	float PixelCoC = PixelColor.w;

	float3 ResultColor = 0;
	float Weight = 0;
	
	int TAP_COUNT = BLUR_TAP_COUNT; // Higher means less noise and make floodfilling easier after

	// Multiplying by PixelCoC guarantees a smooth evolution of the blur radius
	// especially visible on plane (like the floor) where CoC slowly grows with the distance.
	// This makes all the difference between a natural bokeh and some noticeable
	// in-focus and out-of-focus layer blending
	float radius = KernelSize.x * PixelCoC;

	uint noiseTextureFrameIndex = _GatherDOF_BlurFarCSCB.GatherDOF_AnimateNoiseTextures ? _GatherDOF_BlurFarCSCB.GatherDOF_FrameIndex : 0;
	uint3 pxAndFrame = uint3(px, noiseTextureFrameIndex);

	if (PixelCoC > 0) { // Ignore any pixel not belonging to far field
	
		// Weighted average of the texture samples inside the bokeh pattern
		// High radius and low sample count can create "gaps" which are fixed later (floodfill).
		for (int u = 0; u < TAP_COUNT; ++u)
		{
			for (int v = 0; v < TAP_COUNT; ++v)
			{
				float2 uv = GetApertureSamplePoint(pxAndFrame, u, v, TAP_COUNT, KernelSize);
				uv /= float2(FarFieldColorCoCSize);

				//float2 uv = float2(u, v) / (TAP_COUNT - 1); // map to [0, 1]
				//uv = SquareToPolygonMapping( uv, KernelSize ) / float2(FarFieldColorCoCSize); // map to bokeh shape, then to texel size
				uv = UVAndScreenPos.xy + radius * uv;

				float4 tapColor = FarFieldColorCoC.SampleLevel(linearClampSampler, uv, 0); //Texture2DSampleLevel(PostprocessInput0, PostprocessInput0Sampler, uv, 0);
				
				// Weighted by CoC. Gives more influence to taps with a CoC higher than us.
				float TapWeight = tapColor.w * saturate(1.0f - (PixelCoC - tapColor.w)); 
				
				ResultColor +=  tapColor.xyz * TapWeight; 
				Weight += TapWeight;
			}
		}
		if (Weight > 0) ResultColor /= Weight;
		Weight = Weight / TAP_COUNT / TAP_COUNT;
	}
	
	Weight = saturate(Weight * 10); // From CoC 0.1, completely rely on the far field layer and stop lerping with in-focus layer
	float4 OutColor = float4(ResultColor, Weight);

	BlurredFarFieldColorAlpha[px] = OutColor;
}

/*
Shader Resources:
	Texture FarFieldColorCoC (as SRV)
	Texture BlurredFarFieldColorAlpha (as UAV)
*/
