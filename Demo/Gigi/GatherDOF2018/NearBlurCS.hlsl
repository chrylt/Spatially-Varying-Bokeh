///////////////////////////////////////////////////////////////////////////////
//     Filter-Adapted Spatio-Temporal Sampling With General Distributions    //
//        Copyright (c) 2024 Electronic Arts Inc. All rights reserved.       //
///////////////////////////////////////////////////////////////////////////////

// Unnamed technique, shader NearBlur
/*$(ShaderResources)*/

#include "Common.hlsli"
#include "PCG.hlsli"
#include "LDSShuffler.hlsli"

float2 ReadVec2STTextureRaw(in uint3 pxAndFrame, in Texture2DArray<float2> tex)
{
    uint3 dims;
    tex.GetDimensions(dims.x, dims.y, dims.z);

    // Extend the noise texture over time
	uint cycleCount = pxAndFrame.z / dims.z;
	switch(/*$(Variable:LensRNGExtend)*/)
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

	if (/*$(Variable:JitterNoiseTextures)*/)
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
	if (!/*$(Variable:UseNoiseTextures)*/)
	{
		float2 uv = float2(u, v) / (maxuv.xx - 1); // map to [0, 1]
		return SquareToPolygonMapping( uv, KernelSize );
	}

	// calculate what sample index we are on
	uint sampleIndex = pxAndFrame.z * maxuv * maxuv;
	sampleIndex += v * maxuv + u;
	uint3 pxAndSampleIndex = uint3(pxAndFrame.xy, sampleIndex);

	float2 offset = float2(0.0f, 0.0f);
	switch(/*$(Variable:LensRNGSource)*/)
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
			return ReadVec2STTexture(pxAndSampleIndex, /*$(Image2DArray:Assets\NoiseTextures\UniformCircle\UniformCircle_%i.0.png:RG8_UNorm:float2:false:false)*/);
		}
		case LensRNG::UniformCircleBlue:
		{
			return ReadVec2STTexture(pxAndSampleIndex, /*$(Image2DArray:Assets\NoiseTextures\UniformCircle\UniformCircle_%i.png:RG8_UNorm:float2:false:false)*/);
		}
		case LensRNG::UniformHexagonWhite:
		{
			return ReadVec2STTexture(pxAndSampleIndex, /*$(Image2DArray:Assets\NoiseTextures\UniformHexagon\UniformHexagon_%i.0.png:RG8_UNorm:float2:false:false)*/);
		}
		case LensRNG::UniformHexagonBlue:
		{
			return ReadVec2STTexture(pxAndSampleIndex, /*$(Image2DArray:Assets\NoiseTextures\UniformHexagon\UniformHexagon_%i.png:RG8_UNorm:float2:false:false)*/);
        }
        case LensRNG::UniformHexagonICDF_White:
        {
            uint RNG = HashInit(pxAndSampleIndex);
            float2 rng = float2(RandomFloat01(RNG), RandomFloat01(RNG));
            return SampleICDF(rng, /*$(Image2D:Assets\NoiseTextures\UniformHexagon\UniformHexagon.icdf.exr:R32_Float:float:false:false)*/);
        }
        case LensRNG::UniformHexagonICDF_Blue:
        {
            float2 rng = ReadVec2STTextureRaw(pxAndSampleIndex, /*$(Image2DArray:Assets\NoiseTextures\FAST\vector2_uniform_gauss1_0_Gauss10_separate05_%i.png:RG8_UNorm:float2:false:false)*/);
            return SampleICDF(rng, /*$(Image2D:Assets\NoiseTextures\UniformHexagon\UniformHexagon.icdf.exr:R32_Float:float:false:false)*/);
        }
		case LensRNG::UniformStarWhite:
		{
			return ReadVec2STTexture(pxAndSampleIndex, /*$(Image2DArray:Assets\NoiseTextures\UniformStar\UniformStar_%i.0.png:RG8_UNorm:float2:false:false)*/);
		}
		case LensRNG::UniformStarBlue:
		{
			return ReadVec2STTexture(pxAndSampleIndex, /*$(Image2DArray:Assets\NoiseTextures\UniformStar\UniformStar_%i.png:RG8_UNorm:float2:false:false)*/);
        }
        case LensRNG::UniformStarICDF_White:
        {
            uint RNG = HashInit(pxAndSampleIndex);
            float2 rng = float2(RandomFloat01(RNG), RandomFloat01(RNG));
            return SampleICDF(rng, /*$(Image2D:Assets\NoiseTextures\UniformStar\UniformStar.icdf.exr:R32_Float:float:false:false)*/);
        }
        case LensRNG::UniformStarICDF_Blue:
        {
            float2 rng = ReadVec2STTextureRaw(pxAndSampleIndex, /*$(Image2DArray:Assets\NoiseTextures\FAST\vector2_uniform_gauss1_0_Gauss10_separate05_%i.png:RG8_UNorm:float2:false:false)*/);
            return SampleICDF(rng, /*$(Image2D:Assets\NoiseTextures\UniformStar\UniformStar.icdf.exr:R32_Float:float:false:false)*/);
        }
		case LensRNG::NonUniformStarWhite:
		{
			return ReadVec2STTexture(pxAndSampleIndex, /*$(Image2DArray:Assets\NoiseTextures\NonUniformStar\NonUniformStar_%i.0.png:RG8_UNorm:float2:false:false)*/);
		}
		case LensRNG::NonUniformStarBlue:
		{
			return ReadVec2STTexture(pxAndSampleIndex, /*$(Image2DArray:Assets\NoiseTextures\NonUniformStar\NonUniformStar_%i.png:RG8_UNorm:float2:false:false)*/);
		}
		case LensRNG::NonUniformStar2White:
		{
			return ReadVec2STTexture(pxAndSampleIndex, /*$(Image2DArray:Assets\NoiseTextures\NonUniformStar2\NonUniformStar2_%i.0.png:RG8_UNorm:float2:false:false)*/);
		}
		case LensRNG::NonUniformStar2Blue:
		{
			return ReadVec2STTexture(pxAndSampleIndex, /*$(Image2DArray:Assets\NoiseTextures\NonUniformStar2\NonUniformStar2_%i.png:RG8_UNorm:float2:false:false)*/);
		}
		case LensRNG::LKCP6White:
		{
			return ReadVec2STTexture(pxAndSampleIndex, /*$(Image2DArray:Assets\NoiseTextures\Lens_kernel_compositingpro.006\Lens_kernel_compositingpro.006_%i.0.png:RG8_UNorm:float2:false:false)*/);
		}
		case LensRNG::LKCP6Blue:
		{
			return ReadVec2STTexture(pxAndSampleIndex, /*$(Image2DArray:Assets\NoiseTextures\Lens_kernel_compositingpro.006\Lens_kernel_compositingpro.006_%i.png:RG8_UNorm:float2:false:false)*/);
		}
		case LensRNG::LKCP204White:
		{
			return ReadVec2STTexture(pxAndSampleIndex, /*$(Image2DArray:Assets\NoiseTextures\Lens_kernel_compositingpro.204\Lens_kernel_compositingpro.204_%i.0.png:RG8_UNorm:float2:false:false)*/);
		}
		case LensRNG::LKCP204Blue:
		{
			return ReadVec2STTexture(pxAndSampleIndex, /*$(Image2DArray:Assets\NoiseTextures\Lens_kernel_compositingpro.204\Lens_kernel_compositingpro.204_%i.png:RG8_UNorm:float2:false:false)*/);
        }
        case LensRNG::LKCP204ICDF_White:
        {
            uint RNG = HashInit(pxAndSampleIndex);
            float2 rng = float2(RandomFloat01(RNG), RandomFloat01(RNG));
            return SampleICDF(rng, /*$(Image2D:Assets\NoiseTextures\Lens_kernel_compositingpro.204\Lens_kernel_compositingpro.204.icdf.exr:R32_Float:float:false:false)*/);
        }
        case LensRNG::LKCP204ICDF_Blue:
        {
            float2 rng = ReadVec2STTextureRaw(pxAndSampleIndex, /*$(Image2DArray:Assets\NoiseTextures\FAST\vector2_uniform_gauss1_0_Gauss10_separate05_%i.png:RG8_UNorm:float2:false:false)*/);
            return SampleICDF(rng, /*$(Image2D:Assets\NoiseTextures\Lens_kernel_compositingpro.204\Lens_kernel_compositingpro.204.icdf.exr:R32_Float:float:false:false)*/);
        }
	}

	return float2(0.0f, 0.0f);
}

#define BLUR_TAP_COUNT /*$(Variable:BlurTapCount)*/

// .x : size of the bokeh blur radius in texel space
// .y : rotation in radius to apply to the bokeh shape
// .z : Number of edge of the polygon (number of blades). 0: circle. 4: square, 6: hexagon...
#define KernelSize /*$(Variable:KernelSize)*/

/*$(_compute:csmain)*/(uint3 DTid : SV_DispatchThreadID)
{
	uint2 px = DTid.xy;

	uint2 NearFieldColorCoCBorderSize;
	NearFieldColorCoCBorder.GetDimensions(NearFieldColorCoCBorderSize.x, NearFieldColorCoCBorderSize.y);

	float2 UVAndScreenPos = (float2(px) + float2(0.5f, 0.5f)) / float2(NearFieldColorCoCBorderSize);

	float4 PixelColor = NearFieldColorCoCBorder[px];// Texture2DSampleLevel(PostprocessInput0, PostprocessInput0Sampler, UVAndScreenPos.xy, 0);
	float PixelCoC = NearmaxCoCTilemap_1_8_Halo[px/4];// Texture2DSampleLevel(PostprocessInput1, PostprocessInput1Sampler, UVAndScreenPos.xy, 0); 

	float3 ResultColor = 0;
	float Weight = 0;
	int TAP_COUNT = BLUR_TAP_COUNT; // Higher means less noise and make floodfilling easier after
	
	float radius = KernelSize.x * 0.7f * PixelCoC; 

	uint noiseTextureFrameIndex = /*$(Variable:AnimateNoiseTextures)*/ ? /*$(Variable:FrameIndex)*/ : 0;
	uint3 pxAndFrame = uint3(px, noiseTextureFrameIndex);
	
	if (PixelCoC > 0) { // Early exit based on MaxCoC tilemap
		for (int u = 0; u < TAP_COUNT; ++u)
		{
			for (int v = 0; v < TAP_COUNT; ++v)
			{
				float2 uv = GetApertureSamplePoint(pxAndFrame, u, v, TAP_COUNT, KernelSize);
				uv /= float2(NearFieldColorCoCBorderSize);

				//float2 uv = float2(u, v) / (TAP_COUNT - 1); // map to [0, 1]
				//uv = SquareToPolygonMapping( uv, KernelSize ) / float2(NearFieldColorCoCBorderSize); // map to bokeh shape, then to texel size
				uv = UVAndScreenPos.xy + radius * uv;

				float4 tapColor = NearFieldColorCoCBorder.SampleLevel(linearClampSampler, uv, 0);//  Texture2DSampleLevel(PostprocessInput0, PostprocessInput0Sampler, uv, 0);
				float TapWeight = saturate(tapColor.w * 10.0f); // From CoC 0.1 rely only on the near field and stop lerping with in-focus area 
				
				ResultColor +=  tapColor.xyz * TapWeight; 
				Weight += TapWeight;
			}
		}
		
		ResultColor /= (Weight + 0.0000001f);
		Weight /= (TAP_COUNT * TAP_COUNT);
	}
	float4 OutColor = float4(ResultColor, Weight);	

	NearFieldColorCoCBorderBlurred[px] = OutColor;
}

/*
Shader Resources:
	Texture NearFieldColorCoCBorder (as SRV)
	Count NearmaxCoCTilemap_1_8_Halo (as Count)
	Texture NearFieldColorCoCBorderBlurred (as UAV)
*/
