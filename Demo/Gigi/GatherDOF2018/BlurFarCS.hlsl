///////////////////////////////////////////////////////////////////////////////
//     Filter-Adapted Spatio-Temporal Sampling With General Distributions    //
//        Copyright (c) 2024 Electronic Arts Inc. All rights reserved.       //
///////////////////////////////////////////////////////////////////////////////

// Unnamed technique, shader BlurFarCS
/*$(ShaderResources)*/

#include "Common.hlsli"
#include "PCG.hlsli"
#include "LDSShuffler.hlsli"

uint3 AdjustNoiseTextureCoords(uint3 pxAndFrame, uint3 dims)
{
	uint3 coords = pxAndFrame;
	uint cycleCount = pxAndFrame.z / dims.z;

	switch(/*$(Variable:LensRNGExtend)*/)
	{
		case NoiseTexExtends::None: break;
		case NoiseTexExtends::White:
		{
			uint OffsetRNG = HashInit(uint3(0x1337, 0xbeef, cycleCount));
			coords.x += HashPCG(OffsetRNG);
			coords.y += HashPCG(OffsetRNG);
			break;
		}
		case NoiseTexExtends::Shuffle1D:
		{
			uint shuffleIndex = LDSShuffle1D_GetValueAtIndex(cycleCount, 16384, 10127, 435);
			coords.x += shuffleIndex % dims.x;
			coords.y += shuffleIndex / dims.x;
			break;
		}
		case NoiseTexExtends::Shuffle1DHilbert:
		{
			uint shuffleIndex = LDSShuffle1D_GetValueAtIndex(cycleCount, 16384, 10127, 435);
			coords.xy += Convert1DTo2D_Hilbert(shuffleIndex, 16384);
			break;
		}
	}

	return coords % dims;
}

float2 ReadVec2STTextureRaw(in uint3 pxAndFrame, in Texture2DArray<float2> tex)
{
	uint3 dims;
	tex.GetDimensions(dims.x, dims.y, dims.z);
	uint3 sampleCoord = AdjustNoiseTextureCoords(pxAndFrame, dims);
	return tex[sampleCoord].rg;
}

float ReadFloatSTTextureRaw(in uint3 pxAndFrame, in Texture2DArray<float> tex)
{
	uint3 dims;
	tex.GetDimensions(dims.x, dims.y, dims.z);
	uint3 sampleCoord = AdjustNoiseTextureCoords(pxAndFrame, dims);
	return tex[sampleCoord];
}

float2 ReadVec2STTexture(in uint3 pxAndFrame, in Texture2DArray<float2> tex) // already applies transformation to [-1.0, 1.0]
{
	uint jitterRNG = HashInit(pxAndFrame);

    float2 ret = ReadVec2STTextureRaw(pxAndFrame, tex) * 2.0f - 1.0f;

	if (/*$(Variable:JitterNoiseTextures)*/)
		ret += float2((RandomFloat01(jitterRNG) - 0.5f) / 255.0f, (RandomFloat01(jitterRNG) - 0.5f) / 255.0f);

	return ret;
}

//------------- spatially varying bokeh --------------------

struct RotationBasis
{
	float2 row0;
	float2 row1;
};

RotationBasis BuildRotationBasis(float angle)
{
	float sine;
	float cosine;
	sincos(angle, sine, cosine);

	RotationBasis basis;
	basis.row0 = float2(cosine, -sine);
	basis.row1 = float2(sine, cosine);
	return basis;
}

float2 RotateForward(float2 v, RotationBasis basis)
{
	return float2(dot(basis.row0, v), dot(basis.row1, v));
}

float2 RotateBackward(float2 v, RotationBasis basis)
{
	float2 column0 = float2(basis.row0.x, basis.row1.x);
	float2 column1 = float2(basis.row0.y, basis.row1.y);
	return float2(dot(column0, v), dot(column1, v));
}



struct ScreenGeometry
{
	float2 center;
	float invCenterToCornerDistance;
	float2 pixelPosition;
	float2 screenSize;
};

// slow distortion start

struct DistortionStageInfo
{
	uint stageIndex;
	uint maxStageIndex;
	float alpha;
};

DistortionStageInfo ComputeDistortionStageInfo(float normalizedDistance, uint stageCount)
{
	DistortionStageInfo info = (DistortionStageInfo)0;
	if (stageCount == 0)
	{
		return info;
	}

	info.maxStageIndex = stageCount - 1;
	float stagePosition = normalizedDistance * info.maxStageIndex;
	float stageFloor = floor(stagePosition);
	info.stageIndex = (uint)min(stageFloor, (float)info.maxStageIndex);
	info.alpha = stagePosition - stageFloor;
	if (info.stageIndex == info.maxStageIndex)
	{
		info.alpha = 0.0f;
	}
	return info;
}

float2 SampleDistortionStage(float2 currentOffset, uint stageIndex, Texture2DArray<float2> distortionMaps)
{
	float2 uv = currentOffset * 0.5f + 0.5f;
	float2 sample = distortionMaps.SampleLevel(linearClampSampler, float3(saturate(uv), stageIndex), 0).rg;
	return sample * 2.0f - 1.0f;
}

float2 ApplyDistortionStagesSlow(float2 offset, ScreenGeometry screen, float2 centerToSamplePos)
{
	const uint kStageCount = 8u;

	float normalizedDistance = length(centerToSamplePos) * screen.invCenterToCornerDistance;
	Texture2DArray<float2> distortionMaps = /*$(Image2DArray:Assets\DistortionMaps\one_after_another\distortion_map_%i.png:RG8_UNorm:float2:false:false)*/;
	DistortionStageInfo stageInfo = ComputeDistortionStageInfo(normalizedDistance, kStageCount);

	for (uint stage = 0; stage < stageInfo.stageIndex; ++stage)
	{
		offset = SampleDistortionStage(offset, stage, distortionMaps);
	}

	if (stageInfo.alpha > 0.0f && stageInfo.stageIndex < stageInfo.maxStageIndex)
	{
		float2 nextOffset = SampleDistortionStage(offset, stageInfo.stageIndex, distortionMaps);
		offset = lerp(offset, nextOffset, stageInfo.alpha);
	}

	return offset;
}

// slow distortion end

float2 ApplyDistortionStagesFast(float2 offset, ScreenGeometry screen, float2 centerToSamplePos)
{
	const uint kStageCount = 8u;
	float normalizedDistance = length(centerToSamplePos) * screen.invCenterToCornerDistance;
	DistortionStageInfo stageInfo = ComputeDistortionStageInfo(normalizedDistance, kStageCount);
	float2 uv = offset * 0.5f + 0.5f;
	Texture2DArray<float2> distortionMaps = /*$(Image2DArray:Assets\DistortionMaps\one_sample\distortion_map_%i.png:RG8_UNorm:float2:false:false)*/;
	float2 sample = distortionMaps.SampleLevel(linearClampSampler, float3(uv, normalizedDistance * (kStageCount - 1)), 0).rg;
	return sample * 2.0f - 1.0f;
}

float GetSpatialIntensity(float normalizedDistance)
{
	float n = saturate(normalizedDistance);
	float n2 = n * n;
	return mad(-0.41720654f, n2, mad(-0.25085544f, n, 1.00672758f));
}

float3 getSpatiallyVaryingOffset(uint3 pxAndSampleIndex, uint2 screenSize)
{
	ScreenGeometry screen;
	screen.center = float2(screenSize) * 0.5f;
	screen.invCenterToCornerDistance = rcp(max(length(screen.center), 1e-5f));
	screen.pixelPosition = float2(pxAndSampleIndex.xy);
	screen.screenSize = float2(screenSize);

	float2 sampledOffset = ReadVec2STTexture(pxAndSampleIndex, /*$(Image2DArray:Assets\NoiseTextures\bokeh\bokeh_fl45.0_as6_samples1000000_od400_lidx0of15_%i.png:RG8_UNorm:float2:false:false)*/);
	float PixelCoC = FarFieldColorCoC[pxAndSampleIndex.xy].w;
	float blurRadius = /*$(Variable:KernelSize)*/.x * PixelCoC;
	
	float2 centerToSamplePos = (screen.pixelPosition + sampledOffset * blurRadius) - screen.center;
	float sampleAngle = atan2(centerToSamplePos.y, centerToSamplePos.x);

	static const float c_bottomLeftDirectionAngle = 2.5535900500422257; // atan2(2, -3) to match aspect ratio at which the bokeh textures were generated
	RotationBasis rotation = BuildRotationBasis(sampleAngle - c_bottomLeftDirectionAngle);
	float2 offsetLocal = RotateBackward(sampledOffset, rotation);

	offsetLocal = ApplyDistortionStagesFast(offsetLocal, screen, centerToSamplePos);

	float2 offsetScreen = RotateForward(offsetLocal, rotation);
	float2 samplePos = screen.pixelPosition + offsetScreen * blurRadius;

	samplePos = screen.pixelPosition + offsetScreen * blurRadius;

	float spatialIntensity = GetSpatialIntensity(length(samplePos - screen.center) * screen.invCenterToCornerDistance);

	return float3(offsetScreen, spatialIntensity);
}

//------------- spatially varying bokeh end --------------------

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

float2 GetApertureSamplePoint(uint3 pxAndFrame, int u, int v, int maxuv, in float4 KernelSize, out float sampleWeight, uint2 screenSize)
{
	sampleWeight = 1.0f;

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
		case LensRNG::NonUniformStarWhite:
		{
			return ReadVec2STTexture(pxAndSampleIndex, /*$(Image2DArray:Assets\NoiseTextures\NonUniformStar\NonUniformStar_%i.0.png:RG8_UNorm:float2:false:false)*/);
		}
		case LensRNG::NonUniformStarBlue:
		{
			return ReadVec2STTexture(pxAndSampleIndex, /*$(Image2DArray:Assets\NoiseTextures\NonUniformStar\NonUniformStar_%i.png:RG8_UNorm:float2:false:false)*/);
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
		case LensRNG::bokeh:
		{
			float3 svoffset = getSpatiallyVaryingOffset(pxAndSampleIndex, screenSize);
			sampleWeight = svoffset.z;
			
			return svoffset.rg;
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

	uint noiseTextureFrameIndex = /*$(Variable:AnimateNoiseTextures)*/ ? /*$(Variable:FrameIndex)*/ : 0;
	uint3 pxAndFrame = uint3(px, noiseTextureFrameIndex);


	if (PixelCoC > 0) { // Ignore any pixel not belonging to far field
	
		// Weighted average of the texture samples inside the bokeh pattern
		// High radius and low sample count can create "gaps" which are fixed later (floodfill).
		for (int u = 0; u < TAP_COUNT; ++u)
		{
			for (int v = 0; v < TAP_COUNT; ++v)
			{
				float sampleWeight = 1.0f;
				float2 uv = GetApertureSamplePoint(pxAndFrame, u, v, TAP_COUNT, KernelSize, sampleWeight, FarFieldColorCoCSize);
				uv /= float2(FarFieldColorCoCSize);

				//float2 uv = float2(u, v) / (TAP_COUNT - 1); // map to [0, 1]
				//uv = SquareToPolygonMapping( uv, KernelSize ) / float2(FarFieldColorCoCSize); // map to bokeh shape, then to texel size
				uv = UVAndScreenPos.xy + radius * uv;

				float4 tapColor = FarFieldColorCoC.SampleLevel(linearClampSampler, uv, 0); //Texture2DSampleLevel(PostprocessInput0, PostprocessInput0Sampler, uv, 0);
				
				// Weighted by CoC. Gives more influence to taps with a CoC higher than us.
				float TapWeight = tapColor.w * saturate(1.0f - (PixelCoC - tapColor.w)); 
				
				ResultColor +=  tapColor.xyz * sampleWeight * TapWeight;
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
