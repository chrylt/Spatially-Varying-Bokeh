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

static const bool tdebug = /*$(Variable:Debug)*/;

#include "SpatiallyVaryingBokeh.hlsli"

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

float3 GetApertureSamplePoint(uint3 pxAndFrame, float pixelCoC, int u, int v, int maxuv, in float4 KernelSize, out float sampleWeight, uint2 screenSize)
{
	sampleWeight = 1.0f;

	// calculate what sample index we are on
	uint sampleIndex = pxAndFrame.z * maxuv * maxuv;
	sampleIndex += v * maxuv + u;
	uint3 pxAndSampleIndex = uint3(pxAndFrame.xy, sampleIndex);

	float2 offset = float2(0.0f, 0.0f);
	switch(/*$(Variable:LensRNGSource)*/)
	{
		case LensRNG::bokeh:
		{
			Texture2DArray<float2> noiseTexture = /*$(Image2DArray:Assets\NoiseTextures\bokeh\base_bokeh_%i.png:RG8_UNorm:float2:false:false)*/;
			Texture3D<float2> distortionMapsFast = /*$(Image3D:Assets\DistortionMaps\one_sample\distortion_map_%i.png:RG8_UNorm:float2:false:false)*/;
			Texture2DArray<float2> distortionMapsSlow = /*$(Image2DArray:Assets\DistortionMaps\one_after_another\distortion_map_%i.png:RG8_UNorm:float2:false:false)*/;
			
			float3 svoffset;

			bool spatiallyVarying = /*$(Variable:SpatiallyVarying)*/;
			bool fastDistortion = /*$(Variable:FastDistortion)*/;
			
			if(spatiallyVarying)
				svoffset = getSpatiallyVaryingOffset(pxAndSampleIndex, pixelCoC, screenSize, noiseTexture, KernelSize, distortionMapsFast, distortionMapsSlow, fastDistortion);
			else
				svoffset = getSpatiallyConstantOffset(pxAndSampleIndex, noiseTexture);

			sampleWeight = svoffset.z;
			
			return float3(svoffset.rg, svoffset.z);
        }
	}

	return float3(0.0f, 0.0f, 0.0f);
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
				float3 uv = GetApertureSamplePoint(pxAndFrame, PixelCoC, u, v, TAP_COUNT, KernelSize, sampleWeight, FarFieldColorCoCSize);
				uv.xy /= float2(FarFieldColorCoCSize);

				//float2 uv = float2(u, v) / (TAP_COUNT - 1); // map to [0, 1]
				//uv = SquareToPolygonMapping( uv, KernelSize ) / float2(FarFieldColorCoCSize); // map to bokeh shape, then to texel size
				uv.xy = UVAndScreenPos.xy + radius * uv.xy;

				// Mirror coordinates outside [0,1] to prevent edge artifacts
				uv.xy = 1.0f - abs(fmod(abs(uv.xy), 2.0f) - 1.0f);

				float4 tapColor = FarFieldColorCoC.SampleLevel(linearClampSampler, uv.xy, 0); //Texture2DSampleLevel(PostprocessInput0, PostprocessInput0Sampler, uv, 0);
				// Weighted by CoC. Gives more influence to taps with a CoC higher than us.
				float TapWeight = tapColor.w * saturate(1.0f - (PixelCoC - tapColor.w)); 
				
				ResultColor +=  tapColor.xyz * sampleWeight * TapWeight;
				Weight += TapWeight;

				if (tdebug) {
					ResultColor = float4(uv.z, 0.0f, 0.0f, 1.0f); // debug: visualize UV coordinates as color
					Weight = 1.0f; // debug: disable weighting
				} 
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
