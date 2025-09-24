///////////////////////////////////////////////////////////////////////////////
//     Filter-Adapted Spatio-Temporal Sampling With General Distributions    //
//        Copyright (c) 2024 Electronic Arts Inc. All rights reserved.       //
///////////////////////////////////////////////////////////////////////////////

// Unnamed technique, shader NearBorderCS


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

Texture2D<float4> NearFieldColorCoC : register(t0);
RWTexture2D<float4> NearFieldColorCoCBorder : register(u0);

#line 7


[numthreads(8, 8, 1)]
#line 9
void csmain(uint3 DTid : SV_DispatchThreadID)
{
	uint2 px = DTid.xy;
	
	float4 PixelColor = NearFieldColorCoC[px];// Texture2DSampleLevel(PostprocessInput0, PostprocessInput0Sampler, tapUV, 0); 
	
	if (PixelColor.w == 0) // Only fill the empty areas around near field
	{
		PixelColor = 0;
		float Weight = 0;
		int RADIUS_TAPS = 1;
		for (int u = -RADIUS_TAPS; u <= RADIUS_TAPS; ++u)
		{
			for (int v = -RADIUS_TAPS; v <= RADIUS_TAPS; ++v)
			{
				float4 tapValue = NearFieldColorCoC[px + int2(u,v)];//Texture2DSampleLevel(PostprocessInput0, PostprocessInput0Sampler, tapUV + float2(u,v) * PostprocessInput0Size.zw, 0); 
				float tapWeight = tapValue.w == 0.0f? 0.0f : 1.0f;
				PixelColor += tapWeight * tapValue;
				Weight += tapWeight;
			}
		}
		PixelColor /= (Weight + 0.0000001f);
		PixelColor.w = 0;
	}
	
	float4 OutColor = PixelColor;

	NearFieldColorCoCBorder[px] = OutColor;
}

/*
Shader Resources:
	Texture NearFieldColorCoC (as SRV)
	Texture NearFieldColorCoCBorder (as UAV)
*/
