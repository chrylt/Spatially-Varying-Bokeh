///////////////////////////////////////////////////////////////////////////////
//     Filter-Adapted Spatio-Temporal Sampling With General Distributions    //
//        Copyright (c) 2024 Electronic Arts Inc. All rights reserved.       //
///////////////////////////////////////////////////////////////////////////////

// Unnamed technique, shader DownscaleTimeMapCS


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

Texture2D<float> Source : register(t0);
RWTexture2D<float> Dest : register(u0);

#line 7


[numthreads(8, 8, 1)]
#line 9
void csmain(uint3 DTid : SV_DispatchThreadID)
{
	uint2 px = DTid.xy;

	float CoCs[4];

	CoCs[0] = Source[px*2+uint2(0,0)];
	CoCs[1] = Source[px*2+uint2(1,0)];
	CoCs[2] = Source[px*2+uint2(0,1)];
	CoCs[3] = Source[px*2+uint2(1,1)];

	Dest[px] = max( CoCs[0], max( CoCs[1], max( CoCs[2], CoCs[3] ) ) );
}

/*
Shader Resources:
	Texture Source (as SRV)
	Texture Dest (as UAV)
*/
