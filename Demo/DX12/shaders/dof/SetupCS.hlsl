///////////////////////////////////////////////////////////////////////////////
//     Filter-Adapted Spatio-Temporal Sampling With General Distributions    //
//        Copyright (c) 2024 Electronic Arts Inc. All rights reserved.       //
///////////////////////////////////////////////////////////////////////////////

// Unnamed technique, shader SetupCS


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

struct Struct__GatherDOF_SetupCSCB
{
    float GatherDOF_FarTransitionRegion;
    float GatherDOF_FocalDistance;
    float GatherDOF_FocalLength;
    float GatherDOF_FocalRegion;
    float GatherDOF_NearTransitionRegion;
    float GatherDOF_Scale;
    uint GatherDOF_SuppressBokeh;
    float _padding0;
};

Texture2D<float4> Color : register(t0);
Texture2D<float> Depth : register(t1);
RWTexture2D<float4> FarFieldColorCoC : register(u0);
RWTexture2D<float4> NearFieldColorCoC : register(u1);
RWTexture2D<float4> NearMaxCocTilemap : register(u2);
ConstantBuffer<Struct__GatherDOF_SetupCSCB> _GatherDOF_SetupCSCB : register(b0);

#line 7


// An implementation of GatherDOF2018, based on: https://www.adriancourreges.com/blog/2018/12/02/ue4-optimized-post-effects/

#include "Common.hlsli"

// Computed the "Circle Of Confusion" radius for "Depth of Field"
// Formula can be found in many places e.g. http://http.developer.nvidia.com/GPUGems/gpugems_ch23.html
// @param SceneDepth
// @return 0..1 0=in focus, 1:max blurry
float ComputeCircleOfConfusion(float SceneDepth, in ViewStruct View)
{
	// artificial area where all content is in focus (starting at FocalLength, ending at FocalLength+FocalRegion)
	if(SceneDepth > View.DepthOfFieldFocalDistance)
	{
		SceneDepth = View.DepthOfFieldFocalDistance + max(0, SceneDepth - View.DepthOfFieldFocalDistance - View.DepthOfFieldFocalRegion);
	}

	// depth of the pixel in unreal units
	float D = SceneDepth;
	// e.g. Focal length in mm (Camera property e.g. 75mm)
	float F = View.DepthOfFieldFocalLength;
	// Plane in Focus in unreal units
	float P = View.DepthOfFieldFocalDistance;
	// Camera property e.g. 0.5f, like aperture
	float Aperture = View.DepthOfFieldScale;
		
	 
	// convert unreal units (100=1m) to mm
	P *= 0.001f / 100.0f;
	D *= 0.001f / 100.0f;
/*
	float Div = abs(D * (P - F));
	// clamp crazy numbers
//	Div = max(0.01f, Div);
	float CoCRadiusFactor = Aperture * F * abs(P - D) / Div;
	return saturate(CoCRadiusFactor);
*/
	// note: F has almost no effect
	float CoCRadius = Aperture * F * (P - D) / (D * (P - F));

	return saturate(abs(CoCRadius));
}

// TODO Taken from BokehDOF. Centralize!
// @return x:layer in front of focal plane, y: layer behind focal plane  1-x-y:layer in focus
float2 ComputeLayerContributions(float Depth, ViewStruct View)
{
	float Front = saturate((View.DepthOfFieldFocalDistance - Depth) / View.DepthOfFieldNearTransitionRegion);
	float Back = saturate((Depth - View.DepthOfFieldFocalDistance - View.DepthOfFieldFocalRegion) / max(View.DepthOfFieldFarTransitionRegion, 0.0001f));
	return float2(Front, Back);
}

// TODO Taken from BokehDOF. Centralize!
float4 CommonDOFSetup(/*in float2 CenterUV*/ in uint2 px, out bool bFrontLayer, out float4 Mask, ViewStruct View)
{
	/*
	float2 Offset = PostprocessInput0Size.zw;

	float2 UV[4];

	// no filtering (2x2 kernel) to get no leaking in Depth of Field
	UV[0] = CenterUV + Offset * float2(-0.5f, -0.5f);
	UV[1] = CenterUV + Offset * float2( 0.5f, -0.5f);
	UV[2] = CenterUV + Offset * float2(-0.5f,  0.5f);
	UV[3] = CenterUV + Offset * float2( 0.5f,  0.5f);
	*/
	uint2 samplePoints[4];
	samplePoints[0] = px*2 + uint2(0,0);
	samplePoints[1] = px*2 + uint2(1,0);
	samplePoints[2] = px*2 + uint2(0,1);
	samplePoints[3] = px*2 + uint2(1,1);

	float4 ColorAndDepth[4];
	float2 Layer[4];

	for(uint i = 0; i < 4; ++i)
	{
		// clamping to a small number fixes black dots appearing (denorms?, 0 doesn't fix it)
		ColorAndDepth[i].rgb = max(float3(0.0001f, 0.0001f, 0.0001f), Color[samplePoints[i]].rgb /*Texture2DSample(PostprocessInput0, PostprocessInput0Sampler, UV[i]).rgb*/);
		ColorAndDepth[i].a = Depth[samplePoints[i]];
		Layer[i] = ComputeLayerContributions(ColorAndDepth[i].a, View);
	}

	float2 LayerSum = Layer[0] + Layer[1] + Layer[2] + Layer[3];
	bFrontLayer = LayerSum.x > LayerSum.y;

	Mask = bFrontLayer ?
		float4(Layer[0].x, Layer[1].x, Layer[2].x, Layer[3].x) :
		float4(Layer[0].y, Layer[1].y, Layer[2].y, Layer[3].y);

	float SumMask = dot(Mask, 1);

	float4 OutColor;

	if(SumMask > 0.001f)
	{
		OutColor = (
			ColorAndDepth[0] * Mask.x +
			ColorAndDepth[1] * Mask.y +
			ColorAndDepth[2] * Mask.z +
			ColorAndDepth[3] * Mask.w ) / SumMask;
	}
	else
	{
		OutColor = ColorAndDepth[0];
	}
	return OutColor;
}


[numthreads(8, 8, 1)]
#line 118
void csmain(uint3 DTid : SV_DispatchThreadID)
{
	// ================================================
	// Our setup code
	uint2 px = DTid.xy;
	float4 OutColor0 = float4(0.0f, 0.0f, 0.0f, 0.0f);
	float4 OutColor1 = float4(0.0f, 0.0f, 0.0f, 0.0f);
	float4 OutColor2 = float4(0.0f, 0.0f, 0.0f, 0.0f);

	ViewStruct View = (ViewStruct)0;
	View.DepthOfFieldFocalLength = _GatherDOF_SetupCSCB.GatherDOF_FocalLength;
	View.DepthOfFieldFocalDistance = _GatherDOF_SetupCSCB.GatherDOF_FocalDistance;
	View.DepthOfFieldFocalRegion = _GatherDOF_SetupCSCB.GatherDOF_FocalRegion;
	View.DepthOfFieldNearTransitionRegion = _GatherDOF_SetupCSCB.GatherDOF_NearTransitionRegion;
	View.DepthOfFieldFarTransitionRegion = _GatherDOF_SetupCSCB.GatherDOF_FarTransitionRegion;
	View.DepthOfFieldScale = _GatherDOF_SetupCSCB.GatherDOF_Scale;
	// ================================================

	// unused for this pass
	bool bFrontLayer = false;
	float4 Mask = float4(0.0f, 0.0f, 0.0f, 0.0f);

	float4 SceneColorAndDepth = CommonDOFSetup(px, bFrontLayer, Mask, View);

	// clamp to avoid artifacts from exceeding fp16 through framebuffer blending of multiple very bright lights
	SceneColorAndDepth.rgb = min(float3(256 * 256, 256 * 256, 256 * 256), SceneColorAndDepth.rgb);
	
	float SceneDepth = SceneColorAndDepth.a;
	float CircleOfConfusion = ComputeCircleOfConfusion(SceneDepth, View);
	
	if (SceneDepth < View.DepthOfFieldFocalDistance) {
		// Near
		OutColor0 = float4(0, 0, 0, 0);
		OutColor1 = float4(SceneColorAndDepth.rgb, CircleOfConfusion);
		OutColor2 = CircleOfConfusion; // Separate CoC to build Max TileMap
	} 
	else if (SceneDepth >= View.DepthOfFieldFocalDistance + View.DepthOfFieldFocalRegion)
	{
		// Far
		OutColor0 = float4(SceneColorAndDepth.rgb, CircleOfConfusion);
		OutColor1 = float4(0, 0, 0, 0);
		OutColor2 = 0.0f;
	}
	else
	{
		// In-focus (inside focal region)
		OutColor0 = float4(0, 0, 0, 0);
		OutColor1 = float4(0, 0, 0, 0);
		OutColor2 = 0.0f;
	}

	if (_GatherDOF_SetupCSCB.GatherDOF_SuppressBokeh)
	{
		OutColor0.rgb = clamp(OutColor0.rgb, float3(0.0f, 0.0f, 0.0f), float3(1.0f, 1.0f, 1.0f));
		OutColor1.rgb = clamp(OutColor1.rgb, float3(0.0f, 0.0f, 0.0f), float3(1.0f, 1.0f, 1.0f));
	}

	FarFieldColorCoC[px] = OutColor0;
	NearFieldColorCoC[px] = OutColor1;
	NearMaxCocTilemap[px] = OutColor2;
}

/*
Shader Resources:
	Texture Color (as SRV)
	Texture Depth (as SRV)
	Texture FarFieldColorCoC (as UAV)
	Texture NearFieldColorCoC (as UAV)
	Texture NearMaxCocTilemap (as UAV)
*/
