///////////////////////////////////////////////////////////////////////////////
//     Filter-Adapted Spatio-Temporal Sampling With General Distributions    //
//        Copyright (c) 2024 Electronic Arts Inc. All rights reserved.       //
///////////////////////////////////////////////////////////////////////////////

// Unnamed technique, shader TemporalAccum


struct Struct__TemporalAccumulation_AccumulateCB
{
    uint CameraChanged;
    float TemporalAccumulation_Alpha;
    uint TemporalAccumulation_Enabled;
    float _padding0;
};

Texture2D<float4> Input : register(t0);
RWTexture2D<float4> Accum : register(u0);
ConstantBuffer<Struct__TemporalAccumulation_AccumulateCB> _TemporalAccumulation_AccumulateCB : register(b0);

#line 7


[numthreads(8, 8, 1)]
#line 9
void csmain(uint3 DTid : SV_DispatchThreadID)
{
	float alpha = ((bool)_TemporalAccumulation_AccumulateCB.CameraChanged || !(bool)_TemporalAccumulation_AccumulateCB.TemporalAccumulation_Enabled)
		? 1.0f
		: _TemporalAccumulation_AccumulateCB.TemporalAccumulation_Alpha;

	uint2 px = DTid.xy;

	float4 oldValue = Accum[px];
	float4 newValue = Input[px];

	Accum[px] = lerp(oldValue, newValue, alpha);
}

/*
Shader Resources:
	Texture Input (as SRV)
	Texture Accum (as UAV)
*/
