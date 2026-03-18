

// bokeh config counts
static const float kRenderedBokehConfigs = 15.0f;
static const float kHiddenBokehCount     = 2.0f;
static const float kVisibleBokehConfigs  = kRenderedBokehConfigs - kHiddenBokehCount;
static const float kDistortionMapsCount  = floor(kVisibleBokehConfigs / 2.0f) + 1.0f;

struct RotationBasis
{
	float2 row0;
	float2 row1;
};

RotationBasis BuildRotationBasis(float angle)
{
	float s, c;
	sincos(angle, s, c);

	RotationBasis basis;
	basis.row0 = float2(c, -s);
	basis.row1 = float2(s,  c);
	return basis;
}

float2 RotateForward(float2 v, RotationBasis basis)
{
	return float2(dot(basis.row0, v), dot(basis.row1, v));
}

float2 RotateBackward(float2 v, RotationBasis basis)
{
	float2 col0 = float2(basis.row0.x, basis.row1.x);
	float2 col1 = float2(basis.row0.y, basis.row1.y);
	return float2(dot(col0, v), dot(col1, v));
}

struct ScreenGeometry
{
	float2 center;
	float  invCenterToCornerDistance;
	float2 pixelPosition;
	float2 screenSize;
};

ScreenGeometry MakeScreenGeometry(uint2 screenSize, float2 pixelPosition)
{
	ScreenGeometry sg;
	sg.center                    = float2(screenSize) * 0.5f;
	sg.invCenterToCornerDistance = rcp(max(length(sg.center), 1e-5f));
	sg.pixelPosition             = pixelPosition;
	sg.screenSize                = float2(screenSize);
	return sg;
}

// maps a [-1,1] offset to [0,1] UV, with half-texel inset so bilinear is correct
float2 OffsetToTexelCenterUV(float2 offset, float texDim)
{
	float2 uv = offset * 0.5f + 0.5f;
	return uv * (texDim - 1.0f) / texDim + 0.5f / texDim;
}

// slower path: steps through each distortion stage individually
struct DistortionStageInfo
{
	uint  stageIndex;
	uint  maxStageIndex;
	float alpha;
};

DistortionStageInfo ComputeDistortionStageInfo(float normalizedDistance)
{
	DistortionStageInfo info = (DistortionStageInfo)0;
	info.maxStageIndex = kDistortionMapsCount - 1;

	float stagePos  = saturate(normalizedDistance) * float(info.maxStageIndex);
	info.stageIndex = floor(stagePos);
	info.alpha      = (info.stageIndex == info.maxStageIndex) ? 0.0f : frac(stagePos);
	return info;
}

float2 SampleDistortionStage(float2 currentOffset, uint stageIndex, Texture2DArray<float2> distortionMaps, float texDim)
{
	float2 uv = OffsetToTexelCenterUV(currentOffset, texDim);
	float2 s  = distortionMaps.SampleLevel(linearClampSampler, float3(uv, stageIndex), 0).rg;
	return s * 2.0f - 1.0f;
}

float2 ApplyDistortionStagesSlow(float2 offset, float normalizedDistance, Texture2DArray<float2> distortionMaps)
{
	uint w, h, d;
    distortionMaps.GetDimensions(w, h, d);
    float texDim = float(w);

	DistortionStageInfo si  = ComputeDistortionStageInfo(normalizedDistance);

	for (uint stage = 1; stage <= si.stageIndex; ++stage)
		offset = SampleDistortionStage(offset, stage, distortionMaps, texDim);

	if (si.alpha > 0.0f && si.stageIndex < si.maxStageIndex)
	{
		float2 next = SampleDistortionStage(offset, si.stageIndex + 1, distortionMaps, texDim);
		offset = lerp(offset, next, si.alpha);
	}

	return offset;
}

// faster path: baked into a 3D texture so one sample does it all
float2 ApplyDistortionStagesFast(float2 offset, float normalizedDistance, Texture3D<float2> distortionMaps)
{
	uint tw, th, td;
	distortionMaps.GetDimensions(tw, th, td);

	float2 uv = OffsetToTexelCenterUV(offset, float(tw));
	float w = (normalizedDistance * (float(td) - 1.0f) + 0.5f) / float(td); // texel-center-corrected depth coordinate

	float2 s = distortionMaps.SampleLevel(linearClampSampler, float3(uv, w), 0).rg;
	return s * 2.0f - 1.0f;
}

float GetSpatialIntensity(float normalizedDistance)
{
	float n  = saturate(normalizedDistance);
	float n2 = n * n;
	return mad(-1.15455, n2, mad(0.289646, n, 1.00833));
}

float3 getSpatiallyVaryingOffset(uint3 pxAndSampleIndex, float pixelCoC, uint2 screenSize, Texture2DArray<float2> noiseTexture, float4 kernelSize, Texture3D<float2> distortionMapsFast, Texture2DArray<float2> distortionMapsSlow, bool fastDistortion)
{
	ScreenGeometry screen = MakeScreenGeometry(screenSize, float2(pxAndSampleIndex.xy));
	float2 centerToPixel = screen.pixelPosition - screen.center;
	float bokehClampFraction = (13.0f - 1.0f) / (15.0f - 1.0f); // compensate for clamped bokehs

	float2 sampledOffset = ReadVec2STTexture(pxAndSampleIndex, noiseTexture);

	const float kCanonicalAngle = atan2(-2, 3); // matches aspect ratio of bokeh textures
	float radius = kernelSize.x * pixelCoC;

	// Scatter-to-gather correction: the distortion maps define scatter shapes,
	// so the shape depends on the SOURCE pixel's position, not the gather pixel's.
	// Estimate the source position using the undistorted offset, then look up
	// distortion and rotation for that estimated source.
	RotationBasis gatherRotation = BuildRotationBasis(kCanonicalAngle - atan2(centerToPixel.y, centerToPixel.x));
	float2 undistortedScreenOffset = RotateBackward(-sampledOffset, gatherRotation);
	float2 centerToSource = centerToPixel + undistortedScreenOffset * radius;

	// distortion and rotation at the estimated source position
	float sourceNormalizedDistance = length(centerToSource) * screen.invCenterToCornerDistance / bokehClampFraction;
	RotationBasis sourceRotation = BuildRotationBasis(kCanonicalAngle - atan2(centerToSource.y, centerToSource.x));

	float2 offsetLocal = fastDistortion
		? ApplyDistortionStagesFast(sampledOffset, sourceNormalizedDistance, distortionMapsFast)
		: ApplyDistortionStagesSlow(sampledOffset, sourceNormalizedDistance, distortionMapsSlow);

	// negate for scatter-to-gather, rotate into the source's radial frame
	float2 offsetScreen = RotateBackward(-offsetLocal, sourceRotation);

	// weight by spatial intensity at the source position
	float spatialIntensity = GetSpatialIntensity(length(centerToSource) * screen.invCenterToCornerDistance);

	return float3(offsetScreen, spatialIntensity);
}

float3 getSpatiallyConstantOffset(uint3 pxAndSampleIndex, Texture2DArray<float2> noiseTexture)
{
	return float3(ReadVec2STTexture(pxAndSampleIndex, noiseTexture), 1.0f);
}