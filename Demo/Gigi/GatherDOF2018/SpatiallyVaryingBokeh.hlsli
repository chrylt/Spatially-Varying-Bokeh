


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

DistortionStageInfo ComputeDistortionStageInfo(float normalizedDistance)
{
	const float kRenderedBokehConfigs = 15.0f;
	const float kHiddenBokehCount = 2.0f;
	const float kDistortionMapsCount = floor((kRenderedBokehConfigs - kHiddenBokehCount) / 2.0f) + 1.0f;

	DistortionStageInfo info = (DistortionStageInfo)0;

	info.maxStageIndex = kDistortionMapsCount - 1;
	float stagePosition = min(normalizedDistance * (kRenderedBokehConfigs - 1), (kRenderedBokehConfigs - kHiddenBokehCount - 1)) / 2.0f; // compensate for hidden stages
	info.stageIndex = floor(stagePosition);
	info.alpha = frac(stagePosition);
	if (info.stageIndex == info.maxStageIndex)
	{
		info.alpha = 0.0f;
	}
	return info;
}

float2 SampleDistortionStage(float2 currentOffset, uint stageIndex, Texture2DArray<float2> distortionMaps)
{
	float2 uv = currentOffset * 0.5f + 0.5f;
	uv = uv * 51.0f / 52.0f + 1.0f / 52.0f / 2.0f;
	float2 sample = distortionMaps.SampleLevel(linearClampSampler, float3(uv, stageIndex), 0).rg;
	return sample * 2.0f - 1.0f;
}

float3 ApplyDistortionStagesSlow(float2 offset, ScreenGeometry screen, float2 centerToSamplePos, Texture2DArray<float2> distortionMaps) 
{
	float normalizedDistance = length(centerToSamplePos) * screen.invCenterToCornerDistance;
	DistortionStageInfo stageInfo = ComputeDistortionStageInfo(normalizedDistance);

	float debugFloat = 1.0f;

	for (uint stage = 1; stage <= stageInfo.stageIndex; ++stage)
	{
		offset = SampleDistortionStage(offset, stage, distortionMaps);
	}

	if (stageInfo.alpha > 0.0f && stageInfo.stageIndex < stageInfo.maxStageIndex)
	{
		float2 nextOffset = SampleDistortionStage(offset, stageInfo.stageIndex + 1, distortionMaps);
		offset = lerp(offset, nextOffset, stageInfo.alpha);
	}

	//if (stageInfo.stageIndex == stageInfo.maxStageIndex)
	//{
		//debugFloat = 1.0f; // debug: visualize interpolation factor of last stage as color
	//}
	//debugFloat = (float(stageInfo.stageIndex) + stageInfo.alpha) / float(stageInfo.maxStageIndex);

	return float3(offset, debugFloat);
}

// slow distortion end

float3 ApplyDistortionStagesFast(float2 offset, ScreenGeometry screen, float2 centerToSamplePos, Texture3D<float2> distortionMaps)
{
	const float kRenderedBokehConfigs = 15.0f;
	const float kHiddenBokehCount = 2.0f;
	const float kDistortionMapsCount = floor((kRenderedBokehConfigs - kHiddenBokehCount) / 2.0f) + 1.0f;

	float normalizedDistance = length(centerToSamplePos) * screen.invCenterToCornerDistance;
	normalizedDistance = min(normalizedDistance * (kRenderedBokehConfigs - 1), (kRenderedBokehConfigs - kHiddenBokehCount - 1)) / 2.0f; // compensate for hidden stages
	float2 uv = offset * 0.5f + 0.5f;
	uv = uv * 51.0f / 52.0f + 1.0f / 52.0f / 2.0f;
	float w = (normalizedDistance / (kDistortionMapsCount - 1)) * 6.0f / 7.0f + 1.0f / 7.0f / 2.0f; // compensate for texel center at 0.5
	float2 sample = distortionMaps.SampleLevel(linearClampSampler, float3(uv, w), 0).rg;
	return float3(sample * 2.0f - 1.0f, 1.0f);
}

float GetSpatialIntensity(float normalizedDistance)
{
	float n = saturate(normalizedDistance);
	float n2 = n * n;
	return mad(-1.15455, n2, mad(0.289646, n, 1.00833));
}

float3 getSpatiallyVaryingOffset(uint3 pxAndSampleIndex, float pixelCoC, uint2 screenSize, Texture2DArray<float2> noiseTexture, float4 kernelSize, Texture3D<float2> distortionMapsFast, Texture2DArray<float2> distortionMapsSlow, bool fastDistortion)
{
	ScreenGeometry screen;
	screen.center = float2(screenSize) * 0.5f;
	screen.invCenterToCornerDistance = rcp(max(length(screen.center), 1e-5f));
	screen.pixelPosition = float2(pxAndSampleIndex.xy);
	screen.screenSize = float2(screenSize);

	float2 sampledOffset = ReadVec2STTexture(pxAndSampleIndex, noiseTexture);
	float blurRadius = kernelSize.x * pixelCoC;
	
	float2 centerToSamplePos = (screen.pixelPosition + sampledOffset * blurRadius) - screen.center;
	float sampleAngle = atan2(centerToSamplePos.y, centerToSamplePos.x);

	static const float c_bottomLeftDirectionAngle = 2.5535900500422257; // atan2(2, -3) to match aspect ratio at which the bokeh textures were generated
	RotationBasis rotation = BuildRotationBasis(sampleAngle - c_bottomLeftDirectionAngle);
	float2 offset = RotateBackward(sampledOffset, rotation);
	float3 offsetLocal = float3(offset, 1.0f);// GetSpatialIntensity(length(centerToSamplePos) * screen.invCenterToCornerDistance));

	if(fastDistortion) {
		offsetLocal = ApplyDistortionStagesFast(offset, screen, centerToSamplePos, distortionMapsFast);
	} else {
		offsetLocal = ApplyDistortionStagesSlow(offset, screen, centerToSamplePos, distortionMapsSlow);
	}

	float2 offsetScreen = RotateForward(offsetLocal.xy, rotation);
	float2 samplePos = screen.pixelPosition + offsetScreen * blurRadius;

	//float spatialIntensity = GetSpatialIntensity(length(samplePos - screen.center) * screen.invCenterToCornerDistance);

	return float3(offsetScreen, offsetLocal.z);//spatialIntensity);
}

float3 getSpatiallyConstantOffset(uint3 pxAndSampleIndex, Texture2DArray<float2> noiseTexture)
{
	float2 sampledOffset = ReadVec2STTexture(pxAndSampleIndex, noiseTexture);

	return float3(sampledOffset, 1.0f);
}