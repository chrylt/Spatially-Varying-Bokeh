
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
	float2 sample = distortionMaps.SampleLevel(linearClampSampler, float3(uv, stageIndex), 0).rg;
	return sample * 2.0f - 1.0f;
}

float2 ApplyDistortionStagesSlow(float2 offset, ScreenGeometry screen, float2 centerToSamplePos, Texture2DArray<float2> distortionMaps) 
{
	const uint kStageCount = 8u;

	float normalizedDistance = length(centerToSamplePos) * screen.invCenterToCornerDistance;
	DistortionStageInfo stageInfo = ComputeDistortionStageInfo(normalizedDistance, kStageCount);

	for (uint stage = 1; stage < stageInfo.stageIndex; ++stage)
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

float2 ApplyDistortionStagesFast(float2 offset, ScreenGeometry screen, float2 centerToSamplePos, Texture3D<float2> distortionMaps)
{
	const float kRenderedBokehConfigs = 15u;
	const float kHiddenBokehCount = 2u;
	const float kDistortionMapsCount = (kRenderedBokehConfigs - kHiddenBokehCount - 1) / 2;

	// todo: how to compensate for the hidden stages?
	float normalizedDistance = length(centerToSamplePos) * screen.invCenterToCornerDistance;
	normalizedDistance = normalizedDistance * kDistortionMapsCount / kRenderedBokehConfigs; // compensate for hidden stages
	float2 uv = offset * 0.5f + 0.5f;
	float w = (normalizedDistance * (kDistortionMapsCount - 1) + 0.5) / max(kDistortionMapsCount, 1); // compensate for texel center at 0.5
	float2 sample = distortionMaps.SampleLevel(linearClampSampler, float3(uv, w), 0).rg;
	return sample * 2.0f - 1.0f;
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
	float2 offsetLocal = RotateBackward(sampledOffset, rotation);

	if(fastDistortion) {
		offsetLocal = ApplyDistortionStagesFast(offsetLocal, screen, centerToSamplePos, distortionMapsFast);
	} else {
		offsetLocal = ApplyDistortionStagesSlow(offsetLocal, screen, centerToSamplePos, distortionMapsSlow);
	}

	float2 offsetScreen = RotateForward(offsetLocal, rotation);
	float2 samplePos = screen.pixelPosition + offsetScreen * blurRadius;

	float spatialIntensity = GetSpatialIntensity(length(samplePos - screen.center) * screen.invCenterToCornerDistance);

	return float3(offsetScreen, spatialIntensity);
}

float3 getSpatiallyConstantOffset(uint3 pxAndSampleIndex, Texture2DArray<float2> noiseTexture)
{
	float2 sampledOffset = ReadVec2STTexture(pxAndSampleIndex, noiseTexture);

	return float3(sampledOffset, 1.0f);
}