// Forward Declarations
float TestSphereTrace(in float3 rayPos, in float3 rayDir, in float4 sphere, out float3 normal);

// Fixed configuration constants
static const float3 BCONF_LIGHT_COLOR = float3(1.0f, 1.0f, 1.0f);
static const float VERTICAL_FOV = 22.0f; // degrees

bool TraceLightsDiagonal(float3 pos, float3 dir, float3 planeCenter, float3 cameraRight, float3 cameraUp, float fieldWidth, float fieldHeight, int targetIndex, inout float globalHitT)
{
	uint diagCount = max(t_config_light_count.x, t_config_light_count.y);
	diagCount = max(diagCount, 1u);

	float halfWidth = fieldWidth * 0.5f;
	float halfHeight = fieldHeight * 0.5f;
	float3 diagonalStart = planeCenter;
	float3 diagonalEnd = planeCenter + cameraRight * halfWidth + cameraUp * halfHeight;

	bool anyHit = false;

	for (uint i = 0u; i < diagCount; ++i)
	{
		int currentIndex = int(i);

		if (targetIndex >= 0)
		{
			if (currentIndex < targetIndex)
				continue;
			if (currentIndex > targetIndex)
				break;
		}

		float fraction = (diagCount > 1u) ? float(i) / float(diagCount - 1u) : 0.0f;
		float3 lightPos = lerp(diagonalStart, diagonalEnd, fraction);

		float3 sphereNormal;
		float t = TestSphereTrace(pos, dir, float4(lightPos, t_smallLightRadius), sphereNormal);
		if (t < 0.0f || t > globalHitT)
		{
			if (targetIndex >= 0)
				break;
			continue;
		}

		globalHitT = t;
		anyHit = true;

		if (targetIndex >= 0)
			break;
	}

	return anyHit;
}

bool TraceLightsGrid(float3 pos, float3 dir, float3 planeCenter, float3 cameraRight, float3 cameraUp, float fieldWidth, float fieldHeight, int targetIndex, inout float globalHitT)
{
	uint gridLightsX = max(t_config_light_count.x, 1u);
	uint gridLightsY = max(t_config_light_count.y, 1u);

	float halfWidth = fieldWidth * 0.5f;
	float halfHeight = fieldHeight * 0.5f;

	bool anyHit = false;

	for (uint y = 0u; y < gridLightsY; ++y)
	{
		float v = ((float(y) + 0.5f) / float(gridLightsY)) * 2.0f - 1.0f;

		for (uint x = 0u; x < gridLightsX; ++x)
		{
			int currentIndex = int(y * gridLightsX + x);

			if (targetIndex >= 0)
			{
				if (currentIndex < targetIndex)
					continue;
				if (currentIndex > targetIndex)
					return anyHit;
			}

			float u = ((float(x) + 0.5f) / float(gridLightsX)) * 2.0f - 1.0f;

			float3 lightPos = planeCenter
				+ (u * halfWidth) * cameraRight
				+ (v * halfHeight) * cameraUp;

			float3 sphereNormal;
			float t = TestSphereTrace(pos, dir, float4(lightPos, t_smallLightRadius), sphereNormal);
			if (t < 0.0f || t > globalHitT)
			{
				if (targetIndex >= 0)
					return anyHit;
				continue;
			}

			globalHitT = t;
			anyHit = true;

			if (targetIndex >= 0)
				return anyHit;
		}
	}

	return anyHit;
}

bool VisualFieldLightContributions(float3 pos, float3 dir, uint2 screenDims, out float3 lightColor)
{
	float3 cameraRight = mul(float4(1.0f, 0.0f, 0.0f, 0.0f), t_invViewMtx).xyz;
	float3 cameraUp    = mul(float4(0.0f, 1.0f, 0.0f, 0.0f), t_invViewMtx).xyz;
	float3 cameraFwd   = mul(float4(0.0f, 0.0f, -1.0f, 0.0f), t_invViewMtx).xyz;
	float3 camPos      = t_cameraPos;

	float aspect = (screenDims.y > 0u) ? (float(screenDims.x) / float(screenDims.y)) : 1.0f;
	float planeDistance = t_config_light_distance;

	float horizontalFov = VERTICAL_FOV * aspect;
	horizontalFov = (horizontalFov > 3.14159265f) ? radians(horizontalFov) : horizontalFov;
	float halfHorizontalFov = horizontalFov * 0.5f;
	float distanceAbs = abs(planeDistance);
	float fieldWidth = (halfHorizontalFov > 0.0f) ? (2.0f * distanceAbs * tan(halfHorizontalFov)) : 0.0f;
	float fieldHeight = (aspect > 0.0f) ? (fieldWidth / aspect) : fieldWidth;

	float3 planeCenter = camPos + cameraFwd * planeDistance;

	float globalHitT = c_maxT;
	bool anyHit = false;

	if (t_config_only_diagonal)
	{
		anyHit = TraceLightsDiagonal(pos, dir, planeCenter, cameraRight, cameraUp, fieldWidth, fieldHeight, t_only_this_light_by_index, globalHitT);
	}
	else
	{
		anyHit = TraceLightsGrid(pos, dir, planeCenter, cameraRight, cameraUp, fieldWidth, fieldHeight, t_only_this_light_by_index, globalHitT);
	}

	if (!anyHit)
	{
		lightColor = float3(0.0f, 0.0f, 0.0f);
		return false;
	}

	lightColor = BCONF_LIGHT_COLOR * t_smallLightBrightness;
	return true;
}