// Forward Declarations
float TestSphereTrace(in float3 rayPos, in float3 rayDir, in float4 sphere, out float3 normal);

// Fixed configuration constants
static const float3 BCONF_LIGHT_COLOR = float3(1.0f, 1.0f, 1.0f);

bool TraceLightsDiagonal(float3 pos, float3 dir, int targetIndex, inout float globalHitT)
{
	uint diagCount = max(t_config_light_count.x, 1u);

	float2 uv = float2(0.5, 0.5);
	float3 distortedCenterOrigin = getDistortedScreenToWorldPosition(uv);
	float3 distortedCenterDirection = getDistortedScreenToWorldDirection(uv);
	
	float3 diagonalStart = distortedCenterOrigin + distortedCenterDirection * t_config_light_distance;

	uv = float2(1, 1);
	float3 distortedCornerOrigin = getDistortedScreenToWorldPosition(uv);
	float3 distortedCornerDirection = getDistortedScreenToWorldDirection(uv);

	float3 diagonalEnd = distortedCornerOrigin + distortedCornerDirection * t_config_light_distance;

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

bool TraceLightsGrid(float3 pos, float3 dir, int targetIndex, inout float globalHitT)
{
	uint gridLightsX = max(t_config_light_count.x, 1u);
	uint gridLightsY = max(t_config_light_count.y, 1u);

	bool anyHit = false;

	for (uint y = 0u; y < gridLightsY; ++y)
	{
		float v = float(y) / float(gridLightsY - 1);

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

			float u = float(x) / float(gridLightsX - 1);

			float2 uv = float2(u, v);
			uv.y = 1 - uv.y;
			float3 distortedPinholeOrigin = getDistortedScreenToWorldPosition(uv);
			float3 distortedPinholeDirection = getDistortedScreenToWorldDirection(uv);

			// this makes the grid consistent with the distortion in screen space, but not world space
			float3 lightPos = distortedPinholeOrigin + distortedPinholeDirection * t_config_light_distance; 

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

bool VisualFieldLightContributions(float3 pos, float3 dir, out float3 lightColor, out float hitT)
{
	float globalHitT = c_maxT;
	bool anyHit = false;

	if (t_config_only_diagonal)
	{
		anyHit = TraceLightsDiagonal(pos, dir, t_only_this_light_by_index, globalHitT);
	}
	else
	{
		anyHit = TraceLightsGrid(pos, dir, t_only_this_light_by_index, globalHitT);
	}

	if (!anyHit)
	{
		lightColor = float3(0.0f, 0.0f, 0.0f);
		hitT = c_maxT;
		return false;
	}

	lightColor = BCONF_LIGHT_COLOR * t_smallLightBrightness;
	hitT = globalHitT;
	return true;
}