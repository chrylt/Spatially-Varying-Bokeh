// Forward Declarations
float TestSphereTrace(in float3 rayPos, in float3 rayDir, in float4 sphere, out float3 normal);

// Configuration Constants
static const float  BCONF_PLANE_DIST   = 2000.0f;  // Distance of light plane in front of camera
static const float  BCONF_FIELD_WIDTH  = 1100.0f;  // Width of light distribution plane (world units)
static const int    BCONF_LIGHTS_X     = 11;       // Grid resolution (horizontal)
static const int    BCONF_LIGHTS_Y     = 7;        // Grid resolution (vertical)

// Derived Constants
static const int    BCONF_LIGHT_COUNT  = BCONF_LIGHTS_X * BCONF_LIGHTS_Y;

bool VisualFieldLightContributions(float3 pos, float3 dir, uint2 screenDims, out float3 lightColor)
{
	// Camera basis
	float3 cameraRight = mul(float4(1.0f, 0.0f, 0.0f, 0.0f), t_invViewMtx).xyz;
	float3 cameraUp    = mul(float4(0.0f, 1.0f, 0.0f, 0.0f), t_invViewMtx).xyz;
	float3 cameraFwd   = mul(float4(0.0f, 0.0f, -1.0f, 0.0f), t_invViewMtx).xyz;
	float3 camPos      = t_cameraPos;

	// Plane extents
	float aspect      = float(screenDims.x) / float(screenDims.y);
	float fieldHeight = BCONF_FIELD_WIDTH / aspect;
	float3 planeCenter = camPos + cameraFwd * BCONF_PLANE_DIST;

	float globalHitT = c_maxT;
	bool anyHit = false;

	for (int y = 0; y < BCONF_LIGHTS_Y; ++y)
	{
		for (int x = 0; x < BCONF_LIGHTS_X; ++x)
		{
			float u = ((float(x) + 0.5f) / float(BCONF_LIGHTS_X)) * 2.0f - 1.0f; // [-1,1]
			float v = ((float(y) + 0.5f) / float(BCONF_LIGHTS_Y)) * 2.0f - 1.0f; // [-1,1]

			// Position on plane
			float3 lightPos = planeCenter
				+ (u * (BCONF_FIELD_WIDTH  * 0.5f)) * cameraRight
				+ (v * (fieldHeight     * 0.5f)) * cameraUp;

			// Ray-sphere (light) test
			float3 sphereNormal;
			float t = TestSphereTrace(pos, dir, float4(lightPos, t_smallLightRadius), sphereNormal);
			if (t < 0.0f || t > globalHitT)
				continue;

			globalHitT = t;
			anyHit = true;
		}
	}

	if (!anyHit)
	{
		lightColor = float3(0.0f, 0.0f, 0.0f);
		return false;
	}

	lightColor = float3(1.0f, 1.0f, 1.0f) * t_smallLightBrightness;
	return true;
}