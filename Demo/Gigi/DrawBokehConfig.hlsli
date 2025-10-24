// Forward Declarations
float TestSphereTrace(in float3 rayPos, in float3 rayDir, in float4 sphere, out float3 normal);

// Configuration Constants
#define BCONF_USE_DIAGONAL 1                 // Set to 1 to render diagonal layout instead of grid
static const float  BCONF_PLANE_DIST   = 2000.0f;  // Distance of light plane in front of camera
static const float  BCONF_FIELD_WIDTH  = 1100.0f;  // Width of light distribution plane (world units)
static const int    BCONF_LIGHTS_X     = 11;       // Grid resolution (horizontal)
static const int    BCONF_LIGHTS_Y     = 7;        // Grid resolution (vertical)
static const int    BCONF_DIAG_LIGHTS  = 8;       // Number of lights along the diagonal (used when BCONF_USE_DIAGONAL == 1)

// Derived Constants
#if BCONF_USE_DIAGONAL
static const int    BCONF_LIGHT_COUNT  = (BCONF_DIAG_LIGHTS > 0) ? BCONF_DIAG_LIGHTS : 1;
#else
static const int    BCONF_LIGHT_COUNT  = BCONF_LIGHTS_X * BCONF_LIGHTS_Y;
#endif

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

#if BCONF_USE_DIAGONAL
	float halfWidth = BCONF_FIELD_WIDTH * 0.5f;
	float halfHeight = fieldHeight * 0.5f;
	float3 diagonalEnd = planeCenter + cameraRight * halfWidth + cameraUp * halfHeight;
	int diagCount = (BCONF_DIAG_LIGHTS > 0) ? BCONF_DIAG_LIGHTS : 1;

	for (int i = 0; i < diagCount; ++i)
	{
		float fraction = (diagCount > 1) ? (float(i) / float(diagCount - 1)) : 0.0f;
		float3 lightPos = lerp(planeCenter, diagonalEnd, fraction);

		float3 sphereNormal;
		float t = TestSphereTrace(pos, dir, float4(lightPos, t_smallLightRadius), sphereNormal);
		if (t < 0.0f || t > globalHitT)
			continue;

		globalHitT = t;
		anyHit = true;
	}
#else
	for (int y = 0; y < BCONF_LIGHTS_Y; ++y)
	{
		for (int x = 0; x < BCONF_LIGHTS_X; ++x)
		{
			float u = ((float(x) + 0.5f) / float(BCONF_LIGHTS_X)) * 2.0f - 1.0f; // [-1,1]
			float v = ((float(y) + 0.5f) / float(BCONF_LIGHTS_Y)) * 2.0f - 1.0f; // [-1,1]

			float3 lightPos = planeCenter
				+ (u * (BCONF_FIELD_WIDTH  * 0.5f)) * cameraRight
				+ (v * (fieldHeight     * 0.5f)) * cameraUp;

			float3 sphereNormal;
			float t = TestSphereTrace(pos, dir, float4(lightPos, t_smallLightRadius), sphereNormal);
			if (t < 0.0f || t > globalHitT)
				continue;

			globalHitT = t;
			anyHit = true;
		}
	}
#endif

	if (!anyHit)
	{
		lightColor = float3(0.0f, 0.0f, 0.0f);
		return false;
	}

	lightColor = float3(1.0f, 1.0f, 1.0f) * t_smallLightBrightness;
	return true;
}