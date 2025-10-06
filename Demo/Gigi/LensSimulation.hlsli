//forward declarations
void drawDebugHelios(inout DebugInfo debugInfo);
float3 GetColorForRay(float3 pos, float3 dir, inout uint RNG, inout Struct_PixelDebugStruct pixelDebug, in uint rayIndex, in uint2 px);
bool VisualFieldLightContributions(float3 pos, float3 dir, uint2 screenDims, out float3 lightColor);

static const float kDebugLineThicknessMultiplier = 0.1f;

// Ray-sphere intersection for a sphere at the origin
// Returns true if intersection exists and writes the two t values to "intersections" (t0 <= t1)
bool sphereRayIntersect(out float2 intersections, in float3 rayOrigin, in float3 rayDir, in float radius)
{
	intersections = float2(0.0f, 0.0f);

	float a = dot(rayDir, rayDir);
	// avoid divide by zero for degenerate rays
	if (a <= 1e-12f)
		return false;

	float b = dot(rayOrigin, rayDir);
	float c = dot(rayOrigin, rayOrigin) - radius * radius;

	float discr = b * b - a * c;
	if (discr < 0.0f)
		return false;

	float sqrtD = sqrt(discr);
	float t0 = (-b - sqrtD) / a;
	float t1 = (-b + sqrtD) / a;

	if (t0 > t1)
	{
		float tmp = t0; t0 = t1; t1 = tmp;
	}

	intersections = float2(t0, t1);
	return true;
}

bool intersect(float radius, float center, Ray ray, out float t, out float3 normal)
{
	t = 0.0f;
	normal = float3(0, 0, 0);

	float2 intersections;
	if (!sphereRayIntersect(intersections, ray.Origin - float3(0, 0, center), ray.Direction, radius))
		return false;
	
	bool useCloserT = (ray.Direction.z > 0) ^ (radius < 0);
	t = useCloserT ? min(intersections.y, intersections.x) : max(intersections.y, intersections.x);
	
	normal = normalize(ray.Origin + t * ray.Direction - float3(0, 0, center));

	// If using the second intersection, we need to flip the normal	
	normal *= useCloserT ? 1.0f : -1.0f;

	return true;
}

float getEtaForWavelength(float n_D, float v_D, float wavelength)
{
	// calculate wavelength-dependent refractive index using Cauchy's equation
	const float lambdaF = 486.1327f; // nm (fraunhofer F)
	const float lambdaD = 589.2938f; // nm (fraunhofer D)
	const float lambdaC = 656.2725f; // nm (fraunhofer C)
	const float B = ((n_D - 1.0f) / v_D) / ((1.0f / (lambdaF * lambdaF)) - (1.0f / (lambdaC * lambdaC)));
	const float A = n_D - B / (lambdaD * lambdaD);
	const float eta = A + B / (wavelength * wavelength);
	return eta;
}



bool traceLensesFromFilm(inout DebugInfo debugInfo, Ray ray, in float wavelength, int elementCount, LensElement lensElements[16], out Ray outRay)
{
	float z = 0.0f; // Start at the film, z = 0
	
	for (int i = elementCount - 1; i >= 0; i--)
	{
		const float curvatureRadius = lensElements[i].curvatureRadius;
		const float thickness = lensElements[i].thickness;
		const float nI_D = lensElements[i].n;
		const float nT_D = i > 0 ? lensElements[i - 1].n : n_air;
		const float vI_D = lensElements[i].v;
		const float vT_D = i > 0 ? lensElements[i - 1].v : v_air;
		const float apertureRadius = lensElements[i].apertureRadius;
		const bool useOpeningTexture = lensElements[i].applyOpeningTexture;

		// choose eta (skip dispersion if wavelength == 0)
		float etaI, etaT;
		if (wavelength == 0.0f)
		{
			etaI = nI_D;
			etaT = nT_D;
		}
		else
		{
			etaI = getEtaForWavelength(nI_D, vI_D, wavelength);
			etaT = getEtaForWavelength(nT_D, vT_D, wavelength);
		}

		z -= thickness;
		float t = 0;
		float3 normal = float3(0, 0, 0);
		
		bool isStop = (curvatureRadius == 0.0f);
		if (isStop)
		{
			if (ray.Direction.z >= 0.0f)
				return false;
			t = (z - ray.Origin.z) / ray.Direction.z;
		}
		else
		{
			float center = z + curvatureRadius;
			if (!intersect(curvatureRadius, center, ray, t, normal))
			{
                if(t_debug_toggle)
				    s2h_drawLine(debugInfo.ui, ray.Origin.zy * debugInfo.scale_debug, (ray.Origin + ray.Direction * 10.0f).zy * debugInfo.scale_debug, float4(1, 0, 0, 1), debugInfo.line_thickness *0.1);
				return false;
			}
		}
		
		float3 hit = ray.Origin + t * ray.Direction;

		// Aperture / stop shape test
		if (useOpeningTexture)
		{
			// Normalize hit to aperture space and cull outside the aperture mask
			float2 p = hit.xy / apertureRadius;
			float2 maskUV = p * 0.5f + 0.5f; // [-1,1] -> [0,1]
			bool outOfBounds = (maskUV.x < 0.0f || maskUV.x > 1.0f || maskUV.y < 0.0f || maskUV.y > 1.0f);
			if (outOfBounds)
			{
				if(t_debug_toggle)
					s2h_drawLine(debugInfo.ui, ray.Origin.zy * debugInfo.scale_debug, (ray.Origin + ray.Direction * 10.0f).zy * debugInfo.scale_debug, float4(1, 0, 0, 1), debugInfo.line_thickness * kDebugLineThicknessMultiplier);
				return false;
			}
			float maskValue = sampleHeliosApertureMask(maskUV);
			if (maskValue < 0.5f)
			{
				if(t_debug_toggle)
					s2h_drawLine(debugInfo.ui, ray.Origin.zy * debugInfo.scale_debug, (ray.Origin + ray.Direction * 10.0f).zy * debugInfo.scale_debug, float4(1, 0, 0, 1), debugInfo.line_thickness * kDebugLineThicknessMultiplier);
				return false;
			}
		}
		else
		{
			// Default circular aperture test
			float r2 = hit.x * hit.x + hit.y * hit.y;
			
			if (r2 > (apertureRadius * apertureRadius)) 
			{
                if(t_debug_toggle)
				    s2h_drawLine(debugInfo.ui, ray.Origin.zy * debugInfo.scale_debug, (ray.Origin + ray.Direction * 10.0f).zy * debugInfo.scale_debug, float4(1, 0, 0, 1), debugInfo.line_thickness * 0.1);
				return false;
			}
		}

		// draw debug line
		s2h_drawLine(debugInfo.ui, ray.Origin.zy * debugInfo.scale_debug, hit.zy * debugInfo.scale_debug, debugInfo.color, debugInfo.line_thickness * 0.1);
		
		ray.Origin = hit;
		
		if (!isStop)
		{
			float eta = etaI / (etaT > 0.0f ? etaT : 1.0f);
			float3 refractDir = refract(ray.Direction, normal, eta);
			// refract returns 0 vector on total internal reflection
			if (dot(refractDir, refractDir) < 1e-8f)
				return false;
			ray.Direction = normalize(refractDir);
		}
	} 
	
	outRay = ray;

	return true;
}

// returns PDF
float ApplyRealisticLensSimulation(inout Ray ray, float wavelength, uint3 px, inout uint RNG, uint2 screenDims, float2 screenPos)
{
	float3 cameraRight = mul(float4(1.0f, 0.0f, 0.0f, 0.0f), t_invViewMtx).xyz;
	float3 cameraUp = mul(float4(0.0f, 1.0f, 0.0f, 0.0f), t_invViewMtx).xyz;
	float3 cameraForward = mul(float4(0.0f, 0.0f, 1.0f, 0.0f), t_invViewMtx).xyz;
	float3 camPos = t_cameraPos;

	// Map normalized screen position ([-1,1]) to film plane coordinates in mm
	float aspect = float(screenDims.x) / float(screenDims.y);
	float sensor_height = min(sony_sensor_height, sony_sensor_width / aspect);
	float sensor_width  = sensor_height * aspect;

	float filmX = -screenPos.x * (sensor_width * 0.5f);
	float filmY = -screenPos.y * (sensor_height * 0.5f);

	// Sample a random point on closest lens element using polar coordinates
    wang_hash(RNG);
	float theta = RandomFloat01(RNG) * 2 * PI;
	float r = sqrt(RandomFloat01(RNG));
	float2 apertureOffset = float2(cos(theta), sin(theta)) * r;
	apertureOffset *= r1_size;
;

	// Construct film-space ray with the sampled aperture offset
	Ray filmRay;
	filmRay.Origin = float3(filmX, filmY, 0.0f);
	float3 target = float3(apertureOffset.x, apertureOffset.y, -d_to_film);
	filmRay.Direction = normalize(target - filmRay.Origin);

	// Debug draw
	DebugInfo debugInfo;
	if(t_debug_toggle) {
		debugInfo.offset = -int2(800, 400);
		debugInfo.px = px;
		debugInfo.scale_debug = 6.0f;
		debugInfo.line_thickness = 5.0f;
		debugInfo.sensor_height = sensor_height;
		debugInfo.sensor_width = sensor_width;
		s2h_init(debugInfo.ui, int2(debugInfo.px.xy) + debugInfo.offset);
		drawDebugHelios(debugInfo);
	}

	// Trace through lens elements
	Ray refracted;
	if (traceLensesFromFilm(debugInfo, filmRay, wavelength, lens_element_count, lens_elements, refracted))
	{
		float mm_to_cm = 1.0f / 10;
		ray.Origin = camPos +
			 (refracted.Origin.x * mm_to_cm) * cameraRight +
			 (refracted.Origin.y * mm_to_cm) * cameraUp -
			 (refracted.Origin.z * mm_to_cm) * cameraForward;

		// match camera position with thin-lens simulation
		ray.Origin += (t_lens_position_shift * mm_to_cm) * cameraForward;

		ray.Direction = normalize(
			 refracted.Direction.x * cameraRight +
			 refracted.Direction.y * cameraUp +
			 refracted.Direction.z * cameraForward);
		return 1.0f;
	}
	return 0.0f;
}

// Shade either the debug bokeh targets or the scene, dividing by PDF as needed
float3 ShadePrimarySample(
	Ray    ray,
	float  PDF,
	uint2  screenDims,
	inout Struct_PixelDebugStruct pixelDebug,
	uint   rayIndex,
	uint3  px,
	inout uint RNG)
{
	if (t_bokeh_test)
	{
		float3 lc;
		bool hit = (PDF > 0.0f) && VisualFieldLightContributions(ray.Origin, ray.Direction, screenDims, lc);
		return hit ? (lc / max(PDF, 1e-6f)) : float3(0.0f, 0.0f, 0.0f);
	}
	else
	{
		return (PDF > 0.0f) ? GetColorForRay(ray.Origin, ray.Direction, RNG, pixelDebug, rayIndex, px.xy) / PDF : float3(0.0f, 0.0f, 0.0f);
	}
}

// Realistic lens, single wavelength (no chromatic splitting)
float3 TraceRealisticMonochrome(
	Ray    baseRay,
	float2 screenPos,
	uint2  screenDims,
	uint3  px,
	inout uint RNG,
	inout Struct_PixelDebugStruct pixelDebug,
	uint   rayIndex)
{
	Ray ray = baseRay;
	float PDF = ApplyRealisticLensSimulation(ray, 0.0f, px, RNG, screenDims, screenPos);
	return ShadePrimarySample(ray, PDF, screenDims, pixelDebug, rayIndex, px, RNG);
}

// Realistic lens with simple RGB chromatic aberration splitting
float3 TraceRealisticChromatic(
	Ray    baseRay,
	float2 screenPos,
	uint2  screenDims,
	uint3  px,
	inout uint RNG,
	inout Struct_PixelDebugStruct pixelDebug,
	uint   rayIndex)
{
	// Representative wavelengths for RGB (nm)
	const float wl_r = 625.0f;
	const float wl_g = 510.0f;
	const float wl_b = 440.0f;

	const float wavelengths[3] = { wl_r, wl_g, wl_b };

	// Bokeh test path
	if (t_bokeh_test)
	{
		float3 rc = 0.0f;
		[unroll]
		for (int i = 0; i < 3; ++i)
		{
			Ray ray = baseRay;
			float PDF = ApplyRealisticLensSimulation(ray, wavelengths[i], px, RNG, screenDims, screenPos);
			float3 lc = 0.0f;
			bool hit = (PDF > 0.0f) && VisualFieldLightContributions(ray.Origin, ray.Direction, screenDims, lc);
			rc[i] = hit ? (lc[i] / max(PDF, 1e-6f)) : 0.0f;
		}
		return rc;
	}

	// Regular shading path: shade each wavelength, keep only the matching color channel
	float3 channelSamples[3];
	[unroll]
	for (int i = 0; i < 3; ++i)
	{
		Ray ray = baseRay;
		float PDF = ApplyRealisticLensSimulation(ray, wavelengths[i], px, RNG, screenDims, screenPos);
		channelSamples[i] = ShadePrimarySample(ray, PDF, screenDims, pixelDebug, rayIndex, px, RNG);
	}
	return float3(channelSamples[0].r, channelSamples[1].g, channelSamples[2].b);
}