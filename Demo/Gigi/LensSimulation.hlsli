// forward declarations
float3 GetColorForRay(float3 pos, float3 dir, inout uint RNG, inout PixelInfo pixelInfo, in uint rayIndex, in uint2 px);
bool VisualFieldLightContributions(float3 pos, float3 dir, out float3 lightColor);

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

	// if using the second intersection, we need to flip the normal	
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

bool traceLensesFromFilm(Ray ray, in float wavelength, int elementCount, LensElement lensElements[11], out Ray outRay)
{
	float z = 0.0f; // start at the film, z = 0
	
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
				return false;
			}
		}
		
		float3 hit = ray.Origin + t * ray.Direction;

		// aperture mask test
		if (useOpeningTexture)
		{
			// normalize hit to aperture space and cull outside the aperture mask
			float2 p = hit.xy / apertureRadius;
			float2 maskUV = p * 0.5f + 0.5f; // [-1,1] -> [0,1], negated to compensate for film coordinate inversion
			bool outOfBounds = (maskUV.x < 0.0f || maskUV.x > 1.0f || maskUV.y < 0.0f || maskUV.y > 1.0f);
			if (outOfBounds)
			{
				return false;
			}
			float maskValue = sampleHeliosApertureMask(maskUV);
			if (maskValue < 0.5f)
			{
				return false;
			}
		}
		else
		{
			// default circular aperture test
			float r2 = hit.x * hit.x + hit.y * hit.y;
			
			if (r2 > (apertureRadius * apertureRadius)) 
			{
				return false;
			}
		}

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
float ApplyRealisticLensSimulation(out Ray ray, float wavelength, inout uint RNG, uint2 screenDims, float2 screenPos)
{
	float3 cameraRight = mul(float4(1.0f, 0.0f, 0.0f, 0.0f), t_invViewMtx).xyz;
	float3 cameraUp = mul(float4(0.0f, 1.0f, 0.0f, 0.0f), t_invViewMtx).xyz;
	float3 cameraForward = mul(float4(0.0f, 0.0f, 1.0f, 0.0f), t_invViewMtx).xyz;
	float3 camPos = t_cameraPos;

	// map normalized screen position ([-1,1]) to film plane coordinates in mm
	float aspect = float(screenDims.x) / float(screenDims.y);
	float sensor_height = min(sony_sensor_height, sony_sensor_width / aspect);
	float sensor_width  = sensor_height * aspect;

	float filmX = -screenPos.x * (sensor_width * 0.5f);
	float filmY = -screenPos.y * (sensor_height * 0.5f);

	// sample a random point on closest lens element using polar coordinates
    wang_hash(RNG);
	float theta = RandomFloat01(RNG) * 2 * PI;
	float r = sqrt(RandomFloat01(RNG));
	float2 apertureOffset = float2(cos(theta), sin(theta)) * r;
	apertureOffset *= a4; // scale to aperture radius

	// construct film-space ray with the sampled aperture offset
	Ray filmRay;
	filmRay.Origin = float3(filmX, filmY, 0.0f);
	float3 target = float3(apertureOffset.x, apertureOffset.y, -d_to_film);
	filmRay.Direction = normalize(target - filmRay.Origin);

	// trace ray through lens elements
	Ray refracted;
	if (traceLensesFromFilm(filmRay, wavelength, lens_element_count, lens_elements, refracted))
	{
		float mm_to_cm = 1.0f / 10;
		ray.Origin = camPos +
			 (refracted.Origin.x * mm_to_cm) * cameraRight +
			 (refracted.Origin.y * mm_to_cm) * cameraUp +
			 (refracted.Origin.z * mm_to_cm) * cameraForward;

		ray.Direction = normalize(
			 refracted.Direction.x * cameraRight +
			 refracted.Direction.y * cameraUp +
			 refracted.Direction.z * cameraForward);
		return 1.0f;
	}
	return 0.0f;
}

// Scene shading divides by PDF to keep Monte Carlo estimator unbiased
float3 ShadeSceneSample(
	Ray    ray,
	float  PDF,
	inout PixelInfo pixelInfo,
	uint   rayIndex,
	uint3  px,
	inout uint RNG)
{
	return (PDF > 0.0f) ? GetColorForRay(ray.Origin, ray.Direction, RNG, pixelInfo, rayIndex, px.xy) / PDF : float3(0.0f, 0.0f, 0.0f);
}

float3 ShadeVisualFieldSample(
	Ray   ray,
	float PDF,
	inout PixelInfo pixelInfo)
{
	float3 lightColor = float3(0.0f, 0.0f, 0.0f);
	float hitT = c_maxT;
	bool hit = (PDF > 0.0f) && VisualFieldLightContributions(ray.Origin, ray.Direction, lightColor, hitT);
	pixelInfo.HitT = hitT;  // c_maxT on miss, actual distance on hit
	return hit ? (lightColor / max(PDF, 1e-6f)) : float3(0.0f, 0.0f, 0.0f);
}

// Realistic lens, single wavelength (no chromatic splitting)
float3 TraceRealisticMonochrome(
	float2 screenPos,
	uint2  screenDims,
	uint3  px,
	inout uint RNG,
	inout PixelInfo pixelInfo,
	uint   rayIndex,
	bool   bokehView)
{
	Ray ray;
	float PDF = ApplyRealisticLensSimulation(ray, 0.0f, RNG, screenDims, screenPos);
	return bokehView
		? ShadeVisualFieldSample(ray, PDF, pixelInfo)
		: ShadeSceneSample(ray, PDF, pixelInfo, rayIndex, px, RNG);
}

// Realistic lens with simple RGB chromatic aberration splitting
float3 TraceRealisticChromatic(
	float2 screenPos,
	uint2  screenDims,
	uint3  px,
	inout uint RNG,
	inout PixelInfo pixelInfo,
	uint   rayIndex,
	bool   bokehView)
{
	// representative wavelengths for RGB (nm)
	const float wl_r = 625.0f;
	const float wl_g = 510.0f;
	const float wl_b = 440.0f;

	const float wavelengths[3] = { wl_r, wl_g, wl_b };

	if (bokehView)
	{
		float3 rc = 0.0f;
		float finalHitT = c_maxT;
		[unroll]
		for (int i = 0; i < 3; ++i)
		{
			Ray ray;
			float PDF = ApplyRealisticLensSimulation(ray, wavelengths[i], RNG, screenDims, screenPos);
			float3 lc = 0.0f;
			float hitT = c_maxT;
			bool hit = (PDF > 0.0f) && VisualFieldLightContributions(ray.Origin, ray.Direction, lc, hitT);
			if (hit)
				finalHitT = hitT;
			rc[i] = hit ? (lc[i] / max(PDF, 1e-6f)) : 0.0f;
		}
		pixelInfo.HitT = finalHitT;  // c_maxT if all miss, actual distance if any hit
		return rc;
	}

	// regular shading path: shade each wavelength, keep only the matching color channel
	float3 channelSamples[3];
	[unroll]
	for (int i = 0; i < 3; ++i)
	{
		Ray ray;
		float PDF = ApplyRealisticLensSimulation(ray, wavelengths[i], RNG, screenDims, screenPos);
		channelSamples[i] = ShadeSceneSample(ray, PDF, pixelInfo, rayIndex, px, RNG);
	}
	return float3(channelSamples[0].r, channelSamples[1].g, channelSamples[2].b);
}