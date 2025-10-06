

// Colors
static const float4 kFilmColor           = float4(0.4, 0.4, 0.0, 1.0);
static const float4 kLensColor           = float4(0.0, 0.5, 0.5, 1.0);
static const float4 kApertureColor       = float4(0.0, 0.0, 0.0, 1.0);
static const float4 kTextColor           = float4(0.0, 0.0, 0.0, 1.0);
static const float4 kBackgroundColor     = float4(1.0, 1.0, 1.0, 1.0);
static const float4 kHorizontalLineColor = float4(0.6, 0.6, 0.6, 1.0);
static const float4 kStopTickColor       = float4(0.2, 0.2, 0.2, 1.0);

// 2D debug space uses x=z, y=y from lens space; apply scale here
float2 ToDebug(float z, float y, float scaleVal)
{
	return float2(z * scaleVal, y * scaleVal);
}

void UIPrintHeader(inout DebugInfo di, float filmHeightMM)
{
	s2h_setCursor(di.ui, float2(10, 10) + di.offset);
	s2h_setScale(di.ui, 2.0);
	di.ui.textColor = kTextColor;

	// FilmH
	s2h_printTxt(di.ui, _F, _i, _l, _m, _H);
	s2h_printLF(di.ui);
	s2h_printFloat(di.ui, filmHeightMM);
	s2h_printLF(di.ui);
	s2h_printLF(di.ui);

	// LensL
	s2h_printTxt(di.ui, _L, _e, _n, _s, _L);
	s2h_printLF(di.ui);
	s2h_printFloat(di.ui, helios_lens_length_measured);
	s2h_printLF(di.ui);
	s2h_printLF(di.ui);

	// LensD2Film
	s2h_printTxt(di.ui, _L, _e, _n, _s, _D);
	s2h_printTxt(di.ui, _2, _F, _i, _l, _m);
	s2h_printLF(di.ui);
	s2h_printFloat(di.ui, d_to_film);
	s2h_printLF(di.ui);
}

void DrawFilmPlane(inout DebugInfo di, float filmHeightMM)
{
	float S = di.scale_debug;
	const float2 filmCenter = float2(0, 0);

	// Vertical film line at z=0 from -H/2 to +H/2 in y
	s2h_drawLine(
		di.ui,
		filmCenter + float2(0,  filmHeightMM * S * 0.5f),
		filmCenter - float2(0,  filmHeightMM * S * 0.5f),
		kFilmColor,
		di.line_thickness
	);
}

void DrawAxisBaseline(inout DebugInfo di)
{
	float S = di.scale_debug;

	// Horizontal line from -d_to_film to -(lens_length + d_to_film)
	float2 a = ToDebug(-d_to_film, 0, S);
	float2 b = ToDebug(-(helios_lens_length_measured + d_to_film), 0, S);
	s2h_drawLine(di.ui, a, b, kHorizontalLineColor, di.line_thickness);
}

void DrawApertureStop(inout DebugInfo di, float z, float apertureRadius)
{
	float S = di.scale_debug;
	const float2 center = ToDebug(z, 0, S);

	// Vertical aperture segment
	s2h_drawLine(
		di.ui,
		center - float2(0, apertureRadius * S),
		center + float2(0, apertureRadius * S),
		kApertureColor,
		di.line_thickness
	);
}

void DrawStopTicks(inout DebugInfo di, float z, float apertureRadius)
{
	float S = di.scale_debug;
	float2 center = ToDebug(z, 0, S);
	float th = di.line_thickness;

	// Small horizontal ticks at top and bottom of the clear aperture
	float2 topA = center + float2(-th,  apertureRadius * S);
	float2 topB = center + float2(+th,  apertureRadius * S);
	float2 botA = center + float2(-th, -apertureRadius * S);
	float2 botB = center + float2(+th, -apertureRadius * S);

	s2h_drawLine(di.ui, topA, topB, kStopTickColor, di.line_thickness);
	s2h_drawLine(di.ui, botA, botB, kStopTickColor, di.line_thickness);
}

void DrawLensSurfaceArc(inout DebugInfo di, float z, float curvatureRadius, float apertureRadius)
{
	float S = di.scale_debug;

	// Arc center on z-axis shifted by curvature radius
	float2 arcCenter = ToDebug(z + curvatureRadius, 0, S);

	float startAngle = 0.0f;
	float endAngle   = 0.0f;

	float angle = atan(apertureRadius / abs(curvatureRadius));

	if (curvatureRadius > 0.0f)
	{
		startAngle = PI - angle;
		endAngle   = PI + angle;
	}
	else
	{
		endAngle   = angle;
		startAngle = 2.0f * PI - angle;
	}

	s2h_drawArc(di.ui, arcCenter, abs(curvatureRadius * S), kLensColor, di.line_thickness, startAngle, endAngle);
}

void DrawLensStack(inout DebugInfo di)
{
	float z = 0.0f; // start at film (z=0), go towards -z through the lens
	for (int i = lens_element_count - 1; i >= 0; --i)
	{
		const float curvatureRadius = lens_elements[i].curvatureRadius;
		const float thickness       = lens_elements[i].thickness;
		const float apertureRadius  = lens_elements[i].apertureRadius;

		z -= thickness;

		const bool isStop = (curvatureRadius == 0.0f);
		if (isStop)
		{
			DrawApertureStop(di, z, apertureRadius);
		}
		else
		{
			DrawStopTicks(di, z, apertureRadius);
			DrawLensSurfaceArc(di, z, curvatureRadius, apertureRadius);
		}
	}
}

void DrawExampleRays(inout DebugInfo di, float filmHeightMM)
{
	// Example rays along film diagonal for quick visual sanity checks
	float4 colors[] = {
		float4(1, 0, 0, 1),
		float4(0, 1, 0, 1),
		float4(0, 0, 1, 1)
	};

	Ray filmRay;
	filmRay.Origin = float3(0, filmHeightMM / 2.0f, 0);
	float2 apertureOffset = float2(0.0f, 0.5f) * r3_size;

	float3 target = float3(apertureOffset.x, apertureOffset.y, -d_to_film);
	filmRay.Direction = normalize(target - filmRay.Origin);

	Ray refracted;
	/*di.color = float4(1, 0.686, 0, 1);
	const float lambdaD = 589.2938f;
	traceLensesFromFilm(di, filmRay, lambdaD, lens_element_count, lens_elements, refracted);*/

	di.color = colors[0];
	traceLensesFromFilm(di, filmRay, 625, lens_element_count, lens_elements, refracted);

	di.color = colors[1];
	traceLensesFromFilm(di, filmRay, 500, lens_element_count, lens_elements, refracted);

	di.color = colors[2];
	traceLensesFromFilm(di, filmRay, 450, lens_element_count, lens_elements, refracted);
}

void drawDebugHelios(inout DebugInfo debugInfo)
{
	const float filmHeightMM = debugInfo.sensor_height;

	// Film and UI header
	DrawFilmPlane(debugInfo, filmHeightMM);
	UIPrintHeader(debugInfo, filmHeightMM);

	// Baseline and lens stack
	DrawAxisBaseline(debugInfo);
	DrawLensStack(debugInfo);

	// Example rays
	DrawExampleRays(debugInfo, filmHeightMM);

	// Composite over background
	float3 linearColor = kBackgroundColor.rgb * (1.0 - debugInfo.ui.dstColor.a) + debugInfo.ui.dstColor.rgb;
	DebugTex[debugInfo.px.xy] = float4(s2h_accurateLinearToSRGB(linearColor.rgb), 1.0);
}