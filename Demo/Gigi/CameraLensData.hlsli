// === CAMERA SPECS ===
// Sony a7R III camera
static const float sony_sensor_width = 35.9f;
static const float sony_sensor_height = 24.0f;

// === HELIOS LENS DATA ===
// Helios 44-2 58mm/f2 lens
// measured (mm)
static const float helios_measured_aperture[] = { 3.0f, 8.0f, 12.0f, 15.0f, 17.2f, 19.0f, 20.0f }; // mm

static const float focal_length = 50.0f; // mm
static const float helios_aperture_stops[7] = { 2.0f, 2.8f, 4.0f, 5.6f, 8.0f, 11.0f, 16.0f }; // f-stops [0; 6]

// === BIOTAR PATENT DATA ===
// according to Biotar 1:1.4/5
// curvature radii (patent units)
static const float no_curv = 0.0f; // obviously flat
static const float biotar_r1_p = 83.6f;
static const float biotar_r2_p = 321.0f;
static const float biotar_r3_p = 44.8f;
static const float biotar_r4_p = -1150.0f;
static const float biotar_r5_p = 28.3f;
static const float biotar_r6_p = -38.5f;
static const float biotar_r7_p = 50.5f;
static const float biotar_r8_p = -53.2f;
static const float biotar_r9_p = 106.0f;
static const float biotar_r10_p = -120.0f;

// lens thicknesses (patent units)
static const float biotar_d1_p = 10.75f;
static const float biotar_d2_p = 15.55f;
static const float biotar_d3_p = 5.05f;
static const float biotar_d4_p = 5.05f;
static const float biotar_d5_p = 21.22f;
static const float biotar_d6_p = 13.9f;

// separations (patent units)
static const float biotar_l1_p = 1.65f;
static const float biotar_l2_p = 18.9f;
static const float biotar_l3_p = 0.97f;

// refractive indices
static const float n_air = 1.0f;
static const float biotar_n1 = 1.64238f;
static const float biotar_n2 = 1.62306f;
static const float biotar_n3 = 1.57566f;
static const float biotar_n4 = 1.67270f;
static const float biotar_n5 = 1.64238f;
static const float biotar_n6 = 1.64238f;

// Abbe numbers
static const float v_air = 89.30f;
static const float biotar_v1 = 48.0f;
static const float biotar_v2 = 56.9f;
static const float biotar_v3 = 41.2f;
static const float biotar_v4 = 32.2f;
static const float biotar_v5 = 48.0f;
static const float biotar_v6 = 48.0f;

// measured (pixel)
static const float biotar_a1_px = 455;	// measured as diameter
static const float biotar_a2_px = 406;	// measured as diameter
static const float biotar_a3_px = 270;	// measured as diameter
static const float biotar_a4_px = 362;	// measured as diameter

static const float biotar_l2_1 = 73;
static const float biotar_l2_2 = 46;
static const float biotar_l2_split = biotar_l2_1 / (biotar_l2_1 + biotar_l2_2); // portion of l2 before aperture

// measured (pixel) for pixel to mm conversion
static const float biotar_d1_px = 71;
static const float biotar_d2_px = 100;
static const float biotar_d3_px = 30;
static const float biotar_d4_px = 30;
static const float biotar_d5_px = 134;
static const float biotar_d6_px = 86;

// === CONSTRUCTION OF USABLE LENS DATA ===

// conversions
static const float patent_to_mm = focal_length / 100.0f;
static const float pixel_to_patent =  (biotar_d1_p / biotar_d1_px + biotar_d2_p / biotar_d2_px + biotar_d3_p / biotar_d3_px + biotar_d4_p / biotar_d4_px + biotar_d5_p / biotar_d5_px + biotar_d6_p / biotar_d6_px) / 6.0f; // patent units per pixel, take average for more accuracy

// decide on which measures to use (in mm)
// curvature radii
static const float r1_curv = biotar_r1_p * patent_to_mm;
static const float r2_curv = biotar_r2_p * patent_to_mm;
static const float r3_curv = biotar_r3_p * patent_to_mm;
static const float r4_curv = biotar_r4_p * patent_to_mm;
static const float r5_curv = biotar_r5_p * patent_to_mm;
static const float r6_curv = biotar_r6_p * patent_to_mm;
static const float r7_curv = biotar_r7_p * patent_to_mm;
static const float r8_curv = biotar_r8_p * patent_to_mm;
static const float r9_curv = biotar_r9_p * patent_to_mm;
static const float r10_curv = biotar_r10_p * patent_to_mm;

// lens size radii
static const float a1 = biotar_a1_px * pixel_to_patent * patent_to_mm * 0.5; // radius from diameter
static const float a2 = biotar_a2_px * pixel_to_patent * patent_to_mm * 0.5;
static const float a3 = biotar_a3_px * pixel_to_patent * patent_to_mm * 0.5;
static const float a4 = biotar_a4_px * pixel_to_patent * patent_to_mm * 0.5;

// lens thickness
static const float d1 = biotar_d1_p * patent_to_mm;
static const float d2 = biotar_d2_p * patent_to_mm;
static const float d3 = biotar_d3_p * patent_to_mm;
static const float d4 = biotar_d4_p * patent_to_mm;
static const float d5 = biotar_d5_p * patent_to_mm;
static const float d6 = biotar_d6_p * patent_to_mm;

// lens separations
static const float l1 = biotar_l1_p * patent_to_mm;
static const float l2 = biotar_l2_p * patent_to_mm;
static const float l3 = biotar_l3_p * patent_to_mm;

// variables
static const float aperture = helios_measured_aperture[t_aperture_stop] * (a3 / (helios_measured_aperture[6] * 0.5)) * 0.5f;
static const float d_to_film = 237.480/(t_focus_distance -1.939) + 32.514; // mm, by empirically fitted curve

struct LensElement
{
	float curvatureRadius; // positive = convex toward object, negative = concave toward object, 0 = flat
	float thickness; // distance to next element
	float n; // refractive index of element
	float v; // Abbe number of element
	float apertureRadius; // radius of lens element
	bool applyOpeningTexture; // if a texture should be used to determine the shape
};

LensElement createLensElement(float curvatureRadius, float thickness, float n, float v, float apertureRadius, bool applyOpeningTexture)
{
	LensElement le;
	le.curvatureRadius = curvatureRadius;
	le.thickness = thickness;
	le.n = n;
	le.v = v;
	le.apertureRadius = apertureRadius;
	le.applyOpeningTexture = applyOpeningTexture;
	return le;
}

static const uint lens_element_count = 11;
static LensElement lens_elements[] = {
	// Zeiss BIOTAR 1:1.4 lens with Helios 44-2 aperture shape
	//					| curvature radiii	| separation				|	n		|	v		| opening radius	| use texture
	createLensElement( 	r1_curv,	        d1, 						biotar_n1, 	biotar_v1,	a1, 	    		false),
	createLensElement( 	r2_curv,	        l1, 						n_air, 		v_air, 		a1, 	    		false),
	createLensElement( 	r3_curv,	        d2, 						biotar_n2, 	biotar_v2,	a2, 	   			false),
	createLensElement( 	r4_curv,	        d3, 						biotar_n3, 	biotar_v3,	a2, 	   			false),
	createLensElement( 	r5_curv,	        l2 * biotar_l2_split,		n_air, 		v_air, 		a3, 	   			false),
	createLensElement( 	no_curv, 	        l2 * (1 - biotar_l2_split), n_air, 		v_air, 		aperture,			true),
	createLensElement( 	r6_curv, 	        d4, 						biotar_n4, 	biotar_v4,	a3, 	   			false),
	createLensElement( 	r7_curv, 	        d5, 						biotar_n5, 	biotar_v5,	a4, 	   			false),
	createLensElement( 	r8_curv, 	        l3, 						n_air, 		v_air, 		a4, 	   			false),
	createLensElement( 	r9_curv, 	        d6, 						biotar_n6, 	biotar_v6,	a4, 	   			false),
	createLensElement( 	r10_curv, 	        d_to_film, 					n_air, 		v_air, 		a4, 	   			false), // 11 items
};

