// === CAMERA / LENS SPECS ===
// Sony a7R III camera
// Helios 44-2 58mm/f2 lens
static const float sony_sensor_width = 35.9f; // sonya7riii specs Full frame (35.9 x 24 mm) sensor size
static const float sony_sensor_height = 24.0f;
static const float helios_max_focal_length = 58.0f; // mm
static const float helios_aperture_stops[7] = { 2.0f, 2.8f, 4.0f, 5.6f, 8.0f, 11.0f, 16.0f }; // f-stops [0; 6]

// === BIOTAR PATENT DATA ===
//static const float helios_scale = helios_max_focal_length / 100.0f; // helios unit scaling from patent wrong?

// Curvature radii (patent units)
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

// Lens thicknesses (patent units)
static const float biotar_d1_p = 10.75f;
static const float biotar_d2_p = 15.55f;
static const float biotar_d3_p = 5.05f;
static const float biotar_d4_p = 5.05f;
static const float biotar_d5_p = 21.22f;
static const float biotar_d6_p = 13.9f;

// Separations/distances (patent units)
static const float biotar_l1_p = 1.65f;
static const float biotar_l2_p = 18.9f;
static const float biotar_l3_p = 0.97f;

// Refractive indices
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

// measured (mm)
// focal length variation in reality measured: 9mm
// measured 18mm distance from aperture to r10 center
// adapter length 26mm added distance to film
// measured 16mm adapter end to film
// measured 19mm r10 to adapter end
// measured 28mm r10 to adapter end at infinity focus
static const float helios_lens_length_measured = 40.0f; // mm
static const float helios_front_hood_length_mm = 15.0f - 2.0f; // mm; excluding outermost ring
static const float helios_measured_aperture[] = { 3.0f, 8.0f, 12.0f, 15.0f, 17.2f, 19.0f, 20.0f }; // mm

// Lens radius (mm)
static const float helios_lens_r0_mm = 41.5f / 2; // mm measured
static const float helios_lens_r1_mm = 30.0f / 2; // mm measured
static const float helios_lens_r5_mm = 20.0f / 2; // mm measured
static const float helios_lens_r6_mm = 19.0f / 2; // mm measured
static const float helios_lens_r10_mm = 25.0f / 2; // mm measured; should cover r7-r10

// Useful calculations
static const float biotar_lens_length_p = biotar_d1_p + biotar_d2_p + biotar_d3_p + biotar_d4_p + biotar_d5_p + biotar_d6_p + biotar_l1_p + biotar_l2_p + biotar_l3_p; // sum of sep in mm ~53.9632

// Conversions
//static const float patent_to_mm = helios_lens_length_measured / helios_lens_length_p; // mm per patent unit
static const float patent_to_mm = 0.58f;
static const float biotar_lens_length_mm = biotar_lens_length_p * patent_to_mm;
static const float helios_to_biotar_mm = biotar_lens_length_mm / helios_lens_length_measured; // should be close to patent_to_mm but not exactly the same

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
static const float r0_size = helios_lens_r0_mm * helios_to_biotar_mm;
static const float r1_size = helios_lens_r1_mm * helios_to_biotar_mm;
static const float r2_size = helios_lens_r1_mm * helios_to_biotar_mm;
static const float r3_size = helios_lens_r1_mm * helios_to_biotar_mm; // use r1 size for r2 and r3 because of lack of better measurements
static const float r4_size = helios_lens_r1_mm * helios_to_biotar_mm;
static const float r5_size = helios_lens_r5_mm * helios_to_biotar_mm;
static const float r6_size = helios_lens_r6_mm * helios_to_biotar_mm;
static const float r7_size = helios_lens_r10_mm * helios_to_biotar_mm; // use r10 size for r7-r10
static const float r8_size = helios_lens_r10_mm * helios_to_biotar_mm; // use r10 size for r7-r10
static const float r9_size = helios_lens_r10_mm * helios_to_biotar_mm; // use r10 size for r7-r10
static const float r10_size = helios_lens_r10_mm * helios_to_biotar_mm;

static const float d0 = helios_front_hood_length_mm * helios_to_biotar_mm;
static const float d1 = biotar_d1_p * patent_to_mm;
static const float d2 = biotar_d2_p * patent_to_mm;
static const float d3 = biotar_d3_p * patent_to_mm;
static const float d4 = biotar_d4_p * patent_to_mm;
static const float d5 = biotar_d5_p * patent_to_mm;
static const float d6 = biotar_d6_p * patent_to_mm;

static const float l1 = biotar_l1_p * patent_to_mm;
static const float l2 = biotar_l2_p * patent_to_mm;
static const float l3 = biotar_l3_p * patent_to_mm;

// Variables
static const float helios_aperture = helios_measured_aperture[t_aperture_stop] * 0.5f * helios_to_biotar_mm; // directly take measured aperture size (kept helios_ prefix because measured)
static const float d_to_film = 335.598 / (t_focus_distance + 15.4) + 37.887; // mm, by empirically fitted curve

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
	// Helios 44-2 58mm/f2 lens OR BIOTAR 58mm/f2 lens
	// scaled from 100 units to 58mm
	// 			curvature radiii	    separation			n			v			opening radius	
	//createLensElement( no_curv, 	    d0, 		    n_air, 		v_air, 		r0_size, 		    false),
	createLensElement( r1_curv,	        d1, 			biotar_n1, 	biotar_v1,	r1_size, 		    false),
	createLensElement( r2_curv,	        l1, 			n_air, 		v_air, 		r2_size, 		    false),
	createLensElement( r3_curv,	        d2, 			biotar_n2, 	biotar_v2,	r3_size, 		    false),
	createLensElement( r4_curv,	        d3, 			biotar_n3, 	biotar_v3,	r4_size, 		    false),
	createLensElement( r5_curv,	        l2 / 2,		    n_air, 		v_air, 		r5_size, 		    false),
	createLensElement( no_curv, 	    l2 / 2, 		n_air, 		v_air, 		helios_aperture,    true),
	createLensElement( r6_curv, 	    d4, 			biotar_n4, 	biotar_v4,	r6_size, 		    false),
	createLensElement( r7_curv, 	    d5, 			biotar_n5, 	biotar_v5,	r7_size, 		    false),
	createLensElement( r8_curv, 	    l3, 			n_air, 		v_air, 		r8_size, 		    false),
	createLensElement( r9_curv, 	    d6, 			biotar_n6, 	biotar_v6,	r9_size, 		    false),
	createLensElement( r10_curv, 	d_to_film, 			n_air, 		v_air, 		r10_size, 	        false), //11
	createLensElement( 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f ),
	createLensElement( 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f ),
	createLensElement( 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f ),
	createLensElement( 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f ),
	createLensElement( 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f ),
};