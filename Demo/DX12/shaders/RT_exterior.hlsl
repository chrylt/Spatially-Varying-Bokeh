///////////////////////////////////////////////////////////////////////////////
//     Filter-Adapted Spatio-Temporal Sampling With General Distributions    //
//        Copyright (c) 2024 Electronic Arts Inc. All rights reserved.       //
///////////////////////////////////////////////////////////////////////////////

// Unnamed technique, shader RayGen


struct LensRNG
{
    static const int UniformCircleWhite_PCG = 0;
    static const int UniformCircleWhite = 1;
    static const int UniformCircleBlue = 2;
    static const int UniformHexagonWhite = 3;
    static const int UniformHexagonBlue = 4;
    static const int UniformHexagonICDF_White = 5;
    static const int UniformHexagonICDF_Blue = 6;
    static const int UniformStarWhite = 7;
    static const int UniformStarBlue = 8;
    static const int UniformStarICDF_White = 9;
    static const int UniformStarICDF_Blue = 10;
    static const int NonUniformStarWhite = 11;
    static const int NonUniformStarBlue = 12;
    static const int NonUniformStar2White = 13;
    static const int NonUniformStar2Blue = 14;
    static const int LKCP6White = 15;
    static const int LKCP6Blue = 16;
    static const int LKCP204White = 17;
    static const int LKCP204Blue = 18;
    static const int LKCP204ICDF_White = 19;
    static const int LKCP204ICDF_Blue = 20;
    static const int bokeh = 21;
};

struct BokehConfigState
{
    static const int NoDoF = 0;
    static const int ThinLens = 1;
    static const int RealisticLens = 2;
};

struct PixelJitterType
{
    static const int None = 0;
    static const int PerPixel = 1;
    static const int Global = 2;
};

struct MaterialSets
{
    static const int None = 0;
    static const int Interior = 1;
    static const int Exterior = 2;
};

struct NoiseTexExtends
{
    static const int None = 0;
    static const int White = 1;
    static const int Shuffle1D = 2;
    static const int Shuffle1DHilbert = 3;
};

struct Struct_VBStruct
{
    float3 position;
    float3 normal;
    float4 tangent;
    float2 UV;
    uint materialID;
};

struct Struct_PixelDebugStruct
{
    float2 MousePos;
    uint MaterialID;
    float HitT;
    float3 WorldPos;
    float centerDepth;
    float centerSize;
    float tot;
};

struct Struct__RayGenCB
{
    uint Accumulate;
    uint AlbedoMode;
    float AlbedoModeAlbedoMultiplier;
    float _padding0;
    float2 AnamorphicScaling;
    uint Animate;
    float ApertureRadius;
    int BokehConfigMode;
    float3 CameraPos;
    uint2 ConfigLightCount;
    float ConfigLightDistance;
    float ConfigLightFieldWidth;
    uint ConfigOnlyDiagonal;
    uint DebugToggle;
    float DepthNearPlane;
    float FocalLength;
    float FocusDistance;
    uint FrameIndex;
    uint HeliosApertureStop;
    float _padding1;
    float4x4 InvViewMtx;
    float4x4 InvViewProjMtx;
    uint JitterNoiseTextures;
    int JitterPixels;
    int LensRNGExtend;
    int LensRNGSource;
    float MaterialEmissiveMultiplier;
    int MaterialSet;
    float2 _padding2;
    float4 MouseState;
    uint NoImportanceSampling;
    uint NumBounces;
    float2 _padding3;
    float3 OcclusionSettings;
    int OnlyThisLightByIndex;
    float2 PetzvalScaling;
    float RayPosNormalNudge;
    uint RenderBokehConfig;
    uint RenderLensSimulationDoF;
    uint RenderPinhole;
    uint RenderThinLensDoF;
    uint SamplesPerPixelPerFrame;
    float ShiftHeliosPosition;
    float SkyBrightness;
    float2 _padding4;
    float3 SkyColor;
    float SmallLightBrightness;
    float SmallLightRadius;
    float3 SmallLightsColor;
    uint SmallLightsColorful;
    uint ToggleChromaticAberration;
    float2 _padding5;
};

SamplerState PointWrapSampler : register(s0);
SamplerState PointClampSampler : register(s1);
RWTexture2D<float4> PinholeOut : register(u0);
RWTexture2D<float4> ThinlensOut : register(u1);
RWTexture2D<float4> LensSimulationOut : register(u2);
RWTexture2D<float4> BokehConfigOut : register(u3);
RWTexture2D<float> LinearDepth : register(u4);
RaytracingAccelerationStructure Scene : register(t0);
StructuredBuffer<Struct_VBStruct> VertexBuffer : register(t1);
RWStructuredBuffer<Struct_PixelDebugStruct> PixelDebug : register(u5);
RWTexture2D<float4> DebugLensOut : register(u6);
Texture2D<float> _loadedTexture_0 : register(t2);
Texture2D<float> _loadedTexture_1 : register(t3);
Texture2D<float> _loadedTexture_2 : register(t4);
Texture2D<float> _loadedTexture_3 : register(t5);
Texture2D<float> _loadedTexture_4 : register(t6);
Texture2D<float> _loadedTexture_5 : register(t7);
Texture2D<float> _loadedTexture_6 : register(t8);
Texture2D<float4> _loadedTexture_7 : register(t9);
Texture2D<float4> _loadedTexture_8 : register(t10);
Texture2D<float4> _loadedTexture_9 : register(t11);
Texture2D<float4> _loadedTexture_10 : register(t12);
Texture2D<float4> _loadedTexture_11 : register(t13);
Texture2D<float4> _loadedTexture_12 : register(t14);
Texture2D<float4> _loadedTexture_13 : register(t15);
Texture2D<float4> _loadedTexture_14 : register(t16);
Texture2D<float4> _loadedTexture_15 : register(t17);
Texture2D<float4> _loadedTexture_16 : register(t18);
Texture2D<float4> _loadedTexture_17 : register(t19);
Texture2D<float4> _loadedTexture_18 : register(t20);
Texture2D<float4> _loadedTexture_19 : register(t21);
Texture2D<float4> _loadedTexture_20 : register(t22);
Texture2D<float4> _loadedTexture_21 : register(t23);
Texture2D<float4> _loadedTexture_22 : register(t24);
Texture2D<float4> _loadedTexture_23 : register(t25);
Texture2D<float4> _loadedTexture_24 : register(t26);
Texture2D<float4> _loadedTexture_25 : register(t27);
Texture2D<float4> _loadedTexture_26 : register(t28);
Texture2D<float4> _loadedTexture_27 : register(t29);
Texture2D<float> _loadedTexture_28 : register(t30);
Texture2D<float4> _loadedTexture_29 : register(t31);
Texture2D<float4> _loadedTexture_30 : register(t32);
Texture2D<float4> _loadedTexture_31 : register(t33);
Texture2D<float4> _loadedTexture_32 : register(t34);
Texture2D<float4> _loadedTexture_33 : register(t35);
Texture2D<float4> _loadedTexture_34 : register(t36);
Texture2D<float4> _loadedTexture_35 : register(t37);
Texture2D<float4> _loadedTexture_36 : register(t38);
Texture2D<float4> _loadedTexture_37 : register(t39);
Texture2D<float4> _loadedTexture_38 : register(t40);
Texture2D<float4> _loadedTexture_39 : register(t41);
Texture2D<float4> _loadedTexture_40 : register(t42);
Texture2D<float> _loadedTexture_41 : register(t43);
Texture2D<float4> _loadedTexture_42 : register(t44);
Texture2D<float4> _loadedTexture_43 : register(t45);
Texture2D<float4> _loadedTexture_44 : register(t46);
Texture2D<float4> _loadedTexture_45 : register(t47);
Texture2D<float4> _loadedTexture_46 : register(t48);
Texture2D<float4> _loadedTexture_47 : register(t49);
Texture2D<float4> _loadedTexture_48 : register(t50);
Texture2D<float4> _loadedTexture_49 : register(t51);
Texture2D<float4> _loadedTexture_50 : register(t52);
Texture2D<float4> _loadedTexture_51 : register(t53);
Texture2D<float4> _loadedTexture_52 : register(t54);
Texture2D<float4> _loadedTexture_53 : register(t55);
Texture2D<float4> _loadedTexture_54 : register(t56);
Texture2D<float4> _loadedTexture_55 : register(t57);
Texture2D<float4> _loadedTexture_56 : register(t58);
Texture2D<float4> _loadedTexture_57 : register(t59);
Texture2D<float4> _loadedTexture_58 : register(t60);
Texture2D<float4> _loadedTexture_59 : register(t61);
Texture2D<float4> _loadedTexture_60 : register(t62);
Texture2D<float4> _loadedTexture_61 : register(t63);
Texture2D<float4> _loadedTexture_62 : register(t64);
Texture2D<float4> _loadedTexture_63 : register(t65);
Texture2D<float4> _loadedTexture_64 : register(t66);
Texture2D<float> _loadedTexture_65 : register(t67);
Texture2D<float4> _loadedTexture_66 : register(t68);
Texture2D<float> _loadedTexture_67 : register(t69);
Texture2D<float4> _loadedTexture_68 : register(t70);
Texture2D<float4> _loadedTexture_69 : register(t71);
Texture2D<float> _loadedTexture_70 : register(t72);
Texture2D<float4> _loadedTexture_71 : register(t73);
Texture2D<float> _loadedTexture_72 : register(t74);
Texture2D<float4> _loadedTexture_73 : register(t75);
Texture2D<float4> _loadedTexture_74 : register(t76);
Texture2D<float> _loadedTexture_75 : register(t77);
Texture2D<float4> _loadedTexture_76 : register(t78);
Texture2D<float> _loadedTexture_77 : register(t79);
Texture2D<float4> _loadedTexture_78 : register(t80);
Texture2D<float> _loadedTexture_79 : register(t81);
Texture2D<float4> _loadedTexture_80 : register(t82);
Texture2D<float4> _loadedTexture_81 : register(t83);
Texture2D<float4> _loadedTexture_82 : register(t84);
Texture2D<float4> _loadedTexture_83 : register(t85);
Texture2D<float4> _loadedTexture_84 : register(t86);
Texture2D<float> _loadedTexture_85 : register(t87);
Texture2D<float4> _loadedTexture_86 : register(t88);
Texture2D<float4> _loadedTexture_87 : register(t89);
Texture2D<float4> _loadedTexture_88 : register(t90);
Texture2D<float4> _loadedTexture_89 : register(t91);
Texture2D<float4> _loadedTexture_90 : register(t92);
Texture2D<float4> _loadedTexture_91 : register(t93);
Texture2D<float4> _loadedTexture_92 : register(t94);
Texture2D<float4> _loadedTexture_93 : register(t95);
Texture2D<float4> _loadedTexture_94 : register(t96);
Texture2D<float4> _loadedTexture_95 : register(t97);
Texture2D<float4> _loadedTexture_96 : register(t98);
Texture2D<float4> _loadedTexture_97 : register(t99);
Texture2D<float4> _loadedTexture_98 : register(t100);
Texture2D<float4> _loadedTexture_99 : register(t101);
Texture2D<float4> _loadedTexture_100 : register(t102);
Texture2D<float4> _loadedTexture_101 : register(t103);
Texture2D<float4> _loadedTexture_102 : register(t104);
Texture2D<float4> _loadedTexture_103 : register(t105);
Texture2D<float4> _loadedTexture_104 : register(t106);
Texture2D<float4> _loadedTexture_105 : register(t107);
Texture2D<float4> _loadedTexture_106 : register(t108);
Texture2D<float4> _loadedTexture_107 : register(t109);
Texture2D<float4> _loadedTexture_108 : register(t110);
Texture2D<float4> _loadedTexture_109 : register(t111);
Texture2D<float4> _loadedTexture_110 : register(t112);
Texture2D<float4> _loadedTexture_111 : register(t113);
Texture2D<float4> _loadedTexture_112 : register(t114);
Texture2D<float4> _loadedTexture_113 : register(t115);
Texture2D<float4> _loadedTexture_114 : register(t116);
Texture2D<float4> _loadedTexture_115 : register(t117);
Texture2D<float4> _loadedTexture_116 : register(t118);
Texture2D<float4> _loadedTexture_117 : register(t119);
Texture2D<float4> _loadedTexture_118 : register(t120);
Texture2D<float4> _loadedTexture_119 : register(t121);
Texture2D<float4> _loadedTexture_120 : register(t122);
Texture2D<float4> _loadedTexture_121 : register(t123);
Texture2D<float4> _loadedTexture_122 : register(t124);
Texture2D<float4> _loadedTexture_123 : register(t125);
Texture2D<float4> _loadedTexture_124 : register(t126);
Texture2D<float4> _loadedTexture_125 : register(t127);
Texture2D<float4> _loadedTexture_126 : register(t128);
Texture2D<float4> _loadedTexture_127 : register(t129);
Texture2D<float4> _loadedTexture_128 : register(t130);
Texture2D<float4> _loadedTexture_129 : register(t131);
Texture2D<float> _loadedTexture_130 : register(t132);
Texture2D<float4> _loadedTexture_131 : register(t133);
Texture2D<float4> _loadedTexture_132 : register(t134);
Texture2D<float4> _loadedTexture_133 : register(t135);
Texture2D<float4> _loadedTexture_134 : register(t136);
Texture2D<float4> _loadedTexture_135 : register(t137);
Texture2D<float4> _loadedTexture_136 : register(t138);
Texture2D<float4> _loadedTexture_137 : register(t139);
Texture2D<float4> _loadedTexture_138 : register(t140);
Texture2D<float4> _loadedTexture_139 : register(t141);
Texture2D<float4> _loadedTexture_140 : register(t142);
Texture2D<float4> _loadedTexture_141 : register(t143);
Texture2D<float4> _loadedTexture_142 : register(t144);
Texture2D<float> _loadedTexture_143 : register(t145);
Texture2D<float4> _loadedTexture_144 : register(t146);
Texture2D<float4> _loadedTexture_145 : register(t147);
Texture2D<float4> _loadedTexture_146 : register(t148);
Texture2D<float4> _loadedTexture_147 : register(t149);
Texture2D<float4> _loadedTexture_148 : register(t150);
Texture2D<float> _loadedTexture_149 : register(t151);
Texture2D<float4> _loadedTexture_150 : register(t152);
Texture2D<float4> _loadedTexture_151 : register(t153);
Texture2D<float4> _loadedTexture_152 : register(t154);
Texture2D<float4> _loadedTexture_153 : register(t155);
Texture2D<float4> _loadedTexture_154 : register(t156);
Texture2D<float4> _loadedTexture_155 : register(t157);
Texture2D<float4> _loadedTexture_156 : register(t158);
Texture2D<float4> _loadedTexture_157 : register(t159);
Texture2D<float4> _loadedTexture_158 : register(t160);
Texture2D<float4> _loadedTexture_159 : register(t161);
Texture2D<float4> _loadedTexture_160 : register(t162);
Texture2D<float4> _loadedTexture_161 : register(t163);
Texture2D<float4> _loadedTexture_162 : register(t164);
Texture2D<float4> _loadedTexture_163 : register(t165);
Texture2D<float4> _loadedTexture_164 : register(t166);
Texture2D<float> _loadedTexture_165 : register(t167);
Texture2D<float4> _loadedTexture_166 : register(t168);
Texture2D<float4> _loadedTexture_167 : register(t169);
Texture2D<float4> _loadedTexture_168 : register(t170);
Texture2D<float4> _loadedTexture_169 : register(t171);
Texture2D<float> _loadedTexture_170 : register(t172);
Texture2D<float4> _loadedTexture_171 : register(t173);
Texture2D<float4> _loadedTexture_172 : register(t174);
Texture2D<float4> _loadedTexture_173 : register(t175);
Texture2D<float4> _loadedTexture_174 : register(t176);
Texture2D<float4> _loadedTexture_175 : register(t177);
Texture2D<float4> _loadedTexture_176 : register(t178);
Texture2D<float4> _loadedTexture_177 : register(t179);
Texture2DArray<float2> _loadedTexture_178 : register(t180);
Texture2D<float> _loadedTexture_179 : register(t181);
Texture2D<float> _loadedTexture_180 : register(t182);
Texture2D<float> _loadedTexture_181 : register(t183);
Texture2D<float> _loadedTexture_182 : register(t184);
Texture2D<float> _loadedTexture_183 : register(t185);
Texture2D<float> _loadedTexture_184 : register(t186);
Texture2D<float> _loadedTexture_185 : register(t187);
Texture2DArray<float2> _loadedTexture_186 : register(t188);
Texture2DArray<float2> _loadedTexture_187 : register(t189);
Texture2DArray<float2> _loadedTexture_188 : register(t190);
Texture2DArray<float2> _loadedTexture_189 : register(t191);
Texture2D<float> _loadedTexture_190 : register(t192);
Texture2DArray<float2> _loadedTexture_191 : register(t193);
Texture2DArray<float2> _loadedTexture_192 : register(t194);
Texture2D<float> _loadedTexture_193 : register(t195);
Texture2DArray<float2> _loadedTexture_194 : register(t196);
Texture2DArray<float2> _loadedTexture_195 : register(t197);
Texture2DArray<float2> _loadedTexture_196 : register(t198);
Texture2DArray<float2> _loadedTexture_197 : register(t199);
Texture2DArray<float2> _loadedTexture_198 : register(t200);
Texture2DArray<float2> _loadedTexture_199 : register(t201);
Texture2DArray<float2> _loadedTexture_200 : register(t202);
Texture2DArray<float2> _loadedTexture_201 : register(t203);
Texture2D<float> _loadedTexture_202 : register(t204);
ConstantBuffer<Struct__RayGenCB> _RayGenCB : register(b0);

#line 7


#include "PCG.hlsli"
#include "IndexToColor.hlsli"
#include "LDSShuffler.hlsli"
#include "s2h\\include\\s2h.hlsl"
#include "s2h\\include\\s2h_scatter.hlsl"

static const float PI = 3.14159265358979323846f;
static const float c_maxT = 10000.0f;

// bridge tokens to include files
static const uint t_aperture_stop = _RayGenCB.HeliosApertureStop;
static const float t_focal_length = _RayGenCB.FocalLength;
static const float t_focus_distance = _RayGenCB.FocusDistance;
static const float4x4 t_invViewMtx = _RayGenCB.InvViewMtx;
static const float3 t_cameraPos = _RayGenCB.CameraPos;
static const float t_lens_position_shift = _RayGenCB.ShiftHeliosPosition;
static const bool t_debug_toggle = (bool)_RayGenCB.DebugToggle;
static const float t_smallLightRadius = _RayGenCB.SmallLightRadius;
static const float t_smallLightBrightness = _RayGenCB.SmallLightBrightness;
static const bool t_renderPinhole = (bool)_RayGenCB.RenderPinhole;
static const bool t_renderThinLensDoF = (bool)_RayGenCB.RenderThinLensDoF;
static const bool t_renderLensSimulationDoF = (bool)_RayGenCB.RenderLensSimulationDoF;
static const bool t_renderBokehConfig = (bool)_RayGenCB.RenderBokehConfig;
static const int t_bokehConfigMode = _RayGenCB.BokehConfigMode;
static const float t_config_light_distance = _RayGenCB.ConfigLightDistance;
static const uint2 t_config_light_count = _RayGenCB.ConfigLightCount;
static const bool t_config_only_diagonal = (bool)_RayGenCB.ConfigOnlyDiagonal;
static const int t_only_this_light_by_index = _RayGenCB.OnlyThisLightByIndex;
static const float t_config_light_field_width = _RayGenCB.ConfigLightFieldWidth;

float sampleHeliosApertureMask(float2 uv)
{
	switch(t_aperture_stop)
	{
		case 0: return _loadedTexture_0.SampleLevel(PointClampSampler, uv, 0).r;
		case 1: return _loadedTexture_1.SampleLevel(PointClampSampler, uv, 0).r;
		case 2: return _loadedTexture_2.SampleLevel(PointClampSampler, uv, 0).r;
		case 3: return _loadedTexture_3.SampleLevel(PointClampSampler, uv, 0).r;
		case 4: return _loadedTexture_4.SampleLevel(PointClampSampler, uv, 0).r;
		case 5: return _loadedTexture_5.SampleLevel(PointClampSampler, uv, 0).r;
		case 6: return _loadedTexture_6.SampleLevel(PointClampSampler, uv, 0).r;
		default: return _loadedTexture_0.SampleLevel(PointClampSampler, uv, 0).r;
	}
}

#include "common_structs.hlsli"
#include "DrawBokehConfig.hlsli"
#include "CameraLensData.hlsli"
#include "LensSimulation.hlsli"
#include "DrawDebugLens.hlsli"

//#define FLT_MAX		3.402823466e+38
#define FLT_MAX		c_maxT

struct Payload
{
	float hitT;
	uint primitiveIndex;
	float2 bary;
};

struct MaterialInfo
{
	float3 emissive;
	float3 albedo;
	float alpha;
	bool glass;
	float3 specular;
};

#define MATERIAL(matID,albedo_) case matID: { float4 diffuse = albedo_.SampleLevel(PointWrapSampler, UV, 0); ret.albedo=diffuse.rgb; ret.alpha = 1.0f;}break;
#define MATERIAL_Tr(matID,albedo_,transparency_) case matID: ret.albedo = albedo_.SampleLevel(PointWrapSampler, UV, 0).rgb; ret.alpha = 1.0f - transparency_;break;
#define MATERIAL_Ke(matID,albedo_,emissive_) case matID: ret.albedo = albedo_.SampleLevel(PointWrapSampler, UV, 0).rgb; ret.emissive = emissive_;break;
#define MATERIAL_MapKe(matID,albedo_,emissive_) case matID: ret.albedo = albedo_.SampleLevel(PointWrapSampler, UV, 0).rgb; ret.emissive = emissive_.SampleLevel(PointWrapSampler, UV, 0).rgb; break;
#define MATERIAL_MapKeRRR(matID,albedo_,emissive_) case matID: ret.albedo = albedo_.SampleLevel(PointWrapSampler, UV, 0).rgb; ret.emissive = emissive_.SampleLevel(PointWrapSampler, UV, 0).rrr; break;
#define MATERIAL_MapKe_Mapd(matID,albedo_,emissive_,d_) case matID: ret.albedo = albedo_.SampleLevel(PointWrapSampler, UV, 0).rgb; ret.emissive = emissive_.SampleLevel(PointWrapSampler, UV, 0).rgb; ret.alpha = d_.SampleLevel(PointWrapSampler, UV, 0).r; break;
#define MATERIAL_Mapd(matID,albedo_,d_) case matID: ret.albedo = albedo_.SampleLevel(PointWrapSampler, UV, 0).rgb; ret.alpha = d_.SampleLevel(PointWrapSampler, UV, 0).r; break;
#define MATERIAL_Glass(matID,transmission_,specular_) case matID: ret.albedo = transmission_; ret.specular = specular_; ret.glass = true; break;

MaterialInfo EvaluateMaterial_Exterior(uint materialID, float2 UV)
{
	MaterialInfo ret = (MaterialInfo)0;
	ret.alpha = 1.0f;

	switch(materialID)
	{
		// Pavement_Curbstones
		MATERIAL(0, _loadedTexture_7);

		// Pavement_Cobblestone_Small_BLENDSHADER
		MATERIAL(1, _loadedTexture_8);

		// Pavement_Cobblestone_Big_BLENDSHADER
		MATERIAL(2, _loadedTexture_9);

		// Pavement_Cobblestone_02
		MATERIAL(3, _loadedTexture_10);

		// Pavement_Brick_BLENDSHADER
		MATERIAL(4, _loadedTexture_11);

		// Pavement_Ground_Wet
		MATERIAL(5, _loadedTexture_12);

		// Pavement_Manhole_Cover
		MATERIAL(6, _loadedTexture_13);

		// Pavement_Cobblestone_Wet_BLENDSHADER
		MATERIAL(7, _loadedTexture_14);

		// Pavement_Cobblestone_Wet_Leaves_BLENDSHADER
		MATERIAL(8, _loadedTexture_14);

		// Pavement_Cobblestone_01_BLENDSHADER
		MATERIAL(9, _loadedTexture_15);

		// Pavement_Cobble_Leaves_BLENDSHADER
		MATERIAL(10, _loadedTexture_15);

		// MASTER_Brick_Small_Red_BLENDSHADER
		MATERIAL(11, _loadedTexture_16);

		// Concrete3
		MATERIAL(12, _loadedTexture_17);

		// MASTER_Focus
		MATERIAL(13, _loadedTexture_18);

		// MASTER_Metal
		MATERIAL(14, _loadedTexture_19);

		// MASTER_Focus_Glass
		MATERIAL(15, _loadedTexture_20);

		// MASTER_Focus_Ornament
		MATERIAL(16, _loadedTexture_21);

		// MASTER_Bistro_Main_Door
		MATERIAL(17, _loadedTexture_22);

		// MASTER_Concrete_Grooved
		MATERIAL(18, _loadedTexture_23);

		// MASTER_Glass_Exterior
		MATERIAL(19, _loadedTexture_24);

		// MASTER_Light_Bulb
		MATERIAL(20, _loadedTexture_18);

		// MASTER_Wood_Painted3
		MATERIAL(21, _loadedTexture_25);

		// MASTER_Concrete1
		MATERIAL(22, _loadedTexture_26);

		// MASTER_Side_Letters
		MATERIAL_Mapd(23, _loadedTexture_27, _loadedTexture_28);

		// Balcony_Concrete
		MATERIAL(24, _loadedTexture_29);

		// Balcony_Trims
		MATERIAL(25, _loadedTexture_30);

		// Balcony_Ornaments
		MATERIAL(26, _loadedTexture_31);

		// Balcony_Green_Wood
		MATERIAL(27, _loadedTexture_32);

		// MASTER_Glass_Dirty
		MATERIAL_Glass(28, float3(0.2f, 0.2f, 0.2f), float3(0.5f, 0.5f, 0.5f));

		// MASTER_Plastic
		MATERIAL(29, _loadedTexture_33);

		// MASTER_Grain_Metal
		MATERIAL(30, _loadedTexture_34);

		// MASTER_Concrete_Yellow
		MATERIAL(31, _loadedTexture_35);

		// MASTER_Concrete
		MATERIAL(32, _loadedTexture_17);

		// MASTER_Concrete_Smooth
		MATERIAL(33, _loadedTexture_36);

		// MASTER_Roofing_Shingle_Grey
		MATERIAL(34, _loadedTexture_37);

		// MASTER_Concrete_Plaster1_BLENDSHADER
		MATERIAL(35, _loadedTexture_36);

		// MASTER_Roofing_Shingle_Green
		MATERIAL(36, _loadedTexture_38);

		// MASTER_Trim_Cornice
		MATERIAL(37, _loadedTexture_39);

		// MASTER_Forge_Metal
		MATERIAL_Mapd(38, _loadedTexture_40, _loadedTexture_41);

		// MASTER_Curtains
		MATERIAL(39, _loadedTexture_42);

		// MASTER_Metal_Pipe
		MATERIAL(40, _loadedTexture_34);

		// MASTER_Wood_Polished
		MATERIAL(41, _loadedTexture_43);

		// MASTER_Doors
		MATERIAL(42, _loadedTexture_44);

		// MASTER_Brick_Large_Beige_BLENDSHADER
		MATERIAL(43, _loadedTexture_45);

		// MASTER_Wood_Brown
		MATERIAL(44, _loadedTexture_43);

		// MASTER_Glass_Clean
		MATERIAL_Glass(45, float3(0.2f, 0.2f, 0.2f), float3(0.5f, 0.5f, 0.5f));

		// MASTER_Wood_Painted_Green
		MATERIAL(46, _loadedTexture_46);

		// MASTER_Concrete_Plaster
		MATERIAL(47, _loadedTexture_36);

		// MASTER_Black_Metal
		MATERIAL(48, _loadedTexture_34);

		// MASTER_Brick_Large_White
		MATERIAL(49, _loadedTexture_47);

		// MASTER_Building_Details
		MATERIAL(50, _loadedTexture_48);

		// MASTER_Wood_Painted_Cyan
		MATERIAL(51, _loadedTexture_49);

		// MASTER_Brick_Small_Red
		MATERIAL(52, _loadedTexture_50);

		// MASTER_Roofing_Metal_01
		MATERIAL(53, _loadedTexture_51);

		// MASTER_Room_Interior
		MATERIAL(54, _loadedTexture_52);

		// MASTER_Awning_Beams
		MATERIAL(55, _loadedTexture_53);

		// MASTER_Boulangerie
		MATERIAL(56, _loadedTexture_54);

		// MASTER_Frosted_Glass
		MATERIAL(57, _loadedTexture_55);

		// MASTER_Awning_Fabric_Cyan
		MATERIAL(58, _loadedTexture_56);

		// MASTER_Bronze_BLENDSHADER
		MATERIAL(59, _loadedTexture_57);

		// MASTER_Concrete_White
		MATERIAL(60, _loadedTexture_58);

		// MASTER_Awning_Fabric_Red
		MATERIAL(61, _loadedTexture_59);

		// MASTER_Book_Covers
		MATERIAL(62, _loadedTexture_60);

		// MASTER_Concrete_Smooth_BLENDSHADER
		MATERIAL(63, _loadedTexture_61);

		// MASTER_Wood_Painted_Green_BLENDSHADER
		MATERIAL(64, _loadedTexture_62);

		// MASTER_Rollup_Door
		MATERIAL(65, _loadedTexture_63);

		// MASTER_Details_Dark
		MATERIAL(66, _loadedTexture_48);

		// Foliage_Bux_Hedges46
		MATERIAL_Mapd(67, _loadedTexture_64, _loadedTexture_65);

		// Foliage_Leaves
		MATERIAL_Mapd(68, _loadedTexture_66, _loadedTexture_67);

		// Foliage_Trunk
		MATERIAL(69, _loadedTexture_68);

		// Foliage_Flowers
		MATERIAL_Mapd(70, _loadedTexture_69, _loadedTexture_70);

		// Foliage_Paris_Flowers
		MATERIAL_Mapd(71, _loadedTexture_71, _loadedTexture_72);

		// Foliage_Linde_Tree_Large_Trunk
		MATERIAL(72, _loadedTexture_73);

		// Foliage_Linde_Tree_Large_Orange_Leaves
		MATERIAL_Mapd(73, _loadedTexture_74, _loadedTexture_75);

		// Foliage_Linde_Tree_Large_Green_Leaves
		MATERIAL_Mapd(74, _loadedTexture_76, _loadedTexture_77);

		// Foliage_Ivy_leaf_a
		MATERIAL_Mapd(75, _loadedTexture_78, _loadedTexture_79);

		// Foliage_Ivy_branches
		MATERIAL(76, _loadedTexture_80);

		// Streetlight_Glass
		MATERIAL(77, _loadedTexture_81);

		// Streetlight_Support_Bulb
		MATERIAL(78, _loadedTexture_81);

		// Paris_StringLights_01_White_Color
		MATERIAL_Ke(79, _loadedTexture_82, float3(1.0f, 1.0f, 1.0f));

		// Streetlight_Metal
		MATERIAL(80, _loadedTexture_83);

		// Streetlight_Chains
		MATERIAL_Mapd(81, _loadedTexture_84, _loadedTexture_85);

		// Paris_Streetpivot
		MATERIAL(82, _loadedTexture_86);

		// ElectricBox
		MATERIAL(83, _loadedTexture_87);

		// Shopsign_Bakery
		MATERIAL(84, _loadedTexture_88);

		// Banner_Metal
		MATERIAL(85, _loadedTexture_89);

		// Shopsign_Book_Store
		MATERIAL(86, _loadedTexture_90);

		// Shopsign_Pharmacy
		MATERIAL(87, _loadedTexture_91);

		// Shopsign_Ties_Shop
		MATERIAL(88, _loadedTexture_92);

		// Spotlight_Main
		MATERIAL(89, _loadedTexture_93);

		// Spotlight_Emissive
		MATERIAL_Ke(90, _loadedTexture_93, float3(1.0f, 1.0f, 1.0f) * 20.0f);

		// Spotlight_Glass
		MATERIAL_Glass(91, float3(0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 0.0f));

		// Paris_StreetSign_01
		MATERIAL(92, _loadedTexture_94);

		// Paris_TrafficSign_A
		MATERIAL(93, _loadedTexture_95);

		// MenuSign_01
		MATERIAL(94, _loadedTexture_96);

		// MenuSign_02_Mesh
		MATERIAL(95, _loadedTexture_97);

		// MenuSign_02_Glass
		MATERIAL(96, _loadedTexture_98);

		// Trashcan
		MATERIAL(97, _loadedTexture_99);

		// Paris_StringLights_01_Green_Color
		MATERIAL_Ke(98, _loadedTexture_100, float3(0.1f, 1.0f, 0.1f));

		// Paris_StringLights_01_Red_Color
		MATERIAL_Ke(99, _loadedTexture_101, float3(1.0f, 0.1f, 0.1f));

		// Stringlights
		MATERIAL(100, _loadedTexture_102);

		// Paris_StringLights_01_Blue_Color
		MATERIAL_Ke(101, _loadedTexture_82, float3(0.1f, 0.1f, 1.0f));

		// Paris_StringLights_01_Pink_Color
		MATERIAL_Ke(102, _loadedTexture_103, float3(1.0f, 0.1f, 1.0f));

		// Paris_StringLights_01_Orange_Color
		MATERIAL_Ke(103, _loadedTexture_104, float3(1.0f, 1.0f, 0.0f));

		// Bollards
		MATERIAL(104, _loadedTexture_105);

		// Vespa_Odometer_Glass
		MATERIAL_Glass(105, float3(0.2f, 0.2f, 0.2f), float3(0.5f, 0.5f, 0.5f));

		// Vespa_Headlight
		MATERIAL_Glass(106, float3(0.2f, 0.2f, 0.2f), float3(0.5f, 0.5f, 0.5f));

		// Vespa
		MATERIAL(107, _loadedTexture_106);

		// Vespa_Odometer
		MATERIAL(108, _loadedTexture_107);

		// Antenna_Metal
		MATERIAL(109, _loadedTexture_108);

		// Antenna_Plastic_Blue
		MATERIAL(110, _loadedTexture_109);

		// Antenna_Plastic
		MATERIAL(111, _loadedTexture_109);

		// Awnings_Fabric
		MATERIAL(112, _loadedTexture_110);

		// Awnings_Beams
		MATERIAL(113, _loadedTexture_111);

		// Awnings_Hotel_Fabric
		MATERIAL(114, _loadedTexture_112);

		// Bistro_Sign_Main
		MATERIAL(115, _loadedTexture_113);

		// Bistro_Sign_Letters
		MATERIAL(116, _loadedTexture_114);

		// Plantpots
		MATERIAL(117, _loadedTexture_115);

		// Sidewalkbarrier
		MATERIAL(118, _loadedTexture_116);

		// Concrete
		MATERIAL(119, _loadedTexture_117);

		// Plaster
		MATERIAL(120, _loadedTexture_118);

		// Concrete2
		MATERIAL(121, _loadedTexture_119);

		// Concrete_Striped
		MATERIAL(122, _loadedTexture_120);

		// Chimneys_Metal
		MATERIAL(123, _loadedTexture_121);

		// Lantern
		MATERIAL(124, _loadedTexture_122);

		// Paris_Chair_01
		MATERIAL(125, _loadedTexture_123);

		// Paris_Table_Terrace
		MATERIAL(126, _loadedTexture_124);

		// Ashtray
		MATERIAL(127, _loadedTexture_125);

		// NapkinHolder
		MATERIAL(128, _loadedTexture_126);

		// Paris_LiquorBottle_01_Glass_Wine
		MATERIAL(129, _loadedTexture_127);

		// Paris_LiquorBottle_01_Caps
		MATERIAL(130, _loadedTexture_128);

		// Paris_LiquorBottle_01_Labels
		MATERIAL_Mapd(131, _loadedTexture_129, _loadedTexture_130);

		// unknown material goes magenta
		default:
		{
			ret.albedo = float3(1.0f, 0.0f, 1.0f);
			break;
		}
	}

	ret.emissive *= _RayGenCB.MaterialEmissiveMultiplier;

	return ret;
}

MaterialInfo EvaluateMaterial_Interior(uint materialID, float2 UV)
{
	MaterialInfo ret = (MaterialInfo)0;
	ret.alpha = 1.0f;

	switch(materialID)
	{
		// MASTER_Interior_01_Plaster
		MATERIAL(0, _loadedTexture_17);

		// MASTER_Interior_01_Wood
		MATERIAL(1, _loadedTexture_131);

		// MASTER_Interior_01_Wooden_stuco
		MATERIAL(2, _loadedTexture_132);

		// MASTER_Wood_Painted3
		MATERIAL(3, _loadedTexture_25);

		// MASTER_Interior_01_Plaster_Red
		MATERIAL(4, _loadedTexture_133);

		// MASTER_Interior_01_Grid
		MATERIAL(5, _loadedTexture_134);

		// MASTER_Interior_01_Brushed_Metal
		MATERIAL(6, _loadedTexture_135);

		// MASTER_Interior_01_Plaster2
		MATERIAL(7, _loadedTexture_17);

		// MASTER_Interior_01_Frozen_Glass
		MATERIAL(8, _loadedTexture_55);

		// MASTER_Interior_01_Floor_Tile_Hexagonal_BLENDSHADER
		MATERIAL(9, _loadedTexture_136);

		// MASTER_Interior_01_Glass
		MATERIAL_Tr(10, _loadedTexture_137, 0.8f);

		// MASTER_Interior_01_Paris_Bartrim
		MATERIAL(11, _loadedTexture_138);

		// MASTER_Interior_01_White_Plastic
		MATERIAL(12, _loadedTexture_139);

		// MASTER_Interior_01_Grid1
		MATERIAL(13, _loadedTexture_140);

		// MASTER_Interior_01_Material
		MATERIAL(14, _loadedTexture_141);

		// MASTER_Interior_01_Paris_Lantern
		MATERIAL_MapKe_Mapd(15, _loadedTexture_122, _loadedTexture_142, _loadedTexture_143);

		// curtainB1
		MATERIAL(16, _loadedTexture_144);

		// curtainA
		MATERIAL(17, _loadedTexture_145);

		// Paris_Ceiling_Lamp
		MATERIAL_MapKe(18, _loadedTexture_146, _loadedTexture_147);

		// Paris_Wall_Light_Interior
		MATERIAL_MapKeRRR(19, _loadedTexture_148, _loadedTexture_149);

		// Paris_Wall_Bulb_Light
		MATERIAL_Ke(20, _loadedTexture_82, float3(1.0f, 1.0f, 1.0f));

		// Wall_Lamp
		MATERIAL_MapKe(21, _loadedTexture_122, _loadedTexture_142);

		// Metal_Bronze_01
		MATERIAL(22, _loadedTexture_150);

		// Paris_CeilingFan
		// This had a map_ke but the file doesn't exist on disk
		MATERIAL(23, _loadedTexture_151);

		// Paris_Beertap
		MATERIAL(24, _loadedTexture_152);

		// Paris_LiquorBottle_02_Glass
		MATERIAL_Glass(25, float3(0.2f, 0.1f, 0.0f), float3(0.5f, 0.2f, 0.098f));

		// Paris_LiquorBottle_01_Caps
		MATERIAL(26, _loadedTexture_128);

		// ToffeeJar_Label
		MATERIAL(27, _loadedTexture_129);

		// Paris_LiquorBottle_01_Glass_Wine
		MATERIAL(28, _loadedTexture_127);

		// Paris_LiquorBottle_01_Glass
		MATERIAL_Glass(29, float3(0.2f, 0.2f, 0.2f), float3(0.212f, 0.212f, 0.212f));

		// Paris_LiquorBottle_03_Glass
		MATERIAL_Glass(30, float3(0.2f, 0.2f, 0.2f), float3(0.212f, 0.255f, 0.333f));

		// Metal_Worn_01
		MATERIAL(31, _loadedTexture_108);

		// Wood
		MATERIAL(32, _loadedTexture_153);

		// Plastic_02
		MATERIAL(33, _loadedTexture_154);

		// Paris_Painting_Metal
		MATERIAL(34, _loadedTexture_155);

		// Paris_Paintings
		MATERIAL(35, _loadedTexture_156);

		// Paris_Paintings_Glass
		MATERIAL(36, _loadedTexture_98);

		// Paris_BarStool
		MATERIAL(37, _loadedTexture_157);

		// ToffeeJar_Metal
		MATERIAL(38, _loadedTexture_158);

		// Paris_LiquorBottle_01_Glass1
		MATERIAL_Glass(39, float3(0.03f, 0.05f, 0.04f), float3(0.4f, 0.4f, 0.4f));

		// ToffeeJar_Toffee
		MATERIAL(40, _loadedTexture_82);

		// CookieJar_Glass1
		MATERIAL_Glass(41, float3(0.1f, 0.1f, 0.1f), float3(0.3f, 0.3f, 0.3f));

		// CookieJar_Cookies
		MATERIAL(42, _loadedTexture_159);

		// Paris_Radiator
		MATERIAL(43, _loadedTexture_160);

		// Cutlery_chrome
		MATERIAL(44, _loadedTexture_161);

		// Cutlery_details
		MATERIAL_Mapd(45, _loadedTexture_162, _loadedTexture_162);

		// Paris_Table_03
		MATERIAL(46, _loadedTexture_163);

		// Plates_Details
		MATERIAL_Mapd(47, _loadedTexture_164, _loadedTexture_165);

		// Plates_Ceramic
		MATERIAL(48, _loadedTexture_166);

		// NapkinHolder_01
		MATERIAL(49, _loadedTexture_126);

		// Paris_Table_cloth_01
		MATERIAL(50, _loadedTexture_167);

		// Metal_Chrome1
		MATERIAL(51, _loadedTexture_158);

		// Cloth
		MATERIAL(52, _loadedTexture_168);

		// Plants_plants
		MATERIAL_Mapd(53, _loadedTexture_169,_loadedTexture_170);

		// Plants_Metal_Base_01
		MATERIAL(54, _loadedTexture_108);

		// Trolley_Plastic
		MATERIAL(55, _loadedTexture_154);

		// Trolley_Wheels
		MATERIAL(56, _loadedTexture_99);

		// Trolley_Wood_Painted
		MATERIAL(57, _loadedTexture_171);

		// WickerBasket
		MATERIAL(58, _loadedTexture_172);

		// Paris_Coasters_01
		MATERIAL(59, _loadedTexture_173);

		// Paris_Cashregister
		MATERIAL(60, _loadedTexture_174);

		// Paris_Cashregister_Buttons
		MATERIAL(61, _loadedTexture_175);

		// Paris_Cashregister_Glass
		MATERIAL_Tr(62, _loadedTexture_98, 0.9f);

		// Paris_Doormat
		MATERIAL(63, _loadedTexture_176);

		// Rubber_Bar_Mat_01
		MATERIAL(64, _loadedTexture_177);

		// Plastic_01
		MATERIAL(65, _loadedTexture_154);

		// unknown material goes magenta
		default:
		{
			ret.albedo = float3(1.0f, 0.0f, 1.0f);
			break;
		}
	}

	ret.emissive *= _RayGenCB.MaterialEmissiveMultiplier;

	return ret;
}

// returns -1.0 on miss, else returns time to hit
float TestSphereTrace(in float3 rayPos, in float3 rayDir, in float4 sphere, out float3 normal)
{
	//get the vector from the center of this sphere to where the ray begins.
	float3 m = rayPos - sphere.xyz;

	//get the dot product of the above vector and the ray's vector
	float b = dot(m, rayDir);

	float c = dot(m, m) - sphere.w * sphere.w;

	//exit if r's origin outside s (c > 0) and r pointing away from s (b > 0)
	if(c > 0.0 && b > 0.0)
		return -1.0f;

	//calculate discriminant
	float discr = b * b - c;

	//a negative discriminant corresponds to ray missing sphere
	if(discr < 0.0)
		return -1.0f;
    
	//ray now found to intersect sphere, compute smallest t value of intersection
    bool fromInside = false;
	float dist = -b - sqrt(discr);
    if (dist < 0.0f)
    {
        fromInside = true;
        dist = -b + sqrt(discr);
    }

	normal = normalize((rayPos+rayDir*dist) - sphere.xyz) * (fromInside ? -1.0f : 1.0f);

	return dist;
}

// returns true if pos can see target
bool VisibilityQuery(float3 pos, float3 direction, float dist)
{
	RayDesc ray;
	ray.Origin = pos;
	ray.TMin = 0;
	ray.TMax = dist;
	ray.Direction = direction;

	Payload payload = (Payload)0;

	TraceRay(Scene,
		RAY_FLAG_FORCE_OPAQUE, // Ray flags
		0xFF, // Ray mask
		0,
		1,
		0,
		ray,
		payload);

	return payload.hitT < 0.0f;
}

float3 SmallLightColor(int index)
{
	float3 ret = _RayGenCB.SmallLightsColor;
	if ((bool)_RayGenCB.SmallLightsColorful)
		ret = IndexToColor(index, 0.95f, 0.95f);
	return ret;
}

bool SmallLightContributions(float3 pos, float3 dir, inout Struct_PixelDebugStruct pixelDebug, float maxT, out float3 lightColor, bool writeDebugHitT)
{
	// corner 878,380, 419
	// 815, 380, 395
	// 63, 0, 24

	//static const float3 c_smallLightStart = float3(900.0f, 280.0f, 400.0f);
	//static const float3 c_smallLightEnd = float3(25.0f, 260.0f, -550.0f);

	// back wall: (910, 325, 395) - (1135, 320, -15)
	static const int c_numSmallLights = 10;
	static const float3 c_smallLightStart = float3(910.0f, 325.0f, 395.0f);
	static const float3 c_smallLightEnd = float3(1135.0f, 320.0f, -15.0f);
	static const float c_lightSlackMax = 30.0f;

	static const int c_lightRows = 1;
	static const float3 c_lightRowsOffset = float3(-63.0f, -4.0f, -24.0f) * 6.0f;

	lightColor = float3(0.0f, 0.0f, 0.0f);

	float globalHitT = maxT;
	bool anyHits = false;

	for (int rowIndex = 0; rowIndex < c_lightRows; ++rowIndex)
	{
		for (int i = 0; i < c_numSmallLights; ++i)
		{
			float percent = float(i+1) / float(c_numSmallLights+1);
			float3 lightPos = lerp(c_smallLightStart, c_smallLightEnd, percent);

			float slackPercent = 1.0f - percent;
			lightPos.y += (cos(slackPercent * c_twopi) * 0.5f - 0.5f) * c_lightSlackMax;
			lightPos += c_lightRowsOffset * rowIndex;

			float3 sphereNormal;
			float localHitT = TestSphereTrace(pos, dir, float4(lightPos, _RayGenCB.SmallLightRadius), sphereNormal);
			if (localHitT < 0.0f || localHitT > globalHitT)
				continue;

			globalHitT = localHitT;
			lightColor = SmallLightColor(i) * _RayGenCB.SmallLightBrightness;
			anyHits = true;
		}
	}

	if (!anyHits || !VisibilityQuery(pos, dir, globalHitT))
	{
		lightColor = float3(0.0f, 0.0f, 0.0f);
		return false;
	}

	if (writeDebugHitT)
	{
		pixelDebug.HitT = globalHitT;
	}
	return true;
}

float3 GetColorForRay(float3 pos, float3 dir, inout uint RNG, inout Struct_PixelDebugStruct pixelDebug, in uint rayIndex, in uint2 px)
{
	float3 throughput = float3(1.0f, 1.0f, 1.0f);
	float3 color = float3(0.0f, 0.0f, 0.0f);

	// show small lights for primary ray
	if(SmallLightContributions(pos, dir, pixelDebug, c_maxT, color, rayIndex == 0))
		return color;

	for (uint bounceIndex = 0; bounceIndex < _RayGenCB.NumBounces; ++bounceIndex)
	{
		RayDesc ray;
		ray.Origin = pos;
		ray.TMin = 0;
		ray.TMax = c_maxT;
		ray.Direction = dir;

		Payload payload = (Payload)0;

		TraceRay(Scene,
			RAY_FLAG_FORCE_OPAQUE, // Ray flags
			0xFF, // Ray mask
			0,
			1,
			0,
			ray,
			payload);

		// see if the ray hit the small lights
		float3 smallLightColor = float3(0.0f, 0.0f, 0.0f);
		if(SmallLightContributions(ray.Origin, ray.Direction, pixelDebug, (payload.hitT < 0.0f ? c_maxT : payload.hitT), smallLightColor, false))
		{
			color += smallLightColor * throughput;
			return color;
		}

		// on miss, apply sky lighting and we are done
		if (payload.hitT < 0.0f)
		{
			float3 emissive = _RayGenCB.SkyColor * _RayGenCB.SkyBrightness;
			color += emissive * throughput;

			if (rayIndex == 0 && bounceIndex == 0)
			{
				pixelDebug.MaterialID = ~0;
				pixelDebug.HitT = FLT_MAX;
				pixelDebug.WorldPos = float3(0.0f, 0.0f, 0.0f);
			}
			return color;
		}

		// get surface information
		uint vertA = payload.primitiveIndex*3+0;
		uint vertB = payload.primitiveIndex*3+1;
		uint vertC = payload.primitiveIndex*3+2;

		uint materialID = VertexBuffer[vertA].materialID;

		float2 UV0 = VertexBuffer[vertA].UV;
		float2 UV1 = VertexBuffer[vertB].UV;
		float2 UV2 = VertexBuffer[vertC].UV;
		float2 UV = UV0 + payload.bary.x * (UV1 - UV0) + payload.bary.y * (UV2-UV0);

		float3 normal0 = VertexBuffer[vertA].normal;
		float3 normal1 = VertexBuffer[vertB].normal;
		float3 normal2 = VertexBuffer[vertC].normal;
		float3 normal = normalize(normal0 + payload.bary.x * (normal1 - normal0) + payload.bary.y * (normal2-normal0));

		// Evaluate material
		MaterialInfo materialInfo = (MaterialInfo)0;
		
		switch(_RayGenCB.MaterialSet)
		{
			case MaterialSets::Exterior: materialInfo = EvaluateMaterial_Exterior(materialID, UV); break;
			case MaterialSets::Interior: materialInfo = EvaluateMaterial_Interior(materialID, UV); break;
			default: materialInfo.albedo = float3(1.0f, 0.0f, 1.0f); break;
		}

		if (rayIndex == 0 && bounceIndex == 0)
		{
			pixelDebug.MaterialID = materialID;
			pixelDebug.HitT = payload.hitT;
			pixelDebug.WorldPos = pos + dir * payload.hitT;
		}

		// Glass
		if (materialInfo.glass)
		{
			// Specular reflection
			if (RandomFloat01(RNG) > 0.5f)
			{
				// specular bounce
				throughput *= materialInfo.specular;
				pos = pos + dir * payload.hitT + normal * _RayGenCB.RayPosNormalNudge;
				dir = reflect(dir, normal);
			}
			// transmission
			else
			{
				throughput *= float3(1.0f, 1.0f, 1.0f) - materialInfo.albedo;
				pos = pos + dir * payload.hitT + dir * _RayGenCB.RayPosNormalNudge;
			}
		}
		// Opaque diffuse
		else if(RandomFloat01(RNG) > (1.0f - materialInfo.alpha))
		{
			// apply emissive and albedo
			color += materialInfo.emissive * throughput;
			throughput *= materialInfo.albedo;

			// Opaque bounce
			pos = pos + dir * payload.hitT + normal * _RayGenCB.RayPosNormalNudge;
			dir = normalize(normal + RandomUnitVector(RNG));
		}
		// Transparent
		else
		{
			// transparency
			pos = pos + dir * payload.hitT + dir * _RayGenCB.RayPosNormalNudge;
			//return float3(1.0f, 0.0f, 1.0f);
		}

		if ((bool)_RayGenCB.AlbedoMode && (materialInfo.alpha > 0.0f || materialInfo.emissive.r > 0.0f || materialInfo.emissive.g > 0.0f || materialInfo.emissive.b > 0.0f))
		{
			return materialInfo.albedo * _RayGenCB.AlbedoModeAlbedoMultiplier + materialInfo.emissive;
		}

	}

	return color;
}

float2 ReadVec2STTexture(in uint3 px, inout uint RNG, in Texture2DArray<float2> tex, bool convertToNeg1Plus1 = true)
{
	uint3 dims;
	tex.GetDimensions(dims.x, dims.y, dims.z);

	// Extend the noise texture over time
	uint cycleCount = px.z / dims.z;
	switch(_RayGenCB.LensRNGExtend)
	{
		case NoiseTexExtends::None: break;
		case NoiseTexExtends::White:
		{
			uint OffsetRNG = HashInit(uint3(0x1337, 0xbeef, cycleCount));
			px.x += HashPCG(OffsetRNG);
			px.y += HashPCG(OffsetRNG);
			break;
		}
		case NoiseTexExtends::Shuffle1D:
		{
			uint shuffleIndex = LDSShuffle1D_GetValueAtIndex(cycleCount, 16384, 10127, 435);
			px.x += shuffleIndex % dims.x;
			px.y += shuffleIndex / dims.x;
			break;
		}
		case NoiseTexExtends::Shuffle1DHilbert:
		{
			uint shuffleIndex = LDSShuffle1D_GetValueAtIndex(cycleCount, 16384, 10127, 435);
			px.xy += Convert1DTo2D_Hilbert(shuffleIndex, 16384);
			break;
		}
	}

    float2 ret = tex[px % dims].rg;

    if (convertToNeg1Plus1)
        ret = ret * 2.0f - 1.0f;

    return ret;
}

float2 SampleICDF(float2 rng, in Texture2D<float> MarginalCDF)
{
    rng = clamp(rng, 0.001f, 0.999f);

    // Get the dimensions of the MarginalCDF
    uint2 MarginalCDFDims;
    MarginalCDF.GetDimensions(MarginalCDFDims.x, MarginalCDFDims.y);

    // Find the column of pixels we want to sample from by doing a branchless binary search.  This is our x axis.
    uint2 samplePos = uint2(0, 0);
    {
        uint numColumns = MarginalCDFDims.x;
        uint pow2Size = (uint)1 << uint(ceil(log2(numColumns)));

        // Do first step manually to handle the possibility of non power of 2
        uint searchSize = pow2Size / 2;
        uint ret = 0;
        ret = (MarginalCDF[uint2(ret + searchSize, 0)] <= rng.x) * (numColumns - searchSize);
        searchSize /= 2;

        // Do the rest of the steps
        while (searchSize > 0)
        {
            ret += (MarginalCDF[uint2(ret + searchSize, 0)] <= rng.x) * searchSize;
            searchSize /= 2;
        }
        samplePos.x = ret;
    }

    // Find the row we want to sample, in that column, by doing a branchless binary search. This is our y axis.
    {
        uint numRows = MarginalCDFDims.y - 1;
        uint pow2Size = (uint)1 << uint(ceil(log2(numRows)));

        // Do first step manually to handle the possibility of non power of 2
        uint searchSize = pow2Size / 2;
        uint ret = 0;
        ret = (MarginalCDF[uint2(samplePos.x, ret + searchSize + 1)] <= rng.y) * (numRows - searchSize);
        searchSize /= 2;

        // Do the rest of the steps
        while (searchSize > 0)
        {
            ret += (MarginalCDF[uint2(samplePos.x, ret + searchSize + 1)] <= rng.y) * searchSize;
            searchSize /= 2;
        }
        samplePos.y = ret;
    }

	// Convert to [-1,+1]^2 coordinates
    float2 uv = (float2(samplePos) + float2(0.5f, 0.5f)) / float2(MarginalCDFDims);
    return uv * 2.0f - 1.0f;
}

// returns PDF
float ApplyDOFLensSimulation(inout float3 rayPos, inout float3 rayDir, in uint3 px, inout uint RNG, in uint2 screenDims)
{

	float3 cameraRight = mul(float4(1.0f, 0.0f, 0.0f, 0.0f), _RayGenCB.InvViewMtx).xyz;
	float3 cameraUp = mul(float4(0.0f, 1.0f, 0.0f, 0.0f), _RayGenCB.InvViewMtx).xyz;

	// calculate point on the focal plane
	float3 focalPlanePoint = rayPos + rayDir * _RayGenCB.FocalLength;

	// calculate a random point on the aperture
	float2 offset = float2(0.0f, 0.0f);

	float PDF = 1.0f;
	if ((bool)_RayGenCB.NoImportanceSampling)
	{
		// Random point in square, then we'll adjust the PDF as appropraite
		float2 uvwhite = float2(RandomFloat01(RNG), RandomFloat01(RNG));
        float2 uvblue = ReadVec2STTexture(px, RNG, _loadedTexture_178, false);

		// set uv to either uvwhite or uvblue, depending on the noise type
        float2 uv;
        switch (_RayGenCB.LensRNGSource)
        {
			case LensRNG::UniformCircleBlue:
			case LensRNG::UniformHexagonBlue:
			case LensRNG::UniformHexagonICDF_Blue:
			case LensRNG::UniformStarBlue:
			case LensRNG::UniformStarICDF_Blue:
			case LensRNG::NonUniformStarBlue:
			case LensRNG::NonUniformStar2Blue:
			case LensRNG::LKCP6Blue:
			case LensRNG::LKCP204Blue:
            case LensRNG::LKCP204ICDF_Blue:
            {
                uv = uvblue;
                break;
			}
            default:
            {
                uv = uvwhite;
			}
        }

		// Do the samplng, without importance sampling
		switch(_RayGenCB.LensRNGSource)
		{
			case LensRNG::UniformCircleWhite_PCG:
			case LensRNG::UniformCircleWhite:
			case LensRNG::UniformCircleBlue:
			{
				PDF = 1.0f - _loadedTexture_179.SampleLevel(PointClampSampler, uv, 0).r;
				PDF = (PDF > 0.5f) ? 1.0f : 0.0f; // The circle image is anti aliased and that makes white specs from very small probabilities in the circle. This corrects that
				PDF *= 0.781056f;
				break;
			}
			case LensRNG::UniformHexagonWhite:
            case LensRNG::UniformHexagonBlue:
            case LensRNG::UniformHexagonICDF_White:
            case LensRNG::UniformHexagonICDF_Blue:
			{
				PDF = 1.0f - _loadedTexture_180.SampleLevel(PointClampSampler, uv, 0).r;
				PDF *= 0.652649f;
				break;
            }
			case LensRNG::UniformStarWhite:
            case LensRNG::UniformStarBlue:
            case LensRNG::UniformStarICDF_White:
            case LensRNG::UniformStarICDF_Blue:
            {
				PDF = 1.0f - _loadedTexture_181.SampleLevel(PointClampSampler, uv, 0).r;
				PDF *= 0.368240f;
				break;
			}
			case LensRNG::NonUniformStarWhite:
			case LensRNG::NonUniformStarBlue:
			{
				PDF = 1.0f - _loadedTexture_182.SampleLevel(PointClampSampler, uv, 0).r;
				// TODO: need to adjust the PDF to account for brightness change
				//PDF *= 0.114055f;
				break;
			}
			case LensRNG::NonUniformStar2White:
			case LensRNG::NonUniformStar2Blue:
			{
				PDF = 1.0f - _loadedTexture_183.SampleLevel(PointClampSampler, uv, 0).r;
				// TODO: need to adjust the PDF to account for brightness change
				//PDF *= 0.051030f;
				break;
			}
			case LensRNG::LKCP6White:
			case LensRNG::LKCP6Blue:
			{
				PDF = _loadedTexture_184.SampleLevel(PointClampSampler, uv, 0).r;
				// TODO: need to adjust the PDF to account for brightness change
				//PDF *= 0.206046f;
				break;
			}
			case LensRNG::LKCP204White:
            case LensRNG::LKCP204Blue:
            case LensRNG::LKCP204ICDF_White:
            case LensRNG::LKCP204ICDF_Blue:
			{
				PDF = _loadedTexture_185.SampleLevel(PointClampSampler, uv, 0).r;
				PDF *= 9.024792127102528f;  // Found by hand

				//PDF /= 0.095413f; // Calculated, but odd that we have to divide, instead of multiply

				if (PDF <= 16.0 / 255.0)
					PDF = 0.0f;

				break;
            }
        }

		offset = uv * 2.0f - 1.0f;
	}
	else
	{
		switch(_RayGenCB.LensRNGSource)
		{
			case LensRNG::UniformCircleWhite_PCG:
			{
				float angle = RandomFloat01(RNG) * 2.0f * c_pi;
				float radius = sqrt(RandomFloat01(RNG));
				offset = float2(cos(angle), sin(angle)) * radius;
				break;
			}
			case LensRNG::UniformCircleWhite:
			{
				offset = ReadVec2STTexture(px, RNG, _loadedTexture_186);
				break;
			}
			case LensRNG::UniformCircleBlue:
			{
				offset = ReadVec2STTexture(px, RNG, _loadedTexture_187);
				break;
			}
			case LensRNG::UniformHexagonWhite:
			{
				offset = ReadVec2STTexture(px, RNG, _loadedTexture_188);
				break;
			}
			case LensRNG::UniformHexagonBlue:
			{
				offset = ReadVec2STTexture(px, RNG, _loadedTexture_189);
				break;
            }
            case LensRNG::UniformHexagonICDF_White:
            {
                float2 rng = float2(RandomFloat01(RNG), RandomFloat01(RNG));
                offset = SampleICDF(rng, _loadedTexture_190);
                break;
            }
            case LensRNG::UniformHexagonICDF_Blue:
            {
                float2 rng = ReadVec2STTexture(px, RNG, _loadedTexture_178, false);
                offset = SampleICDF(rng, _loadedTexture_190);
                break;
            }
			case LensRNG::UniformStarWhite:
			{
				offset = ReadVec2STTexture(px, RNG, _loadedTexture_191);
				break;
			}
			case LensRNG::UniformStarBlue:
			{
				offset = ReadVec2STTexture(px, RNG, _loadedTexture_192);
				break;
            }
            case LensRNG::UniformStarICDF_White:
            {
                float2 rng = float2(RandomFloat01(RNG), RandomFloat01(RNG));
                offset = SampleICDF(rng, _loadedTexture_193);
                break;
            }
            case LensRNG::UniformStarICDF_Blue:
            {
                float2 rng = ReadVec2STTexture(px, RNG, _loadedTexture_178, false);
                offset = SampleICDF(rng, _loadedTexture_193);
                break;
            }
			case LensRNG::NonUniformStarWhite:
			{
				offset = ReadVec2STTexture(px, RNG, _loadedTexture_194);
				break;
			}
			case LensRNG::NonUniformStarBlue:
			{
				offset = ReadVec2STTexture(px, RNG, _loadedTexture_195);
				break;
			}
			case LensRNG::NonUniformStar2White:
			{
				offset = ReadVec2STTexture(px, RNG, _loadedTexture_196);
				break;
			}
			case LensRNG::NonUniformStar2Blue:
			{
				offset = ReadVec2STTexture(px, RNG, _loadedTexture_197);
				break;
			}
			case LensRNG::LKCP6White:
			{
				offset = ReadVec2STTexture(px, RNG, _loadedTexture_198);
				break;
			}
			case LensRNG::LKCP6Blue:
			{
				offset = ReadVec2STTexture(px, RNG, _loadedTexture_199);
				break;
			}
			case LensRNG::LKCP204White:
			{
				offset = ReadVec2STTexture(px, RNG, _loadedTexture_200);
				break;
			}
			case LensRNG::LKCP204Blue:
			{
				offset = ReadVec2STTexture(px, RNG, _loadedTexture_201);
				break;
            }
            case LensRNG::LKCP204ICDF_White:
            {
                float2 rng = float2(RandomFloat01(RNG), RandomFloat01(RNG));
                offset = SampleICDF(rng, _loadedTexture_202);
                break;
            }
            case LensRNG::LKCP204ICDF_Blue:
            {
                float2 rng = ReadVec2STTexture(px, RNG, _loadedTexture_178, false);
                offset = SampleICDF(rng, _loadedTexture_202);
                break;
            }
		}

		if ((bool)_RayGenCB.JitterNoiseTextures)
		{
			uint jitterRNG = HashInit(px);
			offset += float2((RandomFloat01(jitterRNG) - 0.5f) / 255.0f, (RandomFloat01(jitterRNG) - 0.5f) / 255.0f);
		}
	}
	offset *= _RayGenCB.ApertureRadius;

	// Anamorphic lens effect
	// I'm making an "oval aperture" which is not the real anamorphic effect
	// See this video for more information: https://www.youtube.com/watch?v=_8hCjO-cyqE
	offset *= _RayGenCB.AnamorphicScaling;

	// petzval and occlusion effects
	if (px.x != screenDims.x / 2 || px.y != screenDims.y / 2)
	{
		float2 pxVectorFromCenter = normalize(float2(px.xy) - float2(screenDims) / 2.0f);
		pxVectorFromCenter.y *= -1.0f;

		// Petzval lens effect
		// Scales bokeh on each axis depending on screen position. Fakes the effect. Defaults to 1.0, 1.0 for no elongation.
		{
			float2 pxVectorFromCenterPerpendicular = float2(-pxVectorFromCenter.y, pxVectorFromCenter.x);

			float2 projections = float2(dot(offset, pxVectorFromCenter), dot(offset, pxVectorFromCenterPerpendicular));

			projections *= _RayGenCB.PetzvalScaling;

			offset = projections.x * pxVectorFromCenter + projections.y * pxVectorFromCenterPerpendicular;
		}

		// Occlusion from barrel
		// Pushes the bounding square of the lens outwards and clips against a unit circle. 1,1,1 means no occlusion.
		// x is how far from the center of the screen to start moving the square. 0 is center, 1 is the corner.
		// y is how much to scale the lens bounding square by.
		// z is how far to move the square, as the pixel is farther from where the occlusion begins.
		if (_RayGenCB.OcclusionSettings.x < 1.0f)
		{
			float distance = length(float2(px.xy) - float2(screenDims) / 2.0f);

			float maxDistance = length(float2(screenDims) / 2.0f);
			
			float normalizedDistance = distance / maxDistance;

			float occlusionPercent = clamp((normalizedDistance - _RayGenCB.OcclusionSettings.x) / (1.0f - _RayGenCB.OcclusionSettings.x), 0.0f, 1.0f);
			
			float2 occlusionOffset = offset * _RayGenCB.OcclusionSettings.y;
			occlusionOffset += pxVectorFromCenter * occlusionPercent * _RayGenCB.OcclusionSettings.z;
			
			if (length(occlusionOffset) > 1.0f)
			{
				PDF = 0.0f;
			}
		}
	}

	// update the camera pos
	rayPos += offset.x * cameraRight + offset.y * cameraUp;

	// update the ray direction
	rayDir = normalize(focalPlanePoint - rayPos);

	return PDF;
}

[shader("raygeneration")]
#line 1375
void RayGen()
{
	const uint2 dispatchDimsUInt = DispatchRaysDimensions().xy;
	const float2 dispatchDims = float2(dispatchDimsUInt);
	const uint2 pixelCoord = DispatchRaysIndex().xy;

	Struct_PixelDebugStruct pinholeDebug = (Struct_PixelDebugStruct)0;
	Struct_PixelDebugStruct thinLensDebug = (Struct_PixelDebugStruct)0;
	Struct_PixelDebugStruct lensSimDebug = (Struct_PixelDebugStruct)0;
	Struct_PixelDebugStruct bokehDebug = (Struct_PixelDebugStruct)0;

	float3 pinholeColor = float3(0.0f, 0.0f, 0.0f);
	float3 thinLensColor = float3(0.0f, 0.0f, 0.0f);
	float3 lensSimColor = float3(0.0f, 0.0f, 0.0f);
	float3 bokehColor = float3(0.0f, 0.0f, 0.0f);

	for (uint rayIndex = 0; rayIndex < _RayGenCB.SamplesPerPixelPerFrame; ++rayIndex)
	{
		uint3 px = uint3(pixelCoord, _RayGenCB.FrameIndex * _RayGenCB.SamplesPerPixelPerFrame + rayIndex);
		uint rngBase = HashInit(px);
		uint rngForJitter = rngBase;

		float2 pixelJitter = float2(0.5f, 0.5f);
		switch(_RayGenCB.JitterPixels)
		{
			case PixelJitterType::None: break;
			case PixelJitterType::PerPixel:
			{
				pixelJitter = float2(RandomFloat01(rngForJitter), RandomFloat01(rngForJitter));
				break;
			}
			case PixelJitterType::Global:
			{
				uint RNGGlobal = HashInit(uint3(1337, 0xbeef, px.z));
				pixelJitter = float2(RandomFloat01(RNGGlobal), RandomFloat01(RNGGlobal));
				break;
			}
		}

		float2 screenPos = (float2(pixelCoord) + pixelJitter) / dispatchDims * 2.0f - 1.0f;
		screenPos.y = -screenPos.y;

		float4 world = mul(float4(screenPos, _RayGenCB.DepthNearPlane, 1.0f), _RayGenCB.InvViewProjMtx);
		world.xyz /= world.w;

		Ray baseRay;
		baseRay.Origin = _RayGenCB.CameraPos;
		baseRay.Direction = normalize(world.xyz - _RayGenCB.CameraPos);

		float sampleWeight = 1.0f / float(rayIndex + 1);

		uint rngPinhole = rngForJitter;
		uint rngThinLens = wang_hash(rngPinhole);
		uint rngLensSimulation = wang_hash(rngThinLens);
		uint rngBokeh = wang_hash(rngLensSimulation);

		if (t_renderPinhole)
		{
			float3 sampleColor = ShadeSceneSample(baseRay, 1.0f, pinholeDebug, rayIndex, px, rngPinhole);
			pinholeColor = lerp(pinholeColor, sampleColor, sampleWeight);
		}

		if (t_renderThinLensDoF)
		{
			Ray thinRay = (Ray)0;
			float thinPDF = 1.0f;
			float3 thinOrigin = baseRay.Origin;
			float3 thinDirection = baseRay.Direction;
			thinPDF = ApplyDOFLensSimulation(thinOrigin, thinDirection, px, rngThinLens, dispatchDimsUInt);
			thinRay.Origin = thinOrigin;
			thinRay.Direction = thinDirection;

			if (t_renderThinLensDoF)
			{
				float3 sampleColor = ShadeSceneSample(thinRay, thinPDF, thinLensDebug, rayIndex, px, rngThinLens);
				thinLensColor = lerp(thinLensColor, sampleColor, sampleWeight);
			}
		}

		if (t_renderLensSimulationDoF)
		{
			uint rngLensScene = rngLensSimulation;
			float3 sampleColor = ((bool)_RayGenCB.ToggleChromaticAberration)
				? TraceRealisticChromatic(baseRay, screenPos, dispatchDimsUInt, px, rngLensScene, lensSimDebug, rayIndex, false)
				: TraceRealisticMonochrome(baseRay, screenPos, dispatchDimsUInt, px, rngLensScene, lensSimDebug, rayIndex, false);
			lensSimColor = lerp(lensSimColor, sampleColor, sampleWeight);
		}

		if (t_renderBokehConfig)
		{
			float3 sampleColor = float3(0.0f, 0.0f, 0.0f);

			switch (t_bokehConfigMode)
			{
				case BokehConfigState::NoDoF:
				{
					sampleColor = ShadeVisualFieldSample(baseRay, 1.0f, dispatchDimsUInt);
					break;
				}
				case BokehConfigState::ThinLens:
				{
					Ray thinRay = (Ray)0;
					float thinPDF = 1.0f;
					float3 thinOrigin = baseRay.Origin;
					float3 thinDirection = baseRay.Direction;
					thinPDF = ApplyDOFLensSimulation(thinOrigin, thinDirection, px, rngThinLens, dispatchDimsUInt);
					thinRay.Origin = thinOrigin;
					thinRay.Direction = thinDirection;
					sampleColor = ShadeVisualFieldSample(thinRay, thinPDF, dispatchDimsUInt);
					break;
				}
				case BokehConfigState::RealisticLens:
				{
					uint rngLensBokeh = rngBokeh;
					sampleColor = ((bool)_RayGenCB.ToggleChromaticAberration)
						? TraceRealisticChromatic(baseRay, screenPos, dispatchDimsUInt, px, rngLensBokeh, bokehDebug, rayIndex, true)
						: TraceRealisticMonochrome(baseRay, screenPos, dispatchDimsUInt, px, rngLensBokeh, bokehDebug, rayIndex, true);
					break;
				}
				default:
				{
					break;
				}
			}

			bokehColor = lerp(bokehColor, sampleColor, sampleWeight);
		}
	}

	static const uint c_minFrameIndex = 5;
	float accumulationAlpha = (_RayGenCB.FrameIndex < c_minFrameIndex || !(bool)_RayGenCB.Accumulate || !(bool)_RayGenCB.Animate)
		? 1.0f
		: 1.0f / float(_RayGenCB.FrameIndex - c_minFrameIndex + 1);

	if (t_renderPinhole)
	{
		float3 oldColor = PinholeOut[pixelCoord].rgb;
		float3 blended = lerp(oldColor, pinholeColor, accumulationAlpha);
		PinholeOut[pixelCoord] = float4(blended, 1.0f);
		LinearDepth[pixelCoord] = pinholeDebug.HitT;
	}

	if (t_renderThinLensDoF)
	{
		float3 oldColor = ThinlensOut[pixelCoord].rgb;
		float3 blended = lerp(oldColor, thinLensColor, accumulationAlpha);
		ThinlensOut[pixelCoord] = float4(blended, 1.0f);
	}

	if (t_renderLensSimulationDoF)
	{
		float3 oldColor = LensSimulationOut[pixelCoord].rgb;
		float3 blended = lerp(oldColor, lensSimColor, accumulationAlpha);
		LensSimulationOut[pixelCoord] = float4(blended, 1.0f);
	}

	if (t_renderBokehConfig)
	{
		float3 oldColor = BokehConfigOut[pixelCoord].rgb;
		float3 blended = lerp(oldColor, bokehColor, accumulationAlpha);
		BokehConfigOut[pixelCoord] = float4(blended, 1.0f);
	}

	if (all(uint2(_RayGenCB.MouseState.xy) == pixelCoord))
	{
		pinholeDebug.MousePos = _RayGenCB.MouseState.xy;
		PixelDebug[0] = pinholeDebug;
	}
}

[shader("miss")]
#line 1545
void Miss(inout Payload payload : SV_RayPayload)
{
	payload.hitT = -1.0f;
}

[shader("closesthit")]
#line 1550
void ClosestHit(inout Payload payload : SV_RayPayload, in BuiltInTriangleIntersectionAttributes intersection : SV_IntersectionAttributes)
{
	payload.hitT = RayTCurrent();
	payload.primitiveIndex = PrimitiveIndex();
	payload.bary = intersection.barycentrics;
}

/*
Info about obj format: https://paulbourke.net/dataformats/mtl/

Obj notes:
map_ka
map_kd
map_ks
map_ke
map_d
map_bump

* bump (MASTER_Interior_01_Floor_Tile_Hexagonal_BLENDSHADER)
* d - transparency
* illum - illumination model
* ka - ambient color
* kd - diffuse color
* ks - specular color
* ke - emissive color
* Ni - refraction index
* Ns - specular exponent
* Tr - transparency (1-d)
* Tf - transmission color
*/
