///////////////////////////////////////////////////////////////////////////////
//     Filter-Adapted Spatio-Temporal Sampling With General Distributions    //
//        Copyright (c) 2024 Electronic Arts Inc. All rights reserved.       //
///////////////////////////////////////////////////////////////////////////////

// Unnamed technique, shader RayGen
/*$(ShaderResources)*/

#include "PCG.hlsli"
#include "IndexToColor.hlsli"
#include "LDSShuffler.hlsli"

static const float c_maxT = 10000.0f;
static const float PI = 3.14159265358979323846f;

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
		MATERIAL(0, /*$(Image2D:Assets\BuildingTextures\Paris_Curbstones_01_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Pavement_Cobblestone_Small_BLENDSHADER
		MATERIAL(1, /*$(Image2D:Assets\BuildingTextures\Pavement_Cobblestone_Small_01_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Pavement_Cobblestone_Big_BLENDSHADER
		MATERIAL(2, /*$(Image2D:Assets\BuildingTextures\Pavement_Cobblestone_Big_01_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Pavement_Cobblestone_02
		MATERIAL(3, /*$(Image2D:Assets\BuildingTextures\Pavement_Cobblestone_03_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Pavement_Brick_BLENDSHADER
		MATERIAL(4, /*$(Image2D:Assets\BuildingTextures\Pavement_Brick_01_Diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Pavement_Ground_Wet
		MATERIAL(5, /*$(Image2D:Assets\BuildingTextures\Ground_Wet_01_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Pavement_Manhole_Cover
		MATERIAL(6, /*$(Image2D:Assets\BuildingTextures\Paris_manholecover_01b_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Pavement_Cobblestone_Wet_BLENDSHADER
		MATERIAL(7, /*$(Image2D:Assets\BuildingTextures\Cobble_02B_Diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Pavement_Cobblestone_Wet_Leaves_BLENDSHADER
		MATERIAL(8, /*$(Image2D:Assets\BuildingTextures\Cobble_02B_Diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Pavement_Cobblestone_01_BLENDSHADER
		MATERIAL(9, /*$(Image2D:Assets\BuildingTextures\Pavement_Cobblestone_01_b_Diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Pavement_Cobble_Leaves_BLENDSHADER
		MATERIAL(10, /*$(Image2D:Assets\BuildingTextures\Pavement_Cobblestone_01_b_Diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// MASTER_Brick_Small_Red_BLENDSHADER
		MATERIAL(11, /*$(Image2D:Assets\BuildingTextures\Brick_Small_03_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Concrete3
		MATERIAL(12, /*$(Image2D:Assets\BuildingTextures\concrete_smooth_02_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// MASTER_Focus
		MATERIAL(13, /*$(Image2D:Assets\BuildingTextures\paris_bistroexteriorspotlight_01_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// MASTER_Metal
		MATERIAL(14, /*$(Image2D:Assets\BuildingTextures\Metal_Chrome_01_tint_Diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// MASTER_Focus_Glass
		MATERIAL(15, /*$(Image2D:Assets\BuildingTextures\Paris_BistroExteriorSpotLightGlass_01_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// MASTER_Focus_Ornament
		MATERIAL(16, /*$(Image2D:Assets\BuildingTextures\bistro_woodpanelornament_01_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// MASTER_Bistro_Main_Door
		MATERIAL(17, /*$(Image2D:Assets\BuildingTextures\Bistro_main_door_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// MASTER_Concrete_Grooved
		MATERIAL(18, /*$(Image2D:Assets\BuildingTextures\concrete_grooved_02_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// MASTER_Glass_Exterior
		MATERIAL(19, /*$(Image2D:Assets\BuildingTextures\Glass_Dirty_01_spec.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// MASTER_Light_Bulb
		MATERIAL(20, /*$(Image2D:Assets\BuildingTextures\paris_bistroexteriorspotlight_01_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// MASTER_Wood_Painted3
		MATERIAL(21, /*$(Image2D:Assets\BuildingTextures\Wood_Painted_02_Diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// MASTER_Concrete1
		MATERIAL(22, /*$(Image2D:Assets\BuildingTextures\Bistro_Main_Door_Concrete_01A_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// MASTER_Side_Letters
		MATERIAL_Mapd(23, /*$(Image2D:Assets\BuildingTextures\Paris_BistroSideBanner_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/, /*$(Image2D:Assets\BuildingTextures\Paris_BistroSideBanner_mask.png:R8_UNorm:float:false:false)*/);

		// Balcony_Concrete
		MATERIAL(24, /*$(Image2D:Assets\PropTextures\Street\Paris_MainBalcony_01\Paris_Doors_04A_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Balcony_Trims
		MATERIAL(25, /*$(Image2D:Assets\PropTextures\Street\Paris_MainBalcony_01\Paris_MainBalcony_02A_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Balcony_Ornaments
		MATERIAL(26, /*$(Image2D:Assets\PropTextures\Street\Paris_MainBalcony_01\Paris_MainBalcony_01A_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Balcony_Green_Wood
		MATERIAL(27, /*$(Image2D:Assets\PropTextures\Street\Paris_MainBalcony_01\Paris_Doors_03A_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// MASTER_Glass_Dirty
		MATERIAL_Glass(28, float3(0.2f, 0.2f, 0.2f), float3(0.5f, 0.5f, 0.5f));

		// MASTER_Plastic
		MATERIAL(29, /*$(Image2D:Assets\BuildingTextures\Plastic_01_Planters_Diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// MASTER_Grain_Metal
		MATERIAL(30, /*$(Image2D:Assets\BuildingTextures\grain_metal_01_black_metal_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// MASTER_Concrete_Yellow
		MATERIAL(31, /*$(Image2D:Assets\BuildingTextures\Plaster_01A_yellow_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// MASTER_Concrete
		MATERIAL(32, /*$(Image2D:Assets\BuildingTextures\concrete_smooth_02_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// MASTER_Concrete_Smooth
		MATERIAL(33, /*$(Image2D:Assets\BuildingTextures\Concrete_Smooth_03_Diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// MASTER_Roofing_Shingle_Grey
		MATERIAL(34, /*$(Image2D:Assets\BuildingTextures\Roofing_Shingle_01_Diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// MASTER_Concrete_Plaster1_BLENDSHADER
		MATERIAL(35, /*$(Image2D:Assets\BuildingTextures\Concrete_Smooth_03_Diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// MASTER_Roofing_Shingle_Green
		MATERIAL(36, /*$(Image2D:Assets\BuildingTextures\Roofing_Shingle_03_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// MASTER_Trim_Cornice
		MATERIAL(37, /*$(Image2D:Assets\BuildingTextures\Trim_cornice_01_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// MASTER_Forge_Metal
		MATERIAL_Mapd(38, /*$(Image2D:Assets\BuildingTextures\Paris_Building01_Window_01_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/, /*$(Image2D:Assets\BuildingTextures\Paris_Building01_Window_01_mask.png:R8_UNorm:float:false:false)*/);

		// MASTER_Curtains
		MATERIAL(39, /*$(Image2D:Assets\BuildingTextures\Windows_curtains_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// MASTER_Metal_Pipe
		MATERIAL(40, /*$(Image2D:Assets\BuildingTextures\grain_metal_01_black_metal_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// MASTER_Wood_Polished
		MATERIAL(41, /*$(Image2D:Assets\BuildingTextures\wood_base_01_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// MASTER_Doors
		MATERIAL(42, /*$(Image2D:Assets\BuildingTextures\Paris_Doors_01A_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// MASTER_Brick_Large_Beige_BLENDSHADER
		MATERIAL(43, /*$(Image2D:Assets\BuildingTextures\brick_large_03_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// MASTER_Wood_Brown
		MATERIAL(44, /*$(Image2D:Assets\BuildingTextures\wood_base_01_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// MASTER_Glass_Clean
		MATERIAL_Glass(45, float3(0.2f, 0.2f, 0.2f), float3(0.5f, 0.5f, 0.5f));

		// MASTER_Wood_Painted_Green
		MATERIAL(46, /*$(Image2D:Assets\BuildingTextures\wood_painted_flat_grey_01_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// MASTER_Concrete_Plaster
		MATERIAL(47, /*$(Image2D:Assets\BuildingTextures\Concrete_Smooth_03_Diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// MASTER_Black_Metal
		MATERIAL(48, /*$(Image2D:Assets\BuildingTextures\grain_metal_01_black_metal_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// MASTER_Brick_Large_White
		MATERIAL(49, /*$(Image2D:Assets\BuildingTextures\brick_large_02_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// MASTER_Building_Details
		MATERIAL(50, /*$(Image2D:Assets\BuildingTextures\Paris_Building_09_Details_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// MASTER_Wood_Painted_Cyan
		MATERIAL(51, /*$(Image2D:Assets\BuildingTextures\wood_painted_flat_cyan_01_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// MASTER_Brick_Small_Red
		MATERIAL(52, /*$(Image2D:Assets\BuildingTextures\brick_small_02_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// MASTER_Roofing_Metal_01
		MATERIAL(53, /*$(Image2D:Assets\BuildingTextures\roofing_metal_01_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// MASTER_Room_Interior
		MATERIAL(54, /*$(Image2D:Assets\BuildingTextures\Room_Interior_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// MASTER_Awning_Beams
		MATERIAL(55, /*$(Image2D:Assets\BuildingTextures\Paris_BistroAwning_Beams_01B_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// MASTER_Boulangerie
		MATERIAL(56, /*$(Image2D:Assets\BuildingTextures\Boulangeria_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// MASTER_Frosted_Glass
		MATERIAL(57, /*$(Image2D:Assets\BuildingTextures\frosted_glass_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// MASTER_Awning_Fabric_Cyan
		MATERIAL(58, /*$(Image2D:Assets\BuildingTextures\Paris_BistroAwning_Fabric_Cyan_01A_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// MASTER_Bronze_BLENDSHADER
		MATERIAL(59, /*$(Image2D:Assets\BuildingTextures\Metal_Clean_01_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// MASTER_Concrete_White
		MATERIAL(60, /*$(Image2D:Assets\BuildingTextures\Plaster_01A_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// MASTER_Awning_Fabric_Red
		MATERIAL(61, /*$(Image2D:Assets\BuildingTextures\paris_bistroawning_fabric_01a_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// MASTER_Book_Covers
		MATERIAL(62, /*$(Image2D:Assets\BuildingTextures\Books_Covers_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// MASTER_Concrete_Smooth_BLENDSHADER
		MATERIAL(63, /*$(Image2D:Assets\BuildingTextures\concrete_smooth_01_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// MASTER_Wood_Painted_Green_BLENDSHADER
		MATERIAL(64, /*$(Image2D:Assets\BuildingTextures\Wood_Painted_04_Diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// MASTER_Rollup_Door
		MATERIAL(65, /*$(Image2D:Assets\OtherTextures\Tiling\Metal_RollDoor_01\Metal_RollDoor_01_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// MASTER_Details_Dark
		MATERIAL(66, /*$(Image2D:Assets\BuildingTextures\Paris_Building_09_Details_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Foliage_Bux_Hedges46
		MATERIAL_Mapd(67, /*$(Image2D:Assets\PropTextures\Natural\Buxus\buxus_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/, /*$(Image2D:Assets\PropTextures\Natural\Buxus\buxus_mask.png:R8_UNorm:float:false:false)*/);

		// Foliage_Leaves
		MATERIAL_Mapd(68, /*$(Image2D:Assets\PropTextures\Natural\Italian_Cypress\Italian_Cypress_Leaves_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/, /*$(Image2D:Assets\PropTextures\Natural\Italian_Cypress\Italian_Cypress_Leaves_mask.png:R8_UNorm:float:false:false)*/);

		// Foliage_Trunk
		MATERIAL(69, /*$(Image2D:Assets\PropTextures\Natural\Italian_Cypress\Italian_Cypress_Bark_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Foliage_Flowers
		MATERIAL_Mapd(70, /*$(Image2D:Assets\PropTextures\Natural\Paris_flowers_01a\Paris_Foliage_01a_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/, /*$(Image2D:Assets\PropTextures\Natural\Paris_flowers_01a\Paris_Foliage_01a_mask.png:R8_UNorm:float:false:false)*/);

		// Foliage_Paris_Flowers
		MATERIAL_Mapd(71, /*$(Image2D:Assets\PropTextures\Natural\Paris_flowers_01a\Paris_flowers_01a_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/, /*$(Image2D:Assets\PropTextures\Natural\Paris_flowers_01a\Paris_flowers_01a_mask.png:R8_UNorm:float:false:false)*/);

		// Foliage_Linde_Tree_Large_Trunk
		MATERIAL(72, /*$(Image2D:Assets\PropTextures\Natural\linden\Linden_Bark_A_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Foliage_Linde_Tree_Large_Orange_Leaves
		MATERIAL_Mapd(73, /*$(Image2D:Assets\PropTextures\Natural\linden\Leaves_B_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/, /*$(Image2D:Assets\PropTextures\Natural\linden\Leaves_B_mask.png:R8_UNorm:float:false:false)*/);

		// Foliage_Linde_Tree_Large_Green_Leaves
		MATERIAL_Mapd(74, /*$(Image2D:Assets\PropTextures\Natural\linden\Leaves_A_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/, /*$(Image2D:Assets\PropTextures\Natural\linden\Leaves_A_mask.png:R8_UNorm:float:false:false)*/);

		// Foliage_Ivy_leaf_a
		MATERIAL_Mapd(75, /*$(Image2D:Assets\PropTextures\Natural\Paris_ivy_01a\Paris_ivy_leaf_a_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/, /*$(Image2D:Assets\PropTextures\Natural\Paris_ivy_01a\Paris_ivy_leaf_a_mask.png:R8_UNorm:float:false:false)*/);

		// Foliage_Ivy_branches
		MATERIAL(76, /*$(Image2D:Assets\PropTextures\Natural\Paris_ivy_01a\Paris_ivy_branch_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Streetlight_Glass
		MATERIAL(77, /*$(Image2D:Assets\PropTextures\Street\Paris_StreetLight_01\Paris_StreetLight_Glass_01_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Streetlight_Support_Bulb
		MATERIAL(78, /*$(Image2D:Assets\PropTextures\Street\Paris_StreetLight_01\Paris_StreetLight_Glass_01_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Paris_StringLights_01_White_Color
		MATERIAL_Ke(79, /*$(Image2D:Assets\OtherTextures\Colors\White.png:RGBA8_UNorm_SRGB:float4:true:false)*/, float3(1.0f, 1.0f, 1.0f));

		// Streetlight_Metal
		MATERIAL(80, /*$(Image2D:Assets\PropTextures\Street\Paris_StreetLight_01\Paris_StreetLight_01_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Streetlight_Chains
		MATERIAL_Mapd(81, /*$(Image2D:Assets\PropTextures\Street\Paris_StreetLight_01\Street_decals_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/, /*$(Image2D:Assets\PropTextures\Street\Paris_StreetLight_01\Street_decals_mask.png:R8_UNorm:float:false:false)*/);

		// Paris_Streetpivot
		MATERIAL(82, /*$(Image2D:Assets\PropTextures\Street\Paris_StreetPivot\Paris_StreetPivot_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// ElectricBox
		MATERIAL(83, /*$(Image2D:Assets\PropTextures\Street\Paris_ElectricBox_01\Paris_ElectricBox_01_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Shopsign_Bakery
		MATERIAL(84, /*$(Image2D:Assets\PropTextures\Street\Paris_ShopSign_01\Paris_ShopSign_bakery_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Banner_Metal
		MATERIAL(85, /*$(Image2D:Assets\OtherTextures\Tiling\Metal_Chrome_01\Metal_Chrome_01_banner_metal_Diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Shopsign_Book_Store
		MATERIAL(86, /*$(Image2D:Assets\PropTextures\Street\Paris_ShopSign_01\Paris_ShopSign_bookshop_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Shopsign_Pharmacy
		MATERIAL(87, /*$(Image2D:Assets\PropTextures\Street\Paris_ShopSign_01\Paris_ShopSign_pharmacy_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Shopsign_Ties_Shop
		MATERIAL(88, /*$(Image2D:Assets\PropTextures\Street\Paris_ShopSign_01\Paris_ShopSign_ties shop_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Spotlight_Main
		MATERIAL(89, /*$(Image2D:Assets\PropTextures\Street\Paris_BistroExteriorSpotLight_01\Paris_BistroExteriorSpotLight_01_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Spotlight_Emissive
		MATERIAL_Ke(90, /*$(Image2D:Assets\PropTextures\Street\Paris_BistroExteriorSpotLight_01\Paris_BistroExteriorSpotLight_01_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/, float3(1.0f, 1.0f, 1.0f) * 20.0f);

		// Spotlight_Glass
		MATERIAL_Glass(91, float3(0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 0.0f));

		// Paris_StreetSign_01
		MATERIAL(92, /*$(Image2D:Assets\PropTextures\Street\Paris_StreetSign_01\Paris_StreetSign_01A_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Paris_TrafficSign_A
		MATERIAL(93, /*$(Image2D:Assets\PropTextures\Street\Paris_TrafficSign_01\Paris_TrafficSign_01_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// MenuSign_01
		MATERIAL(94, /*$(Image2D:Assets\PropTextures\Street\Paris_MenuSign_01\Paris_MenuSign_01A_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// MenuSign_02_Mesh
		MATERIAL(95, /*$(Image2D:Assets\PropTextures\Street\Paris_MenuSign_01\Paris_MenuSign_01B_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// MenuSign_02_Glass
		MATERIAL(96, /*$(Image2D:Assets\OtherTextures\Tiling\Glass_Dirty_01\Glass_Dirty_01_Diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Trashcan
		MATERIAL(97, /*$(Image2D:Assets\PropTextures\Street\Paris_TrashCan_01\Paris_TrashCan_01_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Paris_StringLights_01_Green_Color
		MATERIAL_Ke(98, /*$(Image2D:Assets\OtherTextures\Colors\Green.png:RGBA8_UNorm_SRGB:float4:true:false)*/, float3(0.1f, 1.0f, 0.1f));

		// Paris_StringLights_01_Red_Color
		MATERIAL_Ke(99, /*$(Image2D:Assets\OtherTextures\Colors\Red_Full.png:RGBA8_UNorm_SRGB:float4:true:false)*/, float3(1.0f, 0.1f, 0.1f));

		// Stringlights
		MATERIAL(100, /*$(Image2D:Assets\PropTextures\Street\Paris_StringLights_01\Paris_StringLights_01_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Paris_StringLights_01_Blue_Color
		MATERIAL_Ke(101, /*$(Image2D:Assets\OtherTextures\Colors\White.png:RGBA8_UNorm_SRGB:float4:true:false)*/, float3(0.1f, 0.1f, 1.0f));

		// Paris_StringLights_01_Pink_Color
		MATERIAL_Ke(102, /*$(Image2D:Assets\OtherTextures\Colors\Purple.png:RGBA8_UNorm_SRGB:float4:true:false)*/, float3(1.0f, 0.1f, 1.0f));

		// Paris_StringLights_01_Orange_Color
		MATERIAL_Ke(103, /*$(Image2D:Assets\OtherTextures\Colors\Orange.png:RGBA8_UNorm_SRGB:float4:true:false)*/, float3(1.0f, 1.0f, 0.0f));

		// Bollards
		MATERIAL(104, /*$(Image2D:Assets\PropTextures\Street\Paris_Bollard_01\Paris_Bollard_01A_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Vespa_Odometer_Glass
		MATERIAL_Glass(105, float3(0.2f, 0.2f, 0.2f), float3(0.5f, 0.5f, 0.5f));

		// Vespa_Headlight
		MATERIAL_Glass(106, float3(0.2f, 0.2f, 0.2f), float3(0.5f, 0.5f, 0.5f));

		// Vespa
		MATERIAL(107, /*$(Image2D:Assets\PropTextures\Street\Paris_VespaScooter_01\Paris_VespaScooter_01_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Vespa_Odometer
		MATERIAL(108, /*$(Image2D:Assets\PropTextures\Street\Paris_VespaScooter_01\Paris_VespaScooter_01_odometer_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Antenna_Metal
		MATERIAL(109, /*$(Image2D:Assets\OtherTextures\Tiling\Metal_Clean_01\Metal_Clean_01_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Antenna_Plastic_Blue
		MATERIAL(110, /*$(Image2D:Assets\OtherTextures\Tiling\Plastic_02\Plastic_02_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Antenna_Plastic
		MATERIAL(111, /*$(Image2D:Assets\OtherTextures\Tiling\Plastic_02\Plastic_02_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Awnings_Fabric
		MATERIAL(112, /*$(Image2D:Assets\PropTextures\Street\Paris_BistroAwning_01\Paris_BistroAwning_Fabric_01A_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Awnings_Beams
		MATERIAL(113, /*$(Image2D:Assets\PropTextures\Street\Paris_BistroAwning_01\Paris_BistroAwning_Beams_01A_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Awnings_Hotel_Fabric
		MATERIAL(114, /*$(Image2D:Assets\PropTextures\Street\Paris_BistroAwning_02\Paris_BistroAwning_02_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Bistro_Sign_Main
		MATERIAL(115, /*$(Image2D:Assets\PropTextures\Street\Paris_BistroFrontBanner_01\Paris_BistroFrontBanner_01_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Bistro_Sign_Letters
		MATERIAL(116, /*$(Image2D:Assets\OtherTextures\Colors\Grey_80.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Plantpots
		MATERIAL(117, /*$(Image2D:Assets\PropTextures\Street\Paris_PlantPot_01\Paris_PlantPot_01A_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Sidewalkbarrier
		MATERIAL(118, /*$(Image2D:Assets\PropTextures\Street\Paris_SidewalkBarrier_01\Paris_SidewalkBarrier_01_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Concrete
		MATERIAL(119, /*$(Image2D:Assets\OtherTextures\Tiling\Concrete_Smooth_02\Concrete_Smooth_02_Diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Plaster
		MATERIAL(120, /*$(Image2D:Assets\OtherTextures\Tiling\Brick_Large_01\Brick_Large_01_Diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Concrete2
		MATERIAL(121, /*$(Image2D:Assets\OtherTextures\Tiling\Concrete_Smooth_01\Concrete_Smooth_01_Diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Concrete_Striped
		MATERIAL(122, /*$(Image2D:Assets\OtherTextures\Tiling\Concrete_Grooved_02\Concrete_Grooved_02_Diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Chimneys_Metal
		MATERIAL(123, /*$(Image2D:Assets\OtherTextures\Tiling\Grain_Metal_01\Grain_Metal_01_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Lantern
		MATERIAL(124, /*$(Image2D:Assets\PropTextures\Street\Paris_Lantern_01\Paris_Lantern_01A_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Paris_Chair_01
		MATERIAL(125, /*$(Image2D:Assets\PropTextures\Bistro\Paris_Chair_01\Paris_Chair_01A_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Paris_Table_Terrace
		MATERIAL(126, /*$(Image2D:Assets\PropTextures\Bistro\Paris_Table_01\Paris_Table_01A_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Ashtray
		MATERIAL(127, /*$(Image2D:Assets\PropTextures\Bistro\Paris_Ashtray_01\Paris_Ashtray_01A_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// NapkinHolder
		MATERIAL(128, /*$(Image2D:Assets\PropTextures\Bistro\Paris_NapkinHolder_01\Paris_Napkinholder_01_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Paris_LiquorBottle_01_Glass_Wine
		MATERIAL(129, /*$(Image2D:Assets\OtherTextures\Colors\Grey_30.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Paris_LiquorBottle_01_Caps
		MATERIAL(130, /*$(Image2D:Assets\PropTextures\Bistro\Paris_LiquorBottle_01\Paris_LiquorBottle_01_Cap_Diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Paris_LiquorBottle_01_Labels
		MATERIAL_Mapd(131, /*$(Image2D:Assets\PropTextures\Bistro\Paris_LiquorBottle_01\Paris_LiquorBottle_labels_01_Diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/, /*$(Image2D:Assets\PropTextures\Bistro\Paris_LiquorBottle_01\Paris_LiquorBottle_labels_01_Diff.png:R8_UNorm:float:false:false)*/);

		// unknown material goes magenta
		default:
		{
			ret.albedo = float3(1.0f, 0.0f, 1.0f);
			break;
		}
	}

	ret.emissive *= /*$(Variable:MaterialEmissiveMultiplier)*/;

	return ret;
}

MaterialInfo EvaluateMaterial_Interior(uint materialID, float2 UV)
{
	MaterialInfo ret = (MaterialInfo)0;
	ret.alpha = 1.0f;

	switch(materialID)
	{
		// MASTER_Interior_01_Plaster
		MATERIAL(0, /*$(Image2D:Assets\BuildingTextures\concrete_smooth_02_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// MASTER_Interior_01_Wood
		MATERIAL(1, /*$(Image2D:Assets\BuildingTextures\wood_polished_01_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// MASTER_Interior_01_Wooden_stuco
		MATERIAL(2, /*$(Image2D:Assets\BuildingTextures\wood_planks_01_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// MASTER_Wood_Painted3
		MATERIAL(3, /*$(Image2D:Assets\BuildingTextures\Wood_Painted_02_Diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// MASTER_Interior_01_Plaster_Red
		MATERIAL(4, /*$(Image2D:Assets\BuildingTextures\concrete_smooth_02_red_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// MASTER_Interior_01_Grid
		MATERIAL(5, /*$(Image2D:Assets\BuildingTextures\Grey_128.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// MASTER_Interior_01_Brushed_Metal
		MATERIAL(6, /*$(Image2D:Assets\BuildingTextures\paris_brushed_metal_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// MASTER_Interior_01_Plaster2
		MATERIAL(7, /*$(Image2D:Assets\BuildingTextures\concrete_smooth_02_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// MASTER_Interior_01_Frozen_Glass
		MATERIAL(8, /*$(Image2D:Assets\BuildingTextures\frosted_glass_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// MASTER_Interior_01_Floor_Tile_Hexagonal_BLENDSHADER
		MATERIAL(9, /*$(Image2D:Assets\BuildingTextures\tile_hexagonal_01_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// MASTER_Interior_01_Glass
		MATERIAL_Tr(10, /*$(Image2D:Assets\OtherTextures\Colors\Grey_128.png:RGBA8_UNorm_SRGB:float4:true:false)*/, 0.8f);

		// MASTER_Interior_01_Paris_Bartrim
		MATERIAL(11, /*$(Image2D:Assets\BuildingTextures\Paris_BarTrim_01a_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// MASTER_Interior_01_White_Plastic
		MATERIAL(12, /*$(Image2D:Assets\BuildingTextures\White.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// MASTER_Interior_01_Grid1
		MATERIAL(13, /*$(Image2D:Assets\BuildingTextures\Metal_Hexagonal_Grid_01_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// MASTER_Interior_01_Material
		MATERIAL(14, /*$(Image2D:Assets\BuildingTextures\Paris_MenuSign_01B_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// MASTER_Interior_01_Paris_Lantern
		MATERIAL_MapKe_Mapd(15, /*$(Image2D:Assets\PropTextures\Street\Paris_Lantern_01\Paris_Lantern_01A_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/, /*$(Image2D:Assets\PropTextures\Street\Paris_Lantern_01\Paris_Lantern_01A_emi.png:RGBA8_UNorm_SRGB:float4:true:false)*/, /*$(Image2D:Assets\BuildingTextures\Paris_Lantern_01A_mask.png:R8_UNorm:float:false:false)*/);

		// curtainB1
		MATERIAL(16, /*$(Image2D:Assets\PropTextures\Bistro\Paris_Curtain_01\Paris_Curtain_01B_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// curtainA
		MATERIAL(17, /*$(Image2D:Assets\PropTextures\Bistro\Paris_Curtain_01\Paris_Curtain_01A_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Paris_Ceiling_Lamp
		MATERIAL_MapKe(18, /*$(Image2D:Assets\PropTextures\Bistro\Paris_Ceiling_Lamp_01\Paris_Ceiling_Lamp_01_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/, /*$(Image2D:Assets\PropTextures\Bistro\Paris_Ceiling_Lamp_01\Paris_Ceiling_Lamp_01_emi.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Paris_Wall_Light_Interior
		MATERIAL_MapKeRRR(19, /*$(Image2D:Assets\PropTextures\Bistro\Paris_Wall_Light_Interior_01\Paris_Wall_Light_Interior_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/, /*$(Image2D:Assets\PropTextures\Bistro\Paris_Wall_Light_Interior_01\Paris_Wall_Light_Interior_Emi.png:R8_UNorm:float:false:false)*/);

		// Paris_Wall_Bulb_Light
		MATERIAL_Ke(20, /*$(Image2D:Assets\OtherTextures\Colors\White.png:RGBA8_UNorm_SRGB:float4:true:false)*/, float3(1.0f, 1.0f, 1.0f));

		// Wall_Lamp
		MATERIAL_MapKe(21, /*$(Image2D:Assets\PropTextures\Street\Paris_Lantern_01\Paris_Lantern_01A_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/, /*$(Image2D:Assets\PropTextures\Street\Paris_Lantern_01\Paris_Lantern_01A_emi.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Metal_Bronze_01
		MATERIAL(22, /*$(Image2D:Assets\OtherTextures\Tiling\Bronze_01\Bronze_01_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Paris_CeilingFan
		// This had a map_ke but the file doesn't exist on disk
		MATERIAL(23, /*$(Image2D:Assets\PropTextures\Bistro\Paris_CeilingFan_01\Paris_CeilingFan_01A_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Paris_Beertap
		MATERIAL(24, /*$(Image2D:Assets\PropTextures\Bistro\Paris_BeerTap_01\Paris_BeerTap_01_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Paris_LiquorBottle_02_Glass
		MATERIAL_Glass(25, float3(0.2f, 0.1f, 0.0f), float3(0.5f, 0.2f, 0.098f));

		// Paris_LiquorBottle_01_Caps
		MATERIAL(26, /*$(Image2D:Assets\PropTextures\Bistro\Paris_LiquorBottle_01\Paris_LiquorBottle_01_Cap_Diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// ToffeeJar_Label
		MATERIAL(27, /*$(Image2D:Assets\PropTextures\Bistro\Paris_LiquorBottle_01\Paris_LiquorBottle_labels_01_Diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Paris_LiquorBottle_01_Glass_Wine
		MATERIAL(28, /*$(Image2D:Assets\OtherTextures\Colors\Grey_30.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Paris_LiquorBottle_01_Glass
		MATERIAL_Glass(29, float3(0.2f, 0.2f, 0.2f), float3(0.212f, 0.212f, 0.212f));

		// Paris_LiquorBottle_03_Glass
		MATERIAL_Glass(30, float3(0.2f, 0.2f, 0.2f), float3(0.212f, 0.255f, 0.333f));

		// Metal_Worn_01
		MATERIAL(31, /*$(Image2D:Assets\OtherTextures\Tiling\Metal_Clean_01\Metal_Clean_01_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Wood
		MATERIAL(32, /*$(Image2D:Assets\OtherTextures\Tiling\Wood_Planks_01\Wood_Planks_01_Diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Plastic_02
		MATERIAL(33, /*$(Image2D:Assets\OtherTextures\Tiling\Plastic_01\Plastic_01_Diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Paris_Painting_Metal
		MATERIAL(34, /*$(Image2D:Assets\PropTextures\Bistro\Paris_Painting_01\Paris_Painting_01_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Paris_Paintings
		MATERIAL(35, /*$(Image2D:Assets\PropTextures\Bistro\Paris_Painting_01\Paris_Paintings_01_Diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Paris_Paintings_Glass
		MATERIAL(36, /*$(Image2D:Assets\OtherTextures\Tiling\Glass_Dirty_01\Glass_Dirty_01_Diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Paris_BarStool
		MATERIAL(37, /*$(Image2D:Assets\PropTextures\Bistro\Paris_BarStool_01\Paris_BarStool_01A_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// ToffeeJar_Metal
		MATERIAL(38, /*$(Image2D:Assets\OtherTextures\Tiling\Metal_Chrome_01\Metal_Chrome_01_Diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Paris_LiquorBottle_01_Glass1
		MATERIAL_Glass(39, float3(0.03f, 0.05f, 0.04f), float3(0.4f, 0.4f, 0.4f));

		// ToffeeJar_Toffee
		MATERIAL(40, /*$(Image2D:Assets\OtherTextures\Colors\White.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// CookieJar_Glass1
		MATERIAL_Glass(41, float3(0.1f, 0.1f, 0.1f), float3(0.3f, 0.3f, 0.3f));

		// CookieJar_Cookies
		MATERIAL(42, /*$(Image2D:Assets\PropTextures\Bistro\Paris_CookieJar\Cookie_01_Lo_default_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Paris_Radiator
		MATERIAL(43, /*$(Image2D:Assets\PropTextures\Bistro\Paris_Radiator_01\Paris_Radiator_01_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Cutlery_chrome
		MATERIAL(44, /*$(Image2D:Assets\OtherTextures\Tiling\Metal_Chrome_02\Metal_Chrome_02_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Cutlery_details
		MATERIAL_Mapd(45, /*$(Image2D:Assets\PropTextures\Bistro\Paris_Cutlery_01\Paris_Cutlery_01_details_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/, /*$(Image2D:Assets\PropTextures\Bistro\Paris_Cutlery_01\Paris_Cutlery_01_details_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Paris_Table_03
		MATERIAL(46, /*$(Image2D:Assets\OtherTextures\Tiling\Wood_Polished_01\Wood_Polished_01_Diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Plates_Details
		MATERIAL_Mapd(47, /*$(Image2D:Assets\PropTextures\Bistro\Paris_Plates_01\Paris_Plates_details_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/, /*$(Image2D:Assets\PropTextures\Bistro\Paris_Plates_01\Paris_Plates_details_mask.png:R8_UNorm:float:false:false)*/);

		// Plates_Ceramic
		MATERIAL(48, /*$(Image2D:Assets\OtherTextures\Tiling\Ceramic_01\Ceramic_01_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// NapkinHolder_01
		MATERIAL(49, /*$(Image2D:Assets\PropTextures\Bistro\Paris_NapkinHolder_01\Paris_Napkinholder_01_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Paris_Table_cloth_01
		MATERIAL(50, /*$(Image2D:Assets\PropTextures\Bistro\Paris_Cotton_Placemat_01\Paris_Cotton_Placemat_01_Diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Metal_Chrome1
		MATERIAL(51, /*$(Image2D:Assets\OtherTextures\Tiling\Metal_Chrome_01\Metal_Chrome_01_Diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Cloth
		MATERIAL(52, /*$(Image2D:Assets\PropTextures\Bistro\Paris_Wine_Cooler_01\Paris_Cloths_01_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Plants_plants
		MATERIAL_Mapd(53, /*$(Image2D:Assets\PropTextures\Bistro\Paris_Flower_Pot_01\Paris_interior_plants_01_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/,/*$(Image2D:Assets\PropTextures\Bistro\Paris_Flower_Pot_01\Paris_interior_plants_01_mask.png:R8_UNorm:float:false:false)*/);

		// Plants_Metal_Base_01
		MATERIAL(54, /*$(Image2D:Assets\OtherTextures\Tiling\Metal_Clean_01\Metal_Clean_01_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Trolley_Plastic
		MATERIAL(55, /*$(Image2D:Assets\OtherTextures\Tiling\Plastic_01\Plastic_01_Diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Trolley_Wheels
		MATERIAL(56, /*$(Image2D:Assets\PropTextures\Street\Paris_TrashCan_01\Paris_TrashCan_01_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Trolley_Wood_Painted
		MATERIAL(57, /*$(Image2D:Assets\OtherTextures\Tiling\Wood_Painted_02\Wood_Painted_02_Diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// WickerBasket
		MATERIAL(58, /*$(Image2D:Assets\PropTextures\Bistro\Paris_WickerBasket_01\Paris_WickerBasket_01_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Paris_Coasters_01
		MATERIAL(59, /*$(Image2D:Assets\PropTextures\Bistro\Paris_Coasters_01\Paris_Coasters_01_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Paris_Cashregister
		MATERIAL(60, /*$(Image2D:Assets\PropTextures\Bistro\Paris_CashRegister_01\Paris_CashRegister_01_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Paris_Cashregister_Buttons
		MATERIAL(61, /*$(Image2D:Assets\PropTextures\Bistro\Paris_CashRegister_01\Paris_CashRegister_Buttons_01_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Paris_Cashregister_Glass
		MATERIAL_Tr(62, /*$(Image2D:Assets\OtherTextures\Tiling\Glass_Dirty_01\Glass_Dirty_01_Diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/, 0.9f);

		// Paris_Doormat
		MATERIAL(63, /*$(Image2D:Assets\PropTextures\Bistro\Paris_Doormats\Paris_Doormat_01_diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Rubber_Bar_Mat_01
		MATERIAL(64, /*$(Image2D:Assets\OtherTextures\Tiling\Rubber_Bar_Mat_01\Rubber_Bar_Mat_01_Diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// Plastic_01
		MATERIAL(65, /*$(Image2D:Assets\OtherTextures\Tiling\Plastic_01\Plastic_01_Diff.png:RGBA8_UNorm_SRGB:float4:true:false)*/);

		// unknown material goes magenta
		default:
		{
			ret.albedo = float3(1.0f, 0.0f, 1.0f);
			break;
		}
	}

	ret.emissive *= /*$(Variable:MaterialEmissiveMultiplier)*/;

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

	/*$(RayTraceFn)*/(Scene,
		RAY_FLAG_FORCE_OPAQUE, // Ray flags
		0xFF, // Ray mask
		/*$(RTHitGroupIndex:HitGroupPathTrace)*/,
		/*$(RTHitGroupCount)*/,
		/*$(RTMissIndex:Miss)*/,
		ray,
		payload);

	return payload.hitT < 0.0f;
}

float3 SmallLightColor(int index)
{
	float3 ret = /*$(Variable:SmallLightsColor)*/;
	if (/*$(Variable:SmallLightsColorful)*/)
		ret = IndexToColor(index, 0.95f, 0.95f);
	return ret;
}

bool SmallLightContributions(float3 pos, float3 dir, float maxT, out float3 lightColor)
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
			float localHitT = TestSphereTrace(pos, dir, float4(lightPos, /*$(Variable:SmallLightRadius)*/), sphereNormal);
			if (localHitT < 0.0f || localHitT > globalHitT)
				continue;

			globalHitT = localHitT;
			lightColor = SmallLightColor(i) * /*$(Variable:SmallLightBrightness)*/;
			anyHits = true;
		}
	}

	if (!anyHits || !VisibilityQuery(pos, dir, globalHitT))
	{
		lightColor = float3(0.0f, 0.0f, 0.0f);
		return false;
	}

	return true;
}

float3 GetColorForRay(float3 pos, float3 dir, inout uint RNG, inout Struct_PixelDebugStruct pixelDebug, in uint rayIndex, in uint2 px)
{
	float3 throughput = float3(1.0f, 1.0f, 1.0f);
	float3 color = float3(0.0f, 0.0f, 0.0f);

	// show small lights for primary ray
	if(SmallLightContributions(pos, dir, c_maxT, color))
		return color;

	for (uint bounceIndex = 0; bounceIndex < /*$(Variable:NumBounces)*/; ++bounceIndex)
	{
		RayDesc ray;
		ray.Origin = pos;
		ray.TMin = 0;
		ray.TMax = c_maxT;
		ray.Direction = dir;

		Payload payload = (Payload)0;

		/*$(RayTraceFn)*/(Scene,
			RAY_FLAG_FORCE_OPAQUE, // Ray flags
			0xFF, // Ray mask
			/*$(RTHitGroupIndex:HitGroupPathTrace)*/,
			/*$(RTHitGroupCount)*/,
			/*$(RTMissIndex:Miss)*/,
			ray,
			payload);

		// see if the ray hit the small lights
		float3 smallLightColor = float3(0.0f, 0.0f, 0.0f);
		if(SmallLightContributions(ray.Origin, ray.Direction, (payload.hitT < 0.0f ? c_maxT : payload.hitT), smallLightColor))
		{
			color += smallLightColor * throughput;
			return color;
		}

		// on miss, apply sky lighting and we are done
		if (payload.hitT < 0.0f)
		{
			float3 emissive = /*$(Variable:SkyColor)*/ * /*$(Variable:SkyBrightness)*/;
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
		
		switch(/*$(Variable:MaterialSet)*/)
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
				pos = pos + dir * payload.hitT + normal * /*$(Variable:RayPosNormalNudge)*/;
				dir = reflect(dir, normal);
			}
			// transmission
			else
			{
				throughput *= float3(1.0f, 1.0f, 1.0f) - materialInfo.albedo;
				pos = pos + dir * payload.hitT + dir * /*$(Variable:RayPosNormalNudge)*/;
			}
		}
		// Opaque diffuse
		else if(RandomFloat01(RNG) > (1.0f - materialInfo.alpha))
		{
			// apply emissive and albedo
			color += materialInfo.emissive * throughput;
			throughput *= materialInfo.albedo;

			// Opaque bounce
			pos = pos + dir * payload.hitT + normal * /*$(Variable:RayPosNormalNudge)*/;
			dir = normalize(normal + RandomUnitVector(RNG));
		}
		// Transparent
		else
		{
			// transparency
			pos = pos + dir * payload.hitT + dir * /*$(Variable:RayPosNormalNudge)*/;
			//return float3(1.0f, 0.0f, 1.0f);
		}

		if (/*$(Variable:AlbedoMode)*/ && (materialInfo.alpha > 0.0f || materialInfo.emissive.r > 0.0f || materialInfo.emissive.g > 0.0f || materialInfo.emissive.b > 0.0f))
		{
			return materialInfo.albedo * /*$(Variable:AlbedoModeAlbedoMultiplier)*/ + materialInfo.emissive;
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
	switch(/*$(Variable:LensRNGExtend)*/)
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

/************************************************************************************
Spatially Varying Lens Simulation
*************************************************************************************/

struct Ray
{
	float3 Origin;
	float3 Direction;
};

static float4 fishEyeLens[] = {
	// Muller 16mm/f4 155.9FOV fisheye lens			
	// MLD p164			
	// Scaled to 10 mm from 100 mm			
	// radius	sep	n	aperture
	float4(30.2249f, 0.8335f, 1.620f, 30.34f),
	float4(11.3931f, 7.4136f, 1.0f, 20.68f),
	float4(75.2019f, 1.0654f, 1.639f, 17.80f),
	float4(8.3349f, 11.1549f, 1.0f, 13.42f),
	float4(9.5882f, 2.0054f, 1.654f, 9.02f),
	float4(43.8677f, 5.3895f, 1.0f, 8.14f),
	float4(	0.0f, 1.4163f, 0.0f, 6.08f),
	float4(29.4541f, 2.1934f, 1.517f, 5.96f),
	float4(-5.2265f, 0.9714f, 1.805f, 5.84f),
	float4(-14.2884f, 0.0627f, 1.0f, 5.96f),
	float4(-22.3726f, 0.9400f, 1.673f, 5.96f),
	float4(-15.0404f, 25.0f,  1.0f, 6.52), // 12
	float4(0.0f, 0.0f, 0.0f, 0.0f),
	float4(0.0f, 0.0f, 0.0f, 0.0f),
	float4(0.0f, 0.0f, 0.0f, 0.0f),
	float4(0.0f, 0.0f, 0.0f, 0.0f),
};

static float wideAngleLens[] = {
	// Wide-angle (38-degree) lens. Nakamura.			
	// MLD, p. 360"			
	// Scaled to 22 mm from 100 mm			
	// radius   sep	      n       aperture
	float4( 35.98738f, 1.21638f, 1.540f, 23.716f),
	float4( 11.69718f, 9.99570f, 1.000f, 17.996f),
	float4( 13.08714f, 5.12622f, 1.772f, 12.364f),
	float4(-22.63294f, 1.76924f, 1.617f, 9.8120f),
	float4( 71.05802f, 0.81840f, 1.000f, 9.1520f), 
	float4( 0.000000f, 2.27766f, 0.000f, 8.7560f),
	float4(-9.585840f, 2.43254f, 1.617f, 8.1840f),
	float4(-11.28864f, 0.11506f, 1.000f, 9.1520f),
	float4(-166.7765f, 3.09606f, 1.713f, 10.648f),
	float4(-7.591100f,	1.32682f, 1.805f, 11.440f),
	float4(-16.76620f, 3.98068f, 1.000f, 12.276f),
	float4(-7.702860f, 1.21638f, 1.617f, 13.420f),
	float4(-11.97328f, 5.00000f, 1.000f, 17.996f), //13
	float4(0.0f, 0.0f, 0.0f, 0.0f),
	float4(0.0f, 0.0f, 0.0f, 0.0f),
	float4(0.0f, 0.0f, 0.0f, 0.0f),
};

static const float helios_sensor_width = 20.0f;
static const float helios_max_focal_length = 58.0f; // mm
static const float helios_scale = helios_max_focal_length / 100.0f; // helios unit scaling from patent
static const float helios_aperture = 16.0f; // mm widest possible aperture is 16mm
static const float helios_lens_radius = 16.0f;
static const float helios_lens_length = 93.04f * helios_scale; // sum of sep in mm ~53.9632
static const float helios_d_to_film = /*$(Variable:FocalLength)*/; // mm

static float4 helios[] = {
	// Helios 44-2 58mm/f2 lens
	// scaled from 100 units to 58mm
	// 				radius								sep								n			aperture	
	float4(	/*r1*/	83.6f    * helios_scale, 	/*d1*/ 	10.75f * helios_scale, 	/*n1*/	1.64238f, 	helios_lens_radius),
	float4(	/*r2*/	321.0f   * helios_scale, 	/*l1*/	1.65f  * helios_scale, 	/*air*/	1.0f, 		helios_lens_radius),
	float4(	/*r3*/	44.8f    * helios_scale, 	/*d2*/	15.55f * helios_scale, 	/*n2*/	1.62306f, 	helios_lens_radius),
	float4( /*r4*/	-1150.0f * helios_scale, 	/*d3*/	5.05   * helios_scale, 	/*n3*/	1.57566f, 	helios_lens_radius),
	float4( /*r5*/	28.3f    * helios_scale, 	/*l2*/	18.9/2 * helios_scale, 	/*air*/ 1.0f, 		helios_lens_radius),
	float4( /*aperture*/		0.0f, 			/*l2*/	18.9/2 * helios_scale, 	/*air*/ 1.0f, 		helios_aperture),
	float4( /*r6*/  -38.5f   * helios_scale, 	/*d4*/	5.05f  * helios_scale, 	/*n4*/	1.67270f, 	helios_lens_radius),
	float4( /*r7*/	50.5f    * helios_scale, 	/*d5*/	21.22f * helios_scale, 	/*n5*/	1.64238f, 	helios_lens_radius),
	float4( /*r8*/	-53.2f   * helios_scale, 	/*l3*/	0.97f  * helios_scale, 	/*air*/	1.0f, 		helios_lens_radius),
	float4( /*r9*/	106.0f   * helios_scale, 	/*d6*/	13.9f  * helios_scale, 	/*n6*/	1.64238f, 	helios_lens_radius),
	float4( /*r10*/	-120.0f  * helios_scale, 	/*f*/	helios_d_to_film, 		/*air*/	1.0, 		helios_lens_radius), //11
	float4(0.0f, 0.0f, 0.0f, 0.0f),
	float4(0.0f, 0.0f, 0.0f, 0.0f),
	float4(0.0f, 0.0f, 0.0f, 0.0f),
	float4(0.0f, 0.0f, 0.0f, 0.0f),
	float4(0.0f, 0.0f, 0.0f, 0.0f),
};


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

bool traceLensesFromFilm(Ray ray, int elementCount, float4 lensElements[16], out Ray outRay)
{
	float z = 0.0f; // Start at the film, z = 0
	
	for (int i = elementCount - 1; i >= 0; i--)
	{
		const float curvatureRadius = lensElements[i].x;
		const float thickness = lensElements[i].y;
		const float etaI = lensElements[i].z;
		const float etaT = i > 0 ? lensElements[i - 1].z : 1.0f;
		const float apatureRadius = lensElements[i].w;
		
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
		
		float r2 = hit.x * hit.x + hit.y * hit.y;

		if (r2 > (apatureRadius * apatureRadius))
			return false;
		
		ray.Origin = hit;
		
		if (!isStop)
		{
			float3 refractDir = refract(ray.Direction, normal, etaI / (etaT > 0.0f ? etaT : 1.0f));
			if (all(refractDir == 0.0f))
				return false;
			ray.Direction = normalize(refractDir);
		}
	} 
	
	outRay = ray;

	return true;
}

// returns PDF
float ApplyRealisticLensSimulation(inout float3 rayPos, inout float3 rayDir, in uint3 px, inout uint RNG, in uint2 screenDims, in float2 screenPos)
{
	float3 cameraRight = mul(float4(1.0f, 0.0f, 0.0f, 0.0f), /*$(Variable:InvViewMtx)*/).xyz;
	float3 cameraUp = mul(float4(0.0f, 1.0f, 0.0f, 0.0f), /*$(Variable:InvViewMtx)*/).xyz;
	float3 cameraForward = mul(float4(0.0f, 0.0f, 1.0f, 0.0f), /*$(Variable:InvViewMtx)*/).xyz;
	float3 camPos = /*$(Variable:CameraPos)*/;

	// Map normalized screen position ([-1,1]) to film plane coordinates in mm
	// compensate for mirroring along both axis
	float aspect = float(screenDims.x) / float(screenDims.y);
	float sensor_height = helios_sensor_width / aspect;
	float filmX = -screenPos.x * (helios_sensor_width * 0.5f);
	float filmY = -screenPos.y * (sensor_height * 0.5f);

	// Conversion between world units and millimeters for lens/film space
	float worldToMM = 10.0f;

	// Sample a random point on the aperture using polar coordinates
	float theta = RandomFloat01(RNG) * 2 * PI;
	float r = sqrt(RandomFloat01(RNG));
	float2 apertureOffset = float2(cos(theta), sin(theta)) * r;
	apertureOffset *= helios_lens_radius;

	// Construct film-space ray with the sampled aperture offset
	Ray filmRay;
	filmRay.Origin = float3(filmX, filmY, 0.0f);

	// Aim ray at randomly sampled point on innermost lens element
	float3 target = float3(apertureOffset.x, apertureOffset.y, -helios_d_to_film);
	filmRay.Direction = normalize(target - filmRay.Origin);

	// Trace through lens elements
	Ray refracted;
	if (traceLensesFromFilm(filmRay, 11, helios, refracted))
	{
		float invWorldToMM = 1.0f / worldToMM;
		rayPos = camPos +
				 (refracted.Origin.x * invWorldToMM) * cameraRight +
				 (refracted.Origin.y * invWorldToMM) * cameraUp +
				 (refracted.Origin.z * invWorldToMM) * cameraForward;
		rayDir = normalize(
			 refracted.Direction.x * cameraRight +
			 refracted.Direction.y * cameraUp +
			 refracted.Direction.z * cameraForward);
		return 1.0f;
	}
	return 0.0f;
}

/************************************************************************************
End of Spatially Varying Lens Simulation
*************************************************************************************/

// returns PDF
float ApplyDOFLensSimulation(inout float3 rayPos, inout float3 rayDir, in uint3 px, inout uint RNG, in uint2 screenDims)
{

	float3 cameraRight = mul(float4(1.0f, 0.0f, 0.0f, 0.0f), /*$(Variable:InvViewMtx)*/).xyz;
	float3 cameraUp = mul(float4(0.0f, 1.0f, 0.0f, 0.0f), /*$(Variable:InvViewMtx)*/).xyz;

	// calculate point on the focal plane
	float3 focalPlanePoint = rayPos + rayDir * /*$(Variable:FocalLength)*/;

	// calculate a random point on the aperture
	float2 offset = float2(0.0f, 0.0f);

	float PDF = 1.0f;
	if (/*$(Variable:NoImportanceSampling)*/)
	{
		// Random point in square, then we'll adjust the PDF as appropraite
		float2 uvwhite = float2(RandomFloat01(RNG), RandomFloat01(RNG));
        float2 uvblue = ReadVec2STTexture(px, RNG, /*$(Image2DArray:Assets\NoiseTextures\FAST\vector2_uniform_gauss1_0_Gauss10_separate05_%i.png:RG8_UNorm:float2:false:false)*/, false);

		// set uv to either uvwhite or uvblue, depending on the noise type
        float2 uv;
        switch (/*$(Variable:LensRNGSource)*/)
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
		switch(/*$(Variable:LensRNGSource)*/)
		{
			case LensRNG::UniformCircleWhite_PCG:
			case LensRNG::UniformCircleWhite:
			case LensRNG::UniformCircleBlue:
			{
				PDF = 1.0f - /*$(Image2D:Assets\NoiseTextures\UniformCircle\UniformCircle.png:R8_UNorm:float:false:false)*/.SampleLevel(PointClampSampler, uv, 0).r;
				PDF = (PDF > 0.5f) ? 1.0f : 0.0f; // The circle image is anti aliased and that makes white specs from very small probabilities in the circle. This corrects that
				PDF *= 0.781056f;
				break;
			}
			case LensRNG::UniformHexagonWhite:
            case LensRNG::UniformHexagonBlue:
            case LensRNG::UniformHexagonICDF_White:
            case LensRNG::UniformHexagonICDF_Blue:
			{
				PDF = 1.0f - /*$(Image2D:Assets\NoiseTextures\UniformHexagon\UniformHexagon.png:R8_UNorm:float:false:false)*/.SampleLevel(PointClampSampler, uv, 0).r;
				PDF *= 0.652649f;
				break;
            }
			case LensRNG::UniformStarWhite:
            case LensRNG::UniformStarBlue:
            case LensRNG::UniformStarICDF_White:
            case LensRNG::UniformStarICDF_Blue:
            {
				PDF = 1.0f - /*$(Image2D:Assets\NoiseTextures\UniformStar\UniformStar.png:R8_UNorm:float:false:false)*/.SampleLevel(PointClampSampler, uv, 0).r;
				PDF *= 0.368240f;
				break;
			}
			case LensRNG::NonUniformStarWhite:
			case LensRNG::NonUniformStarBlue:
			{
				PDF = 1.0f - /*$(Image2D:Assets\NoiseTextures\NonUniformStar\NonUniformStar.png:R8_UNorm:float:false:false)*/.SampleLevel(PointClampSampler, uv, 0).r;
				// TODO: need to adjust the PDF to account for brightness change
				//PDF *= 0.114055f;
				break;
			}
			case LensRNG::NonUniformStar2White:
			case LensRNG::NonUniformStar2Blue:
			{
				PDF = 1.0f - /*$(Image2D:Assets\NoiseTextures\NonUniformStar2\NonUniformStar2.png:R8_UNorm:float:false:false)*/.SampleLevel(PointClampSampler, uv, 0).r;
				// TODO: need to adjust the PDF to account for brightness change
				//PDF *= 0.051030f;
				break;
			}
			case LensRNG::LKCP6White:
			case LensRNG::LKCP6Blue:
			{
				PDF = /*$(Image2D:Assets\NoiseTextures\Lens_kernel_compositingpro.006\Lens_kernel_compositingpro.006.png:R8_UNorm:float:false:false)*/.SampleLevel(PointClampSampler, uv, 0).r;
				// TODO: need to adjust the PDF to account for brightness change
				//PDF *= 0.206046f;
				break;
			}
			case LensRNG::LKCP204White:
            case LensRNG::LKCP204Blue:
            case LensRNG::LKCP204ICDF_White:
            case LensRNG::LKCP204ICDF_Blue:
			{
				PDF = /*$(Image2D:Assets\NoiseTextures\Lens_kernel_compositingpro.204\Lens_kernel_compositingpro.204.png:R8_UNorm:float:false:false)*/.SampleLevel(PointClampSampler, uv, 0).r;
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
		switch(/*$(Variable:LensRNGSource)*/)
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
				offset = ReadVec2STTexture(px, RNG, /*$(Image2DArray:Assets\NoiseTextures\UniformCircle\UniformCircle_%i.0.png:RG8_UNorm:float2:false:false)*/);
				break;
			}
			case LensRNG::UniformCircleBlue:
			{
				offset = ReadVec2STTexture(px, RNG, /*$(Image2DArray:Assets\NoiseTextures\UniformCircle\UniformCircle_%i.png:RG8_UNorm:float2:false:false)*/);
				break;
			}
			case LensRNG::UniformHexagonWhite:
			{
				offset = ReadVec2STTexture(px, RNG, /*$(Image2DArray:Assets\NoiseTextures\UniformHexagon\UniformHexagon_%i.0.png:RG8_UNorm:float2:false:false)*/);
				break;
			}
			case LensRNG::UniformHexagonBlue:
			{
				offset = ReadVec2STTexture(px, RNG, /*$(Image2DArray:Assets\NoiseTextures\UniformHexagon\UniformHexagon_%i.png:RG8_UNorm:float2:false:false)*/);
				break;
            }
            case LensRNG::UniformHexagonICDF_White:
            {
                float2 rng = float2(RandomFloat01(RNG), RandomFloat01(RNG));
                offset = SampleICDF(rng, /*$(Image2D:Assets\NoiseTextures\UniformHexagon\UniformHexagon.icdf.exr:R32_Float:float:false:false)*/);
                break;
            }
            case LensRNG::UniformHexagonICDF_Blue:
            {
                float2 rng = ReadVec2STTexture(px, RNG, /*$(Image2DArray:Assets\NoiseTextures\FAST\vector2_uniform_gauss1_0_Gauss10_separate05_%i.png:RG8_UNorm:float2:false:false)*/, false);
                offset = SampleICDF(rng, /*$(Image2D:Assets\NoiseTextures\UniformHexagon\UniformHexagon.icdf.exr:R32_Float:float:false:false)*/);
                break;
            }
			case LensRNG::UniformStarWhite:
			{
				offset = ReadVec2STTexture(px, RNG, /*$(Image2DArray:Assets\NoiseTextures\UniformStar\UniformStar_%i.0.png:RG8_UNorm:float2:false:false)*/);
				break;
			}
			case LensRNG::UniformStarBlue:
			{
				offset = ReadVec2STTexture(px, RNG, /*$(Image2DArray:Assets\NoiseTextures\UniformStar\UniformStar_%i.png:RG8_UNorm:float2:false:false)*/);
				break;
            }
            case LensRNG::UniformStarICDF_White:
            {
                float2 rng = float2(RandomFloat01(RNG), RandomFloat01(RNG));
                offset = SampleICDF(rng, /*$(Image2D:Assets\NoiseTextures\UniformStar\UniformStar.icdf.exr:R32_Float:float:false:false)*/);
                break;
            }
            case LensRNG::UniformStarICDF_Blue:
            {
                float2 rng = ReadVec2STTexture(px, RNG, /*$(Image2DArray:Assets\NoiseTextures\FAST\vector2_uniform_gauss1_0_Gauss10_separate05_%i.png:RG8_UNorm:float2:false:false)*/, false);
                offset = SampleICDF(rng, /*$(Image2D:Assets\NoiseTextures\UniformStar\UniformStar.icdf.exr:R32_Float:float:false:false)*/);
                break;
            }
			case LensRNG::NonUniformStarWhite:
			{
				offset = ReadVec2STTexture(px, RNG, /*$(Image2DArray:Assets\NoiseTextures\NonUniformStar\NonUniformStar_%i.0.png:RG8_UNorm:float2:false:false)*/);
				break;
			}
			case LensRNG::NonUniformStarBlue:
			{
				offset = ReadVec2STTexture(px, RNG, /*$(Image2DArray:Assets\NoiseTextures\NonUniformStar\NonUniformStar_%i.png:RG8_UNorm:float2:false:false)*/);
				break;
			}
			case LensRNG::NonUniformStar2White:
			{
				offset = ReadVec2STTexture(px, RNG, /*$(Image2DArray:Assets\NoiseTextures\NonUniformStar2\NonUniformStar2_%i.0.png:RG8_UNorm:float2:false:false)*/);
				break;
			}
			case LensRNG::NonUniformStar2Blue:
			{
				offset = ReadVec2STTexture(px, RNG, /*$(Image2DArray:Assets\NoiseTextures\NonUniformStar2\NonUniformStar2_%i.png:RG8_UNorm:float2:false:false)*/);
				break;
			}
			case LensRNG::LKCP6White:
			{
				offset = ReadVec2STTexture(px, RNG, /*$(Image2DArray:Assets\NoiseTextures\Lens_kernel_compositingpro.006\Lens_kernel_compositingpro.006_%i.0.png:RG8_UNorm:float2:false:false)*/);
				break;
			}
			case LensRNG::LKCP6Blue:
			{
				offset = ReadVec2STTexture(px, RNG, /*$(Image2DArray:Assets\NoiseTextures\Lens_kernel_compositingpro.006\Lens_kernel_compositingpro.006_%i.png:RG8_UNorm:float2:false:false)*/);
				break;
			}
			case LensRNG::LKCP204White:
			{
				offset = ReadVec2STTexture(px, RNG, /*$(Image2DArray:Assets\NoiseTextures\Lens_kernel_compositingpro.204\Lens_kernel_compositingpro.204_%i.0.png:RG8_UNorm:float2:false:false)*/);
				break;
			}
			case LensRNG::LKCP204Blue:
			{
				offset = ReadVec2STTexture(px, RNG, /*$(Image2DArray:Assets\NoiseTextures\Lens_kernel_compositingpro.204\Lens_kernel_compositingpro.204_%i.png:RG8_UNorm:float2:false:false)*/);
				break;
            }
            case LensRNG::LKCP204ICDF_White:
            {
                float2 rng = float2(RandomFloat01(RNG), RandomFloat01(RNG));
                offset = SampleICDF(rng, /*$(Image2D:Assets\NoiseTextures\Lens_kernel_compositingpro.204\Lens_kernel_compositingpro.204.icdf.exr:R32_Float:float:false:false)*/);
                break;
            }
            case LensRNG::LKCP204ICDF_Blue:
            {
                float2 rng = ReadVec2STTexture(px, RNG, /*$(Image2DArray:Assets\NoiseTextures\FAST\vector2_uniform_gauss1_0_Gauss10_separate05_%i.png:RG8_UNorm:float2:false:false)*/, false);
                offset = SampleICDF(rng, /*$(Image2D:Assets\NoiseTextures\Lens_kernel_compositingpro.204\Lens_kernel_compositingpro.204.icdf.exr:R32_Float:float:false:false)*/);
                break;
            }
		}

		if (/*$(Variable:JitterNoiseTextures)*/)
		{
			uint jitterRNG = HashInit(px);
			offset += float2((RandomFloat01(jitterRNG) - 0.5f) / 255.0f, (RandomFloat01(jitterRNG) - 0.5f) / 255.0f);
		}
	}
	offset *= /*$(Variable:ApertureRadius)*/;

	// Anamorphic lens effect
	// I'm making an "oval aperture" which is not the real anamorphic effect
	// See this video for more information: https://www.youtube.com/watch?v=_8hCjO-cyqE
	offset *= /*$(Variable:AnamorphicScaling)*/;

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

			projections *= /*$(Variable:PetzvalScaling)*/;

			offset = projections.x * pxVectorFromCenter + projections.y * pxVectorFromCenterPerpendicular;
		}

		// Occlusion from barrel
		// Pushes the bounding square of the lens outwards and clips against a unit circle. 1,1,1 means no occlusion.
		// x is how far from the center of the screen to start moving the square. 0 is center, 1 is the corner.
		// y is how much to scale the lens bounding square by.
		// z is how far to move the square, as the pixel is farther from where the occlusion begins.
		if (/*$(Variable:OcclusionSettings)*/.x < 1.0f)
		{
			float distance = length(float2(px.xy) - float2(screenDims) / 2.0f);

			float maxDistance = length(float2(screenDims) / 2.0f);
			
			float normalizedDistance = distance / maxDistance;

			float occlusionPercent = clamp((normalizedDistance - /*$(Variable:OcclusionSettings)*/.x) / (1.0f - /*$(Variable:OcclusionSettings)*/.x), 0.0f, 1.0f);
			
			float2 occlusionOffset = offset * /*$(Variable:OcclusionSettings)*/.y;
			occlusionOffset += pxVectorFromCenter * occlusionPercent * /*$(Variable:OcclusionSettings)*/.z;
			
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

/*$(_raygeneration:RayGen)*/
{
	const float2 dimensions = float2(DispatchRaysDimensions().xy);
	Struct_PixelDebugStruct pixelDebug = (Struct_PixelDebugStruct)0;
	uint3 px;

	// Average N rays per pixel into "color"
	float3 color = float3(0.0f, 0.0f, 0.0f);
	for (uint rayIndex = 0; rayIndex < /*$(Variable:SamplesPerPixelPerFrame)*/; ++rayIndex)
	{
		px = uint3(DispatchRaysIndex().xy, /*$(Variable:FrameIndex)*/ * /*$(Variable:SamplesPerPixelPerFrame)*/ + rayIndex);
		uint RNG = HashInit(px);

		// Calculate the ray target in screen space
		// Use sub pixel jitter to integrate over the whole pixel for anti aliasing.
		float2 pixelJitter = float2(0.5f, 0.5f);
		switch(/*$(Variable:JitterPixels)*/)
		{
			case PixelJitterType::None: break;
			case PixelJitterType::PerPixel:
			{
				pixelJitter = float2(RandomFloat01(RNG), RandomFloat01(RNG));
				break;
			}
			case PixelJitterType::Global:
			{
				uint RNGGlobal = HashInit(uint3(1337, 0xbeef, px.z));
				pixelJitter = float2(RandomFloat01(RNGGlobal), RandomFloat01(RNGGlobal));
				break;
			}
		}
		float2 screenPos = (float2(px.xy)+pixelJitter) / dimensions * 2.0 - 1.0;
		screenPos.y = -screenPos.y;

		// Convert the ray target into world space
		float4 world = mul(float4(screenPos, /*$(Variable:DepthNearPlane)*/, 1), /*$(Variable:InvViewProjMtx)*/);
		world.xyz /= world.w;

		// Apply depth of field through lens simulation
		float3 rayPos = /*$(Variable:CameraPos)*/;
		float3 rayDir = normalize(world.xyz - /*$(Variable:CameraPos)*/);
		float PDF = 1.0f;
		if (/*$(Variable:DOF)*/ == DOFMode::Realistic)
			PDF = ApplyRealisticLensSimulation(rayPos, rayDir, px, RNG, DispatchRaysDimensions().xy, screenPos);
		if (/*$(Variable:DOF)*/ == DOFMode::PathTraced)
			PDF = ApplyDOFLensSimulation(rayPos, rayDir, px, RNG, DispatchRaysDimensions().xy);

		// Shoot the ray
		float3 rayColor = (PDF > 0.0f) ? GetColorForRay(rayPos, rayDir, RNG, pixelDebug, rayIndex, px.xy) / PDF : float3(0.0f, 0.0f, 0.0f);
		//rayColor *= 1000;

		// accumualate the sample
		color = lerp(color, rayColor, 1.0f / float(rayIndex+1));
	}

	// Temporally accumulate "color"
	float3 oldColor = Output[px.xy].rgb;
	static const uint c_minFrameIndex = 5;
	float alpha = (/*$(Variable:FrameIndex)*/ < c_minFrameIndex || !/*$(Variable:Accumulate)*/ || !/*$(Variable:Animate)*/) ? 1.0f : 1.0f / float(/*$(Variable:FrameIndex)*/ - c_minFrameIndex +1);

	color = lerp(oldColor, color, alpha);

	// Write the temporally accumulated color
	Output[px.xy] = float4(color, 1.0f);
	LinearDepth[px.xy] = pixelDebug.HitT;

	// Write pixel debug information for whatever pixel was clicked on
	if (all(uint2(/*$(Variable:MouseState)*/.xy) == px.xy))
	{
		pixelDebug.MousePos = /*$(Variable:MouseState)*/.xy;
		PixelDebug[0] = pixelDebug;
	}
}

/*$(_miss:Miss)*/
{
	payload.hitT = -1.0f;
}

/*$(_closesthit:ClosestHit)*/
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
