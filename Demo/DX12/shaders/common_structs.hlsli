struct DebugInfo{
	ContextGather ui;
	uint3 px;
	int2 offset;
	float scale_debug;
	float line_thickness;
	float sensor_height;
	float sensor_width;
	float4 color;
};

struct Ray
{
	float3 Origin;
	float3 Direction;
};