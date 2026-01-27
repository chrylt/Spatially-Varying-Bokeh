struct Ray
{
	float3 Origin;
	float3 Direction;
};

struct PixelInfo{
	uint MaterialID;
	float HitT;
	float3 WorldPos;
};