///////////////////////////////////////////////////////////////////////////////
//     Filter-Adapted Spatio-Temporal Sampling With General Distributions    //
//        Copyright (c) 2024 Electronic Arts Inc. All rights reserved.       //
///////////////////////////////////////////////////////////////////////////////

#define _CRT_SECURE_NO_WARNINGS

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb/stb_image_write.h"

#include <stdio.h>
#include <array>
#include <vector>
#include <direct.h>
#include <random>

#define RANDOM_SHUFFLE_SEED() 1337 // use 0 to make it non deterministic

using Vec2 = std::array<float, 2>;

static const int c_imageSize = 512;

static const bool c_writeU8Points = false;
static const bool c_writeU8JitteredPoints = false;

template <typename T>
T Clamp(const T& value, const T& themin, const T& themax)
{
	if (value <= themin)
		return themin;
	if (value >= themax)
		return themax;
	return value;
}

bool FileExists(const char* fileName)
{
	FILE* file = nullptr;
	fopen_s(&file, fileName, "rb");
	if (!file)
		return false;
	fclose(file);
	return true;
}

bool LoadPoints(const char* fileName, std::vector<Vec2>& points)
{
	FILE* file = nullptr;
	fopen_s(&file, fileName, "rb");
	if (!file)
		return false;

	unsigned int pointCount = 0;
	if (fscanf(file, "%u", &pointCount) != 1)
	{
		fclose(file);
		return false;
	}

	points.resize(pointCount);
	for (Vec2& p : points)
	{
		if (fscanf(file, "%f %f", &p[0], &p[1]) != 2)
		{
			fclose(file);
			return false;
		}
		p[1] = 1.0f - p[1];
	}
	fclose(file);
	return true;
}

bool Process(const char* baseFileNameIn, const char* baseFileNameOut, std::mt19937& rng)
{
	std::vector<Vec2> points;

	printf("Processing %s to %s...\n", baseFileNameIn, baseFileNameOut);

	// open the .bin file
	FILE* binFile = nullptr;
	{
		char fileName[1024];
		sprintf_s(fileName, "%s.bin", baseFileNameOut);
		fopen_s(&binFile, fileName, "wb");
		if (!binFile)
		{
			printf("Could not open for writing: \"%s\"", fileName);
			return false;
		}
	}

	int fileIndex = 0;
	while (true)
	{
		// Load the points
		{
			char fileName[1024];
			sprintf_s(fileName, "%s%i.txt", baseFileNameIn, fileIndex);

			if (!FileExists(fileName))
			{
				if (fileIndex == 0)
					printf("ERROR: no points found!");
				printf("\n");
				fclose(binFile);
				return fileIndex > 0;
			}

			if (!LoadPoints(fileName, points))
			{
				printf("Could not load points from \"%s\"", fileName);
				fclose(binFile);
				return false;
			}

			// shuffle the points to ensure no correlation - we want them to be arranged in a white noise order.
			std::shuffle(points.begin(), points.end(), rng);

			printf("%s\n", fileName);
		}

		// Draw the first set of points as an image
		if (fileIndex == 0)
		{
			// float points
			{
				std::vector<unsigned char> pixels(c_imageSize * c_imageSize, 255);

				for (const Vec2& p : points)
				{
					int px = Clamp<int>(int(p[0] * float(c_imageSize - 1) + 0.5f), 0, c_imageSize - 1);
					int py = Clamp<int>(int(p[1] * float(c_imageSize - 1) + 0.5f), 0, c_imageSize - 1);

					pixels[py * c_imageSize + px] = 0;
				}

				char fileName[1024];
				sprintf_s(fileName, "%s.%i.png", baseFileNameOut, fileIndex);
				stbi_write_png(fileName, c_imageSize, c_imageSize, 1, pixels.data(), 0);
			}

			// U8 points
			if (c_writeU8Points)
			{
				std::vector<unsigned char> pixels(c_imageSize * c_imageSize, 255);

				for (const Vec2& p : points)
				{
					float x = std::floor(float(p[0]) * 255.0f + 0.5f) / 255.0f;
					float y = std::floor(float(p[1]) * 255.0f + 0.5f) / 255.0f;

					int px = Clamp<int>(int(x * float(c_imageSize - 1) + 0.5f), 0, c_imageSize - 1);
					int py = Clamp<int>(int(y * float(c_imageSize - 1) + 0.5f), 0, c_imageSize - 1);

					pixels[py * c_imageSize + px] = 0;
				}

				char fileName[1024];
				sprintf_s(fileName, "%s.U8.%i.png", baseFileNameOut, fileIndex);
				stbi_write_png(fileName, c_imageSize, c_imageSize, 1, pixels.data(), 0);
			}

			// jittered U8 points
			if (c_writeU8JitteredPoints)
			{
				unsigned int seed = RANDOM_SHUFFLE_SEED();
				if (!seed)
				{
					std::random_device rd;
					seed = rd();
				}
				std::mt19937 rngJitter(seed);
				std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

				std::vector<unsigned char> pixels(c_imageSize * c_imageSize, 255);

				for (const Vec2& p : points)
				{
					float x = std::floor(float(p[0]) * 255.0f + 0.5f) / 255.0f;
					float y = std::floor(float(p[1]) * 255.0f + 0.5f) / 255.0f;

					x += dist(rngJitter) / 255.0f;
					y += dist(rngJitter) / 255.0f;

					int px = Clamp<int>(int(x * float(c_imageSize - 1) + 0.5f), 0, c_imageSize - 1);
					int py = Clamp<int>(int(y * float(c_imageSize - 1) + 0.5f), 0, c_imageSize - 1);

					pixels[py * c_imageSize + px] = 0;
				}

				char fileName[1024];
				sprintf_s(fileName, "%s.U8Jittered.%i.png", baseFileNameOut, fileIndex);
				stbi_write_png(fileName, c_imageSize, c_imageSize, 1, pixels.data(), 0);
			}
		}

		// Dump the points to the bin file, as float4s with 0.0 for z and 1.0 for w.
		{
			for (const Vec2& v : points)
			{
				fwrite(&v[0], sizeof(v[0]), 1, binFile);
				fwrite(&v[1], sizeof(v[1]), 1, binFile);
				float dummy = 0.0f;
				fwrite(&dummy, sizeof(dummy), 1, binFile);
				dummy = 1.0f;
				fwrite(&dummy, sizeof(dummy), 1, binFile);
			}
		}

		// move to the next file
		fileIndex++;
	}

	fclose(binFile);
	return true;
}

int main(int argc, char** argv)
{
	_mkdir("out");

	unsigned int seed = RANDOM_SHUFFLE_SEED();
	if (!seed)
	{
		std::random_device rd;
		seed = rd();
	}

	std::mt19937 rng(seed);

	if (!Process("PointSets/UniformStar", "out/UniformStar", rng))
		return 1;

	if (!Process("PointSets/Lens", "out/Lens_kernel_compositingpro.204", rng))
		return 1;

	if (!Process("PointSets/UniformCircle", "out/UniformCircle", rng))
		return 1;

	if (!Process("PointSets/Hexagon", "out/UniformHexagon", rng))
		return 1;

	if (!Process("PointSets/Lens_kernel_compositingpro.006.", "out/Lens_kernel_compositingpro.006", rng))
		return 1;

	if (!Process("PointSets/NonUniformStar.", "out/NonUniformStar", rng))
		return 1;

	if (!Process("PointSets/NonUniformStar2.", "out/NonUniformStar2", rng))
		return 1;

	if (!Process("PointSets/UniformSquare.", "out/UniformSquare", rng))
		return 1;

	return 0;
}