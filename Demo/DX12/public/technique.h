///////////////////////////////////////////////////////////////////////////////
//     Filter-Adapted Spatio-Temporal Sampling With General Distributions    //
//        Copyright (c) 2025 Electronic Arts Inc. All rights reserved.       //
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "../private/technique.h"
#include <string>
#include <vector>
#include "DX12Utils/logfn.h"
#include "DX12Utils/dxutils.h"

namespace FastBokeh
{
    // Compile time technique settings. Feel free to modify these.
    static const int c_numSRVDescriptors = 4096;  // If 0, no heap will be created. One heap shared by all contexts of this technique.
    static const int c_numRTVDescriptors = 256;  // If 0, no heap will be created. One heap shared by all contexts of this technique.
    static const int c_numDSVDescriptors = 256;  // If 0, no heap will be created. One heap shared by all contexts of this technique.
    static const bool c_debugShaders = true; // If true, will compile shaders with debug info enabled.
    static const bool c_debugNames = true; // If true, will set debug names on objects. If false, debug names should be deadstripped from the executable.

    // Information about the technique
    static const bool c_requiresRaytracing = true; // If true, this technique will not work without raytracing support

    using TPerfEventBeginFn = void (*)(const char* name, ID3D12GraphicsCommandList* commandList, int index);
    using TPerfEventEndFn = void (*)(ID3D12GraphicsCommandList* commandList);

    struct ProfileEntry
    {
        const char* m_label = nullptr;
        float m_gpu = 0.0f;
        float m_cpu = 0.0f;
    };

    struct Context
    {
        static const char* GetTechniqueName()
        {
            return "FastBokeh";
        }

        static const wchar_t* GetTechniqueNameW()
        {
            return L"FastBokeh";
        }

        // This is the input to the technique that you are expected to fill out
        struct ContextInput
        {

            // Variables
            bool variable_Reset = false;
            uint2 variable_RenderSize = {512, 512};
            MaterialSets variable_MaterialSet = MaterialSets::None;
            bool variable_Accumulate = true;
            bool variable_Animate = true;
            uint variable_SamplesPerPixelPerFrame = 1;
            PixelJitterType variable_JitterPixels = PixelJitterType::PerPixel;  // Provides Antialiasing
            float variable_DepthNearPlane = 0.100000f;
            float3 variable_CameraPos = {0.0f,0.0f,0.0f};
            float variable_RayPosNormalNudge = 0.100000f;
            bool variable_CameraChanged = false;
            float4x4 variable_InvViewProjMtx = {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
            float4 variable_MouseState = {0.0f,0.0f,0.0f,0.0f};
            uint variable_NumBounces = 4;  // How many bounces the rays are allowed
            bool variable_AlbedoMode = 1.0f;  // if true, returns albedo * AlbedoModeAlbedoMultiplier + emissive at primary hit
            float variable_AlbedoModeAlbedoMultiplier = 0.500000f;  // How much to multiply albedo by in albedo mode, to darken it or lighten it
            float variable_FocalLength = 1.000000f;
            LensRNG variable_LensRNGSource = LensRNG::UniformCircleWhite;
            NoiseTexExtends variable_LensRNGExtend = NoiseTexExtends::None;  // How to extend the noise textures
            bool variable_JitterNoiseTextures = false;  // The noise textures are 8 bit unorms. This adds a random value between -0.5/255 and +0.5/255 to fill in the unset bits with white noise.
            DOFMode variable_DOF = DOFMode::Off;
            float4x4 variable_InvViewMtx = {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
            float variable_ApertureRadius = 1.000000f;
            float2 variable_AnamorphicScaling = {1.0f, 1.0f};  // Defaults to 1.0, 1.0 for no anamorphic effects. Elongates the aperture, does not simulate anamorphic elements.
            float2 variable_PetzvalScaling = {1.0f, 1.0f};  // Scales bokeh on each axis depending on screen position. Fakes the effect. Defaults to 1.0, 1.0 for no elongation.
            float3 variable_OcclusionSettings = {1.0f, 1.0f, 1.0f};  // Pushes the bounding square of the lens outwards and clips against a unit circle. 1,1,1 means no occlusion. x is how far from the center of the screen to start moving the square. 0 is center, 1 is the corner.  y is how much to scale the lens bounding square by.  z is how far to move the square, as the pixel is farther from where the occlusion begins. Reasonable settings are 0, 0.1, 1.25.
            bool variable_NoImportanceSampling = false;  // If true, the FAST noise textures will not be used, and 
            float3 variable_SkyColor = {1.0f, 1.0f, 1.0f};
            float variable_SkyBrightness = 10.000000f;
            float variable_MaterialEmissiveMultiplier = 1.000000f;
            float variable_SmallLightBrightness = 1.000000f;
            float3 variable_SmallLightsColor = {1.0f, 1.0f, 1.0f};
            bool variable_SmallLightsColorful = false;  // If true, makes the small lights colorful, else makes them all the same color
            float variable_SmallLightRadius = 1.000000f;
            bool variable_GatherDOF_UseNoiseTextures = false;
            bool variable_GatherDOF_AnimateNoiseTextures = true;
            bool variable_GatherDOF_SuppressBokeh = false;  // If true, blurs out of focus areas, but reduces the Bokeh effect of small bright lights
            float variable_GatherDOF_FocalDistance = 500.000000f;  // Anything closer than this is considered near field
            float variable_GatherDOF_FocalRegion = 100.000000f;  // The size in world units of the middle range which is in focus
            float variable_GatherDOF_FocalLength = 75.000000f;  // Focal length in mm (Camera property e.g. 75mm)
            float variable_GatherDOF_NearTransitionRegion = 50.000000f;  // Fade distance in world units
            float variable_GatherDOF_FarTransitionRegion = 200.000000f;  // Fade distance in world units
            float variable_GatherDOF_Scale = 0.500000f;  // Camera property e.g. 0.5f, like aperture
            bool variable_GatherDOF_DoFarField = true;  // Whether or not to do the far field
            bool variable_GatherDOF_DoFarFieldFloodFill = true;  // Whether to do flood fill on the far field
            bool variable_GatherDOF_DoNearField = true;  // Whether or not to do the near field
            bool variable_GatherDOF_DoNearFieldFloodFill = true;  // Whether to do flood fill on the near field
            float4 variable_GatherDOF_KernelSize = {10.0f, 15.0f, 5.0f, 0.0f};  // x = size of the bokeh blur radius in texel space. y = rotation in radians to apply to the bokeh shape. z = Number of edge of the polygon (number of blades). 0: circle. 4: square, 6: hexagon...
            uint variable_GatherDOF_BlurTapCount = 8;  // 8 for high quality, 6 for low quality. Used in a double for loop, so it's this number squared.
            uint variable_GatherDOF_FloodFillTapCount = 4;  // 4 for high quality, 3 for low quality. Used in a double for loop, so it's this number squared.
            float variable_GaussBlur_Sigma = 1.000000f;  // Strength of blur. Standard deviation of gaussian distribution.
            bool variable_GaussBlur_Disable = false;
            float variable_TemporalAccumulation_Alpha = 0.100000f;  // For exponential moving average. From 0 to 1. TAA commonly uses 0.1.
            bool variable_TemporalAccumulation_Enabled = true;
            float variable_ToneMap_ExposureFStops = 0.000000f;
            ToneMap_ToneMappingOperation variable_ToneMap_ToneMapper = ToneMap_ToneMappingOperation::None;

            ID3D12Resource* buffer_Scene = nullptr;
            DXGI_FORMAT buffer_Scene_format = DXGI_FORMAT_UNKNOWN; // For typed buffers, the type of the buffer
            unsigned int buffer_Scene_stride = 0; // For structured buffers, the size of the structure
            unsigned int buffer_Scene_count = 0; // How many items there are
            D3D12_RESOURCE_STATES buffer_Scene_state = D3D12_RESOURCE_STATE_COMMON;

            static const D3D12_RESOURCE_FLAGS c_buffer_Scene_flags =  D3D12_RESOURCE_FLAG_NONE; // Flags the buffer needs to have been created with

            // TLAS and BLAS information. Required for raytracing. Fill out using CreateManagedTLAS().
            // The resource itself is the tlas.
            unsigned int buffer_Scene_tlasSize = 0;
            ID3D12Resource* buffer_Scene_blas = nullptr;
            unsigned int buffer_Scene_blasSize = 0;

            ID3D12Resource* buffer_VertexBuffer = nullptr;
            DXGI_FORMAT buffer_VertexBuffer_format = DXGI_FORMAT_UNKNOWN; // For typed buffers, the type of the buffer
            unsigned int buffer_VertexBuffer_stride = 0; // For structured buffers, the size of the structure
            unsigned int buffer_VertexBuffer_count = 0; // How many items there are
            D3D12_RESOURCE_STATES buffer_VertexBuffer_state = D3D12_RESOURCE_STATE_COMMON;

            static const D3D12_RESOURCE_FLAGS c_buffer_VertexBuffer_flags =  D3D12_RESOURCE_FLAG_NONE; // Flags the buffer needs to have been created with
        };
        ContextInput m_input;

        // This is the output of the technique that you can consume
        struct ContextOutput
        {

            ID3D12Resource* texture_GatherDOF_Output = nullptr;
            unsigned int texture_GatherDOF_Output_size[3] = { 0, 0, 0 };
            unsigned int texture_GatherDOF_Output_numMips = 0;
            DXGI_FORMAT texture_GatherDOF_Output_format = DXGI_FORMAT_UNKNOWN;
            static const D3D12_RESOURCE_FLAGS texture_GatherDOF_Output_flags =  D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
            const D3D12_RESOURCE_STATES c_texture_GatherDOF_Output_endingState = D3D12_RESOURCE_STATE_COPY_SOURCE;

            ID3D12Resource* texture_GaussBlur_Output = nullptr;
            unsigned int texture_GaussBlur_Output_size[3] = { 0, 0, 0 };
            unsigned int texture_GaussBlur_Output_numMips = 0;
            DXGI_FORMAT texture_GaussBlur_Output_format = DXGI_FORMAT_UNKNOWN;
            static const D3D12_RESOURCE_FLAGS texture_GaussBlur_Output_flags =  D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
            const D3D12_RESOURCE_STATES c_texture_GaussBlur_Output_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

            ID3D12Resource* texture_TemporalAccumulation_Accum = nullptr;
            unsigned int texture_TemporalAccumulation_Accum_size[3] = { 0, 0, 0 };
            unsigned int texture_TemporalAccumulation_Accum_numMips = 0;
            DXGI_FORMAT texture_TemporalAccumulation_Accum_format = DXGI_FORMAT_UNKNOWN;
            static const D3D12_RESOURCE_FLAGS texture_TemporalAccumulation_Accum_flags =  D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
            const D3D12_RESOURCE_STATES c_texture_TemporalAccumulation_Accum_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

            ID3D12Resource* texture_ToneMap_Color_SDR = nullptr;
            unsigned int texture_ToneMap_Color_SDR_size[3] = { 0, 0, 0 };
            unsigned int texture_ToneMap_Color_SDR_numMips = 0;
            DXGI_FORMAT texture_ToneMap_Color_SDR_format = DXGI_FORMAT_UNKNOWN;
            static const D3D12_RESOURCE_FLAGS texture_ToneMap_Color_SDR_flags =  D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
            const D3D12_RESOURCE_STATES c_texture_ToneMap_Color_SDR_endingState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        };
        ContextOutput m_output;

        // Internal storage for the technique
        ContextInternal m_internal;

        // If true, will do both cpu and gpu profiling. Call ReadbackProfileData() on the context to get the profiling data.
        bool m_profile = false;
        const ProfileEntry* ReadbackProfileData(ID3D12CommandQueue* commandQueue, int& numItems);

        // Set this static function pointer to your own log function if you want to recieve callbacks on info, warnings and errors.
        static TLogFn LogFn;

        // These callbacks are for perf instrumentation, such as with Pix.
        static TPerfEventBeginFn PerfEventBeginFn;
        static TPerfEventEndFn PerfEventEndFn;

        // The path to where the shader files for this technique are. Defaults to L"./"
        static std::wstring s_techniqueLocation;

        static int GetContextCount();
        static Context* GetContext(int index);

        // Buffer Creation
        template <typename T>
        ID3D12Resource* CreateManagedBuffer(ID3D12Device* device, ID3D12GraphicsCommandList* commandList, D3D12_RESOURCE_FLAGS flags, const T* data, size_t count, const wchar_t* debugName, D3D12_RESOURCE_STATES desiredState = D3D12_RESOURCE_STATE_COPY_DEST)
        {
            return CreateManagedBuffer(device, commandList, flags, (void*)data, count * sizeof(T), debugName, desiredState);
        }

        template <typename T>
        ID3D12Resource* CreateManagedBuffer(ID3D12Device* device, ID3D12GraphicsCommandList* commandList, D3D12_RESOURCE_FLAGS flags, const T& data, const wchar_t* debugName, D3D12_RESOURCE_STATES desiredState = D3D12_RESOURCE_STATE_COPY_DEST)
        {
            return CreateManagedBuffer(device, commandList, flags, (void*)&data, sizeof(T), debugName, desiredState);
        }

        template <typename T>
        ID3D12Resource* CreateManagedBuffer(ID3D12Device* device, ID3D12GraphicsCommandList* commandList, D3D12_RESOURCE_FLAGS flags, const std::vector<T>& data, const wchar_t* debugName, D3D12_RESOURCE_STATES desiredState = D3D12_RESOURCE_STATE_COPY_DEST)
        {
            return CreateManagedBuffer(device, commandList, flags, (void*)data.data(), data.size() * sizeof(T), debugName, desiredState);
        }

        ID3D12Resource* CreateManagedBuffer(ID3D12Device* device, ID3D12GraphicsCommandList* commandList, D3D12_RESOURCE_FLAGS flags, const void* data, size_t size, const wchar_t* debugName, D3D12_RESOURCE_STATES desiredState = D3D12_RESOURCE_STATE_COPY_DEST);

        // Texture Creation

        ID3D12Resource* CreateManagedTexture(ID3D12Device* device, ID3D12GraphicsCommandList* commandList, D3D12_RESOURCE_FLAGS flags, DXGI_FORMAT format, const unsigned int size[3], unsigned int numMips, DX12Utils::ResourceType resourceType, const void* initialData, const wchar_t* debugName, D3D12_RESOURCE_STATES desiredState = D3D12_RESOURCE_STATE_COPY_DEST);
        ID3D12Resource* CreateManagedTextureAndClear(ID3D12Device* device, ID3D12GraphicsCommandList* commandList, D3D12_RESOURCE_FLAGS flags, DXGI_FORMAT format, const unsigned int size[3], unsigned int numMips, DX12Utils::ResourceType resourceType, void* clearValue, size_t clearValueSize, const wchar_t* debugName, D3D12_RESOURCE_STATES desiredState = D3D12_RESOURCE_STATE_COPY_DEST);
        ID3D12Resource* CreateManagedTextureFromFile(ID3D12Device* device, ID3D12GraphicsCommandList* commandList, D3D12_RESOURCE_FLAGS flags, DXGI_FORMAT format, DX12Utils::ResourceType resourceType, const char* fileName, bool sourceIsSRGB, unsigned int size[3], const wchar_t* debugName, D3D12_RESOURCE_STATES desiredState = D3D12_RESOURCE_STATE_COPY_DEST);

        // Helpers for the host app
        void UploadTextureData(ID3D12Device* device, ID3D12GraphicsCommandList* commandList, ID3D12Resource* texture, D3D12_RESOURCE_STATES textureState, const void* data, unsigned int unalignedPitch);
        void UploadBufferData(ID3D12Device* device, ID3D12GraphicsCommandList* commandList, ID3D12Resource* buffer, D3D12_RESOURCE_STATES bufferState, const void* data, unsigned int dataSize);

        // The resource will be freed when the context is destroyed
        void AddManagedResource(ID3D12Resource* resource)
        {
            m_internal.m_managedResources.push_back(resource);
        }

        // Returns the allocated index within the respective heap
        int GetRTV(ID3D12Device* device, ID3D12Resource* resource, DXGI_FORMAT format, D3D12_RTV_DIMENSION dimension, int arrayIndex, int mipIndex, const char* debugName);
        int GetDSV(ID3D12Device* device, ID3D12Resource* resource, DXGI_FORMAT format, D3D12_DSV_DIMENSION dimension, int arrayIndex, int mipIndex, const char* debugName);

        bool CreateManagedTLAS(ID3D12Device* device, ID3D12GraphicsCommandList* commandList, ID3D12Resource* vertexBuffer, int vertexBufferCount, bool isAABBs, D3D12_RAYTRACING_GEOMETRY_FLAGS geometryFlags, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS buildFlags, DXGI_FORMAT vertexPositionFormat, unsigned int vertexPositionOffset, unsigned int vertexPositionStride, ID3D12Resource*& blas, unsigned int& blasSize, ID3D12Resource*& tlas, unsigned int& tlasSize, TLogFn logFn)
        {
            ID3D12Resource* scratch = nullptr;
            ID3D12Resource* instanceDescs = nullptr;

            if (!DX12Utils::CreateTLAS(device, commandList, vertexBuffer, vertexBufferCount, isAABBs, geometryFlags, buildFlags, vertexPositionFormat, vertexPositionOffset, vertexPositionStride, blas, blasSize, tlas, tlasSize, scratch, instanceDescs, LogFn))
                return false;

            AddManagedResource(scratch);
            AddManagedResource(instanceDescs);

            AddManagedResource(blas);
            AddManagedResource(tlas);

            return true;
        }

        // Get information about the primary output texture, if specified in the render graph
        ID3D12Resource* GetPrimaryOutputTexture();
        D3D12_RESOURCE_STATES GetPrimaryOutputTextureState();

    private:
        friend void DestroyContext(Context* context);
        ~Context();

        friend void Execute(Context* context, ID3D12Device* device, ID3D12GraphicsCommandList* commandList);
        void EnsureResourcesCreated(ID3D12Device* device, ID3D12GraphicsCommandList* commandList);
        bool EnsureDrawCallPSOsCreated(ID3D12Device* device, bool dirty);

        ProfileEntry m_profileData[16+1]; // One for each action node, and another for the total
    };

    struct ScopedPerfEvent
    {
        ScopedPerfEvent(const char* name, ID3D12GraphicsCommandList* commandList, int index)
            : m_commandList(commandList)
        {
            Context::PerfEventBeginFn(name, commandList, index);
        }

        ~ScopedPerfEvent()
        {
            Context::PerfEventEndFn(m_commandList);
        }

        ID3D12GraphicsCommandList* m_commandList;
    };

    // Create 0 to N contexts at any point
    Context* CreateContext(ID3D12Device* device);

    // Call at the beginning of your frame
    void OnNewFrame(int framesInFlight);

    // Call this 0 to M times a frame on each context to execute the technique
    void Execute(Context* context, ID3D12Device* device, ID3D12GraphicsCommandList* commandList);

    // Destroy a context
    void DestroyContext(Context* context);

    struct Struct_VBStruct
    {
        float3 position = {0.0f,0.0f,0.0f};
        float3 normal = {0.0f,0.0f,0.0f};
        float4 tangent = {0.0f,0.0f,0.0f,0.0f};
        float2 UV = {0.0f,0.0f};
        uint materialID = 0;
    };
};
