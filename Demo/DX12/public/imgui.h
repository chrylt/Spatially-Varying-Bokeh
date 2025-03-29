///////////////////////////////////////////////////////////////////////////////
//     Filter-Adapted Spatio-Temporal Sampling With General Distributions    //
//        Copyright (c) 2025 Electronic Arts Inc. All rights reserved.       //
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "technique.h"

namespace FastBokeh
{
    inline void ShowToolTip(const char* tooltip)
    {
        if (!tooltip || !tooltip[0])
            return;

        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 255, 0, 255));
        ImGui::Text("[?]");
        ImGui::PopStyleColor();
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
            ImGui::SetTooltip("%s", tooltip);
    }

    void MakeUI(Context* context, ID3D12CommandQueue* commandQueue)
    {
        ImGui::PushID("gigi_FastBokeh");

        context->m_input.variable_Reset = ImGui::Button("Reset");
        {
            float width = ImGui::GetContentRegionAvail().x / 4.0f;
            ImGui::PushID("RenderSize");
            ImGui::PushItemWidth(width);
            int localVarX = (int)context->m_input.variable_RenderSize[0];
            if(ImGui::InputInt("##X", &localVarX, 0))
                context->m_input.variable_RenderSize[0] = (unsigned int)localVarX;
            ImGui::SameLine();
            int localVarY = (int)context->m_input.variable_RenderSize[1];
            if(ImGui::InputInt("##Y", &localVarY, 0))
                context->m_input.variable_RenderSize[1] = (unsigned int)localVarY;
            ImGui::SameLine();
            ImGui::Text("RenderSize");
            ImGui::PopItemWidth();
            ImGui::PopID();
        }
        {
            static const char* labels[] = {
                "None",
                "Interior",
                "Exterior",
            };
            ImGui::Combo("MaterialSet", (int*)&context->m_input.variable_MaterialSet, labels, 3);
        }
        ImGui::Checkbox("Accumulate", &context->m_input.variable_Accumulate);
        ImGui::Checkbox("Animate", &context->m_input.variable_Animate);
        {
            int localVar = (int)context->m_input.variable_SamplesPerPixelPerFrame;
            if(ImGui::InputInt("SamplesPerPixelPerFrame", &localVar, 0))
                context->m_input.variable_SamplesPerPixelPerFrame = (unsigned int)localVar;
        }
        {
            static const char* labels[] = {
                "None",
                "PerPixel",
                "Global",
            };
            ImGui::Combo("JitterPixels", (int*)&context->m_input.variable_JitterPixels, labels, 3);
            ShowToolTip("Provides Antialiasing");
        }
        {
            int localVar = (int)context->m_input.variable_NumBounces;
            if(ImGui::InputInt("NumBounces", &localVar, 0))
                context->m_input.variable_NumBounces = (unsigned int)localVar;
            ShowToolTip("How many bounces the rays are allowed");
        }
        ImGui::Checkbox("AlbedoMode", &context->m_input.variable_AlbedoMode);
        ShowToolTip("if true, returns albedo * AlbedoModeAlbedoMultiplier + emissive at primary hit");
        ImGui::InputFloat("AlbedoModeAlbedoMultiplier", &context->m_input.variable_AlbedoModeAlbedoMultiplier);
        ShowToolTip("How much to multiply albedo by in albedo mode, to darken it or lighten it");
        ImGui::InputFloat("FocalLength", &context->m_input.variable_FocalLength);
        {
            static const char* labels[] = {
                "UniformCircleWhite_PCG",
                "UniformCircleWhite",
                "UniformCircleBlue",
                "UniformHexagonWhite",
                "UniformHexagonBlue",
                "UniformHexagonICDF_White",
                "UniformHexagonICDF_Blue",
                "UniformStarWhite",
                "UniformStarBlue",
                "UniformStarICDF_White",
                "UniformStarICDF_Blue",
                "NonUniformStarWhite",
                "NonUniformStarBlue",
                "NonUniformStar2White",
                "NonUniformStar2Blue",
                "LKCP6White",
                "LKCP6Blue",
                "LKCP204White",
                "LKCP204Blue",
                "LKCP204ICDF_White",
                "LKCP204ICDF_Blue",
            };
            ImGui::Combo("LensRNGSource", (int*)&context->m_input.variable_LensRNGSource, labels, 21);
        }
        {
            static const char* labels[] = {
                "None",
                "White",
                "Shuffle1D",
                "Shuffle1DHilbert",
            };
            ImGui::Combo("LensRNGExtend", (int*)&context->m_input.variable_LensRNGExtend, labels, 4);
            ShowToolTip("How to extend the noise textures");
        }
        ImGui::Checkbox("JitterNoiseTextures", &context->m_input.variable_JitterNoiseTextures);
        ShowToolTip("The noise textures are 8 bit unorms. This adds a random value between -0.5/255 and +0.5/255 to fill in the unset bits with white noise.");
        {
            static const char* labels[] = {
                "Off",
                "PathTraced",
                "PostProcessing",
            };
            ImGui::Combo("DOF", (int*)&context->m_input.variable_DOF, labels, 3);
        }
        ImGui::InputFloat("ApertureRadius", &context->m_input.variable_ApertureRadius);
        {
            float width = ImGui::GetContentRegionAvail().x / 4.0f;
            ImGui::PushID("AnamorphicScaling");
            ImGui::PushItemWidth(width);
            ImGui::InputFloat("##X", &context->m_input.variable_AnamorphicScaling[0]);
            ImGui::SameLine();
            ImGui::InputFloat("##Y", &context->m_input.variable_AnamorphicScaling[1]);
            ImGui::SameLine();
            ImGui::Text("AnamorphicScaling");
            ImGui::PopItemWidth();
            ImGui::PopID();
            ShowToolTip("Defaults to 1.0, 1.0 for no anamorphic effects. Elongates the aperture, does not simulate anamorphic elements.");
        }
        {
            float width = ImGui::GetContentRegionAvail().x / 4.0f;
            ImGui::PushID("PetzvalScaling");
            ImGui::PushItemWidth(width);
            ImGui::InputFloat("##X", &context->m_input.variable_PetzvalScaling[0]);
            ImGui::SameLine();
            ImGui::InputFloat("##Y", &context->m_input.variable_PetzvalScaling[1]);
            ImGui::SameLine();
            ImGui::Text("PetzvalScaling");
            ImGui::PopItemWidth();
            ImGui::PopID();
            ShowToolTip("Scales bokeh on each axis depending on screen position. Fakes the effect. Defaults to 1.0, 1.0 for no elongation.");
        }
        {
            float width = ImGui::GetContentRegionAvail().x / 5.0f;
            ImGui::PushID("OcclusionSettings");
            ImGui::PushItemWidth(width);
            ImGui::InputFloat("##X", &context->m_input.variable_OcclusionSettings[0]);
            ImGui::SameLine();
            ImGui::InputFloat("##Y", &context->m_input.variable_OcclusionSettings[1]);
            ImGui::SameLine();
            ImGui::InputFloat("##Z", &context->m_input.variable_OcclusionSettings[2]);
            ImGui::SameLine();
            ImGui::Text("OcclusionSettings");
            ImGui::PopItemWidth();
            ImGui::PopID();
            ShowToolTip("Pushes the bounding square of the lens outwards and clips against a unit circle. 1,1,1 means no occlusion. x is how far from the center of the screen to start moving the square. 0 is center, 1 is the corner.  y is how much to scale the lens bounding square by.  z is how far to move the square, as the pixel is farther from where the occlusion begins. Reasonable settings are 0, 0.1, 1.25.");
        }
        ImGui::Checkbox("NoImportanceSampling", &context->m_input.variable_NoImportanceSampling);
        ShowToolTip("If true, the FAST noise textures will not be used, and ");
        ImGui::ColorEdit3("SkyColor", &context->m_input.variable_SkyColor[0]);
        ImGui::InputFloat("SkyBrightness", &context->m_input.variable_SkyBrightness);
        ImGui::InputFloat("MaterialEmissiveMultiplier", &context->m_input.variable_MaterialEmissiveMultiplier);
        ImGui::InputFloat("SmallLightBrightness", &context->m_input.variable_SmallLightBrightness);
        ImGui::ColorEdit3("SmallLightsColor", &context->m_input.variable_SmallLightsColor[0]);
        ImGui::Checkbox("SmallLightsColorful", &context->m_input.variable_SmallLightsColorful);
        ShowToolTip("If true, makes the small lights colorful, else makes them all the same color");
        ImGui::InputFloat("SmallLightRadius", &context->m_input.variable_SmallLightRadius);
        ImGui::Checkbox("GatherDOF_UseNoiseTextures", &context->m_input.variable_GatherDOF_UseNoiseTextures);
        ImGui::Checkbox("GatherDOF_AnimateNoiseTextures", &context->m_input.variable_GatherDOF_AnimateNoiseTextures);
        ImGui::Checkbox("GatherDOF_SuppressBokeh", &context->m_input.variable_GatherDOF_SuppressBokeh);
        ShowToolTip("If true, blurs out of focus areas, but reduces the Bokeh effect of small bright lights");
        ImGui::InputFloat("GatherDOF_FocalDistance", &context->m_input.variable_GatherDOF_FocalDistance);
        ShowToolTip("Anything closer than this is considered near field");
        ImGui::InputFloat("GatherDOF_FocalRegion", &context->m_input.variable_GatherDOF_FocalRegion);
        ShowToolTip("The size in world units of the middle range which is in focus");
        ImGui::InputFloat("GatherDOF_FocalLength", &context->m_input.variable_GatherDOF_FocalLength);
        ShowToolTip("Focal length in mm (Camera property e.g. 75mm)");
        ImGui::InputFloat("GatherDOF_NearTransitionRegion", &context->m_input.variable_GatherDOF_NearTransitionRegion);
        ShowToolTip("Fade distance in world units");
        ImGui::InputFloat("GatherDOF_FarTransitionRegion", &context->m_input.variable_GatherDOF_FarTransitionRegion);
        ShowToolTip("Fade distance in world units");
        ImGui::InputFloat("GatherDOF_Scale", &context->m_input.variable_GatherDOF_Scale);
        ShowToolTip("Camera property e.g. 0.5f, like aperture");
        ImGui::Checkbox("GatherDOF_DoFarField", &context->m_input.variable_GatherDOF_DoFarField);
        ShowToolTip("Whether or not to do the far field");
        ImGui::Checkbox("GatherDOF_DoFarFieldFloodFill", &context->m_input.variable_GatherDOF_DoFarFieldFloodFill);
        ShowToolTip("Whether to do flood fill on the far field");
        ImGui::Checkbox("GatherDOF_DoNearField", &context->m_input.variable_GatherDOF_DoNearField);
        ShowToolTip("Whether or not to do the near field");
        ImGui::Checkbox("GatherDOF_DoNearFieldFloodFill", &context->m_input.variable_GatherDOF_DoNearFieldFloodFill);
        ShowToolTip("Whether to do flood fill on the near field");
        {
            float width = ImGui::GetContentRegionAvail().x / 6.0f;
            ImGui::PushID("GatherDOF_KernelSize");
            ImGui::PushItemWidth(width);
            ImGui::InputFloat("##X", &context->m_input.variable_GatherDOF_KernelSize[0]);
            ImGui::SameLine();
            ImGui::InputFloat("##Y", &context->m_input.variable_GatherDOF_KernelSize[1]);
            ImGui::SameLine();
            ImGui::InputFloat("##Z", &context->m_input.variable_GatherDOF_KernelSize[2]);
            ImGui::SameLine();
            ImGui::InputFloat("##W", &context->m_input.variable_GatherDOF_KernelSize[3]);
            ImGui::SameLine();
            ImGui::Text("GatherDOF_KernelSize");
            ImGui::PopItemWidth();
            ImGui::PopID();
            ShowToolTip("x = size of the bokeh blur radius in texel space. y = rotation in radians to apply to the bokeh shape. z = Number of edge of the polygon (number of blades). 0: circle. 4: square, 6: hexagon...");
        }
        {
            int localVar = (int)context->m_input.variable_GatherDOF_BlurTapCount;
            if(ImGui::InputInt("GatherDOF_BlurTapCount", &localVar, 0))
                context->m_input.variable_GatherDOF_BlurTapCount = (unsigned int)localVar;
            ShowToolTip("8 for high quality, 6 for low quality. Used in a double for loop, so it's this number squared.");
        }
        {
            int localVar = (int)context->m_input.variable_GatherDOF_FloodFillTapCount;
            if(ImGui::InputInt("GatherDOF_FloodFillTapCount", &localVar, 0))
                context->m_input.variable_GatherDOF_FloodFillTapCount = (unsigned int)localVar;
            ShowToolTip("4 for high quality, 3 for low quality. Used in a double for loop, so it's this number squared.");
        }
        ImGui::InputFloat("GaussBlur_Sigma", &context->m_input.variable_GaussBlur_Sigma);
        ShowToolTip("Strength of blur. Standard deviation of gaussian distribution.");
        ImGui::Checkbox("GaussBlur_Disable", &context->m_input.variable_GaussBlur_Disable);
        ImGui::InputFloat("TemporalAccumulation_Alpha", &context->m_input.variable_TemporalAccumulation_Alpha);
        ShowToolTip("For exponential moving average. From 0 to 1. TAA commonly uses 0.1.");
        ImGui::Checkbox("TemporalAccumulation_Enabled", &context->m_input.variable_TemporalAccumulation_Enabled);
        ImGui::InputFloat("ToneMap_ExposureFStops", &context->m_input.variable_ToneMap_ExposureFStops);
        {
            static const char* labels[] = {
                "None",
                "Reinhard_Simple",
                "ACES_Luminance",
                "ACES",
            };
            ImGui::Combo("ToneMap_ToneMapper", (int*)&context->m_input.variable_ToneMap_ToneMapper, labels, 4);
        }

        ImGui::Checkbox("Profile", &context->m_profile);
        if (context->m_profile)
        {
            int numEntries = 0;
            const ProfileEntry* entries = context->ReadbackProfileData(commandQueue, numEntries);
            if (ImGui::BeginTable("profiling", 3, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg))
            {
                ImGui::TableSetupColumn("Label");
                ImGui::TableSetupColumn("CPU ms");
                ImGui::TableSetupColumn("GPU ms");
                ImGui::TableHeadersRow();
                float totalCpu = 0.0f;
                float totalGpu = 0.0f;
                for (int entryIndex = 0; entryIndex < numEntries; ++entryIndex)
                {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::TextUnformatted(entries[entryIndex].m_label);
                    ImGui::TableNextColumn();
                    ImGui::Text("%0.3f", entries[entryIndex].m_cpu * 1000.0f);
                    ImGui::TableNextColumn();
                    ImGui::Text("%0.3f", entries[entryIndex].m_gpu * 1000.0f);
                    totalCpu += entries[entryIndex].m_cpu;
                    totalGpu += entries[entryIndex].m_gpu;
                }
                ImGui::EndTable();
            }
        }

        ImGui::PopID();
    }
};
