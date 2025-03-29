///////////////////////////////////////////////////////////////////////////////
//     Filter-Adapted Spatio-Temporal Sampling With General Distributions    //
//        Copyright (c) 2025 Electronic Arts Inc. All rights reserved.       //
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "technique.h"

namespace FastBokeh
{
    inline PyObject* LensRNGToString(PyObject* self, PyObject* args)
    {
        int value;
        if (!PyArg_ParseTuple(args, "i:LensRNGToString", &value))
            return PyErr_Format(PyExc_TypeError, "type error");

        switch((LensRNG)value)
        {
            case LensRNG::UniformCircleWhite_PCG: return Py_BuildValue("s", "UniformCircleWhite_PCG");
            case LensRNG::UniformCircleWhite: return Py_BuildValue("s", "UniformCircleWhite");
            case LensRNG::UniformCircleBlue: return Py_BuildValue("s", "UniformCircleBlue");
            case LensRNG::UniformHexagonWhite: return Py_BuildValue("s", "UniformHexagonWhite");
            case LensRNG::UniformHexagonBlue: return Py_BuildValue("s", "UniformHexagonBlue");
            case LensRNG::UniformHexagonICDF_White: return Py_BuildValue("s", "UniformHexagonICDF_White");
            case LensRNG::UniformHexagonICDF_Blue: return Py_BuildValue("s", "UniformHexagonICDF_Blue");
            case LensRNG::UniformStarWhite: return Py_BuildValue("s", "UniformStarWhite");
            case LensRNG::UniformStarBlue: return Py_BuildValue("s", "UniformStarBlue");
            case LensRNG::UniformStarICDF_White: return Py_BuildValue("s", "UniformStarICDF_White");
            case LensRNG::UniformStarICDF_Blue: return Py_BuildValue("s", "UniformStarICDF_Blue");
            case LensRNG::NonUniformStarWhite: return Py_BuildValue("s", "NonUniformStarWhite");
            case LensRNG::NonUniformStarBlue: return Py_BuildValue("s", "NonUniformStarBlue");
            case LensRNG::NonUniformStar2White: return Py_BuildValue("s", "NonUniformStar2White");
            case LensRNG::NonUniformStar2Blue: return Py_BuildValue("s", "NonUniformStar2Blue");
            case LensRNG::LKCP6White: return Py_BuildValue("s", "LKCP6White");
            case LensRNG::LKCP6Blue: return Py_BuildValue("s", "LKCP6Blue");
            case LensRNG::LKCP204White: return Py_BuildValue("s", "LKCP204White");
            case LensRNG::LKCP204Blue: return Py_BuildValue("s", "LKCP204Blue");
            case LensRNG::LKCP204ICDF_White: return Py_BuildValue("s", "LKCP204ICDF_White");
            case LensRNG::LKCP204ICDF_Blue: return Py_BuildValue("s", "LKCP204ICDF_Blue");
            default: return Py_BuildValue("s", "<invalid LensRNG value>");
        }
    }

    inline PyObject* DOFModeToString(PyObject* self, PyObject* args)
    {
        int value;
        if (!PyArg_ParseTuple(args, "i:DOFModeToString", &value))
            return PyErr_Format(PyExc_TypeError, "type error");

        switch((DOFMode)value)
        {
            case DOFMode::Off: return Py_BuildValue("s", "Off");
            case DOFMode::PathTraced: return Py_BuildValue("s", "PathTraced");
            case DOFMode::PostProcessing: return Py_BuildValue("s", "PostProcessing");
            default: return Py_BuildValue("s", "<invalid DOFMode value>");
        }
    }

    inline PyObject* PixelJitterTypeToString(PyObject* self, PyObject* args)
    {
        int value;
        if (!PyArg_ParseTuple(args, "i:PixelJitterTypeToString", &value))
            return PyErr_Format(PyExc_TypeError, "type error");

        switch((PixelJitterType)value)
        {
            case PixelJitterType::None: return Py_BuildValue("s", "None");
            case PixelJitterType::PerPixel: return Py_BuildValue("s", "PerPixel");
            case PixelJitterType::Global: return Py_BuildValue("s", "Global");
            default: return Py_BuildValue("s", "<invalid PixelJitterType value>");
        }
    }

    inline PyObject* MaterialSetsToString(PyObject* self, PyObject* args)
    {
        int value;
        if (!PyArg_ParseTuple(args, "i:MaterialSetsToString", &value))
            return PyErr_Format(PyExc_TypeError, "type error");

        switch((MaterialSets)value)
        {
            case MaterialSets::None: return Py_BuildValue("s", "None");
            case MaterialSets::Interior: return Py_BuildValue("s", "Interior");
            case MaterialSets::Exterior: return Py_BuildValue("s", "Exterior");
            default: return Py_BuildValue("s", "<invalid MaterialSets value>");
        }
    }

    inline PyObject* NoiseTexExtendsToString(PyObject* self, PyObject* args)
    {
        int value;
        if (!PyArg_ParseTuple(args, "i:NoiseTexExtendsToString", &value))
            return PyErr_Format(PyExc_TypeError, "type error");

        switch((NoiseTexExtends)value)
        {
            case NoiseTexExtends::None: return Py_BuildValue("s", "None");
            case NoiseTexExtends::White: return Py_BuildValue("s", "White");
            case NoiseTexExtends::Shuffle1D: return Py_BuildValue("s", "Shuffle1D");
            case NoiseTexExtends::Shuffle1DHilbert: return Py_BuildValue("s", "Shuffle1DHilbert");
            default: return Py_BuildValue("s", "<invalid NoiseTexExtends value>");
        }
    }

    inline PyObject* GatherDOF_LensRNGToString(PyObject* self, PyObject* args)
    {
        int value;
        if (!PyArg_ParseTuple(args, "i:GatherDOF_LensRNGToString", &value))
            return PyErr_Format(PyExc_TypeError, "type error");

        switch((GatherDOF_LensRNG)value)
        {
            case GatherDOF_LensRNG::UniformCircleWhite_PCG: return Py_BuildValue("s", "UniformCircleWhite_PCG");
            case GatherDOF_LensRNG::UniformCircleWhite: return Py_BuildValue("s", "UniformCircleWhite");
            case GatherDOF_LensRNG::UniformCircleBlue: return Py_BuildValue("s", "UniformCircleBlue");
            case GatherDOF_LensRNG::UniformHexagonWhite: return Py_BuildValue("s", "UniformHexagonWhite");
            case GatherDOF_LensRNG::UniformHexagonBlue: return Py_BuildValue("s", "UniformHexagonBlue");
            case GatherDOF_LensRNG::UniformHexagonICDF_White: return Py_BuildValue("s", "UniformHexagonICDF_White");
            case GatherDOF_LensRNG::UniformHexagonICDF_Blue: return Py_BuildValue("s", "UniformHexagonICDF_Blue");
            case GatherDOF_LensRNG::UniformStarWhite: return Py_BuildValue("s", "UniformStarWhite");
            case GatherDOF_LensRNG::UniformStarBlue: return Py_BuildValue("s", "UniformStarBlue");
            case GatherDOF_LensRNG::UniformStarICDF_White: return Py_BuildValue("s", "UniformStarICDF_White");
            case GatherDOF_LensRNG::UniformStarICDF_Blue: return Py_BuildValue("s", "UniformStarICDF_Blue");
            case GatherDOF_LensRNG::NonUniformStarWhite: return Py_BuildValue("s", "NonUniformStarWhite");
            case GatherDOF_LensRNG::NonUniformStarBlue: return Py_BuildValue("s", "NonUniformStarBlue");
            case GatherDOF_LensRNG::NonUniformStar2White: return Py_BuildValue("s", "NonUniformStar2White");
            case GatherDOF_LensRNG::NonUniformStar2Blue: return Py_BuildValue("s", "NonUniformStar2Blue");
            case GatherDOF_LensRNG::LKCP6White: return Py_BuildValue("s", "LKCP6White");
            case GatherDOF_LensRNG::LKCP6Blue: return Py_BuildValue("s", "LKCP6Blue");
            case GatherDOF_LensRNG::LKCP204White: return Py_BuildValue("s", "LKCP204White");
            case GatherDOF_LensRNG::LKCP204Blue: return Py_BuildValue("s", "LKCP204Blue");
            case GatherDOF_LensRNG::LKCP204ICDF_White: return Py_BuildValue("s", "LKCP204ICDF_White");
            case GatherDOF_LensRNG::LKCP204ICDF_Blue: return Py_BuildValue("s", "LKCP204ICDF_Blue");
            default: return Py_BuildValue("s", "<invalid GatherDOF_LensRNG value>");
        }
    }

    inline PyObject* GatherDOF_NoiseTexExtendsToString(PyObject* self, PyObject* args)
    {
        int value;
        if (!PyArg_ParseTuple(args, "i:GatherDOF_NoiseTexExtendsToString", &value))
            return PyErr_Format(PyExc_TypeError, "type error");

        switch((GatherDOF_NoiseTexExtends)value)
        {
            case GatherDOF_NoiseTexExtends::None: return Py_BuildValue("s", "None");
            case GatherDOF_NoiseTexExtends::White: return Py_BuildValue("s", "White");
            case GatherDOF_NoiseTexExtends::Shuffle1D: return Py_BuildValue("s", "Shuffle1D");
            case GatherDOF_NoiseTexExtends::Shuffle1DHilbert: return Py_BuildValue("s", "Shuffle1DHilbert");
            default: return Py_BuildValue("s", "<invalid GatherDOF_NoiseTexExtends value>");
        }
    }

    inline PyObject* ToneMap_ToneMappingOperationToString(PyObject* self, PyObject* args)
    {
        int value;
        if (!PyArg_ParseTuple(args, "i:ToneMap_ToneMappingOperationToString", &value))
            return PyErr_Format(PyExc_TypeError, "type error");

        switch((ToneMap_ToneMappingOperation)value)
        {
            case ToneMap_ToneMappingOperation::None: return Py_BuildValue("s", "None");
            case ToneMap_ToneMappingOperation::Reinhard_Simple: return Py_BuildValue("s", "Reinhard_Simple");
            case ToneMap_ToneMappingOperation::ACES_Luminance: return Py_BuildValue("s", "ACES_Luminance");
            case ToneMap_ToneMappingOperation::ACES: return Py_BuildValue("s", "ACES");
            default: return Py_BuildValue("s", "<invalid ToneMap_ToneMappingOperation value>");
        }
    }

    inline PyObject* Set_Reset(PyObject* self, PyObject* args)
    {
        int contextIndex;
        bool value;

        if (!PyArg_ParseTuple(args, "ib:Set_Reset", &contextIndex, &value))
            return PyErr_Format(PyExc_TypeError, "type error");

        Context* context = Context::GetContext(contextIndex);
        if (!context)
            return PyErr_Format(PyExc_IndexError, __FUNCTION__, "() : index % i is out of range(count = % i)", contextIndex, Context::GetContextCount());

        context->m_input.variable_Reset = value;

        Py_INCREF(Py_None);
        return Py_None;
    }

    inline PyObject* Set_RenderSize(PyObject* self, PyObject* args)
    {
        int contextIndex;
        uint2 value;

        if (!PyArg_ParseTuple(args, "iII:Set_RenderSize", &contextIndex, &value[0], &value[1]))
            return PyErr_Format(PyExc_TypeError, "type error");

        Context* context = Context::GetContext(contextIndex);
        if (!context)
            return PyErr_Format(PyExc_IndexError, __FUNCTION__, "() : index % i is out of range(count = % i)", contextIndex, Context::GetContextCount());

        context->m_input.variable_RenderSize = value;

        Py_INCREF(Py_None);
        return Py_None;
    }

    inline PyObject* Set_MaterialSet(PyObject* self, PyObject* args)
    {
        int contextIndex;
        int value;

        if (!PyArg_ParseTuple(args, "ii:Set_MaterialSet", &contextIndex, &value))
            return PyErr_Format(PyExc_TypeError, "type error");

        Context* context = Context::GetContext(contextIndex);
        if (!context)
            return PyErr_Format(PyExc_IndexError, __FUNCTION__, "() : index % i is out of range(count = % i)", contextIndex, Context::GetContextCount());

        context->m_input.variable_MaterialSet = (MaterialSets)value;

        Py_INCREF(Py_None);
        return Py_None;
    }

    inline PyObject* Set_Accumulate(PyObject* self, PyObject* args)
    {
        int contextIndex;
        bool value;

        if (!PyArg_ParseTuple(args, "ib:Set_Accumulate", &contextIndex, &value))
            return PyErr_Format(PyExc_TypeError, "type error");

        Context* context = Context::GetContext(contextIndex);
        if (!context)
            return PyErr_Format(PyExc_IndexError, __FUNCTION__, "() : index % i is out of range(count = % i)", contextIndex, Context::GetContextCount());

        context->m_input.variable_Accumulate = value;

        Py_INCREF(Py_None);
        return Py_None;
    }

    inline PyObject* Set_Animate(PyObject* self, PyObject* args)
    {
        int contextIndex;
        bool value;

        if (!PyArg_ParseTuple(args, "ib:Set_Animate", &contextIndex, &value))
            return PyErr_Format(PyExc_TypeError, "type error");

        Context* context = Context::GetContext(contextIndex);
        if (!context)
            return PyErr_Format(PyExc_IndexError, __FUNCTION__, "() : index % i is out of range(count = % i)", contextIndex, Context::GetContextCount());

        context->m_input.variable_Animate = value;

        Py_INCREF(Py_None);
        return Py_None;
    }

    inline PyObject* Set_SamplesPerPixelPerFrame(PyObject* self, PyObject* args)
    {
        int contextIndex;
        uint value;

        if (!PyArg_ParseTuple(args, "iI:Set_SamplesPerPixelPerFrame", &contextIndex, &value))
            return PyErr_Format(PyExc_TypeError, "type error");

        Context* context = Context::GetContext(contextIndex);
        if (!context)
            return PyErr_Format(PyExc_IndexError, __FUNCTION__, "() : index % i is out of range(count = % i)", contextIndex, Context::GetContextCount());

        context->m_input.variable_SamplesPerPixelPerFrame = value;

        Py_INCREF(Py_None);
        return Py_None;
    }

    inline PyObject* Set_JitterPixels(PyObject* self, PyObject* args)
    {
        int contextIndex;
        int value;

        if (!PyArg_ParseTuple(args, "ii:Set_JitterPixels", &contextIndex, &value))
            return PyErr_Format(PyExc_TypeError, "type error");

        Context* context = Context::GetContext(contextIndex);
        if (!context)
            return PyErr_Format(PyExc_IndexError, __FUNCTION__, "() : index % i is out of range(count = % i)", contextIndex, Context::GetContextCount());

        context->m_input.variable_JitterPixels = (PixelJitterType)value;

        Py_INCREF(Py_None);
        return Py_None;
    }

    inline PyObject* Set_NumBounces(PyObject* self, PyObject* args)
    {
        int contextIndex;
        uint value;

        if (!PyArg_ParseTuple(args, "iI:Set_NumBounces", &contextIndex, &value))
            return PyErr_Format(PyExc_TypeError, "type error");

        Context* context = Context::GetContext(contextIndex);
        if (!context)
            return PyErr_Format(PyExc_IndexError, __FUNCTION__, "() : index % i is out of range(count = % i)", contextIndex, Context::GetContextCount());

        context->m_input.variable_NumBounces = value;

        Py_INCREF(Py_None);
        return Py_None;
    }

    inline PyObject* Set_AlbedoMode(PyObject* self, PyObject* args)
    {
        int contextIndex;
        bool value;

        if (!PyArg_ParseTuple(args, "ib:Set_AlbedoMode", &contextIndex, &value))
            return PyErr_Format(PyExc_TypeError, "type error");

        Context* context = Context::GetContext(contextIndex);
        if (!context)
            return PyErr_Format(PyExc_IndexError, __FUNCTION__, "() : index % i is out of range(count = % i)", contextIndex, Context::GetContextCount());

        context->m_input.variable_AlbedoMode = value;

        Py_INCREF(Py_None);
        return Py_None;
    }

    inline PyObject* Set_AlbedoModeAlbedoMultiplier(PyObject* self, PyObject* args)
    {
        int contextIndex;
        float value;

        if (!PyArg_ParseTuple(args, "if:Set_AlbedoModeAlbedoMultiplier", &contextIndex, &value))
            return PyErr_Format(PyExc_TypeError, "type error");

        Context* context = Context::GetContext(contextIndex);
        if (!context)
            return PyErr_Format(PyExc_IndexError, __FUNCTION__, "() : index % i is out of range(count = % i)", contextIndex, Context::GetContextCount());

        context->m_input.variable_AlbedoModeAlbedoMultiplier = value;

        Py_INCREF(Py_None);
        return Py_None;
    }

    inline PyObject* Set_FocalLength(PyObject* self, PyObject* args)
    {
        int contextIndex;
        float value;

        if (!PyArg_ParseTuple(args, "if:Set_FocalLength", &contextIndex, &value))
            return PyErr_Format(PyExc_TypeError, "type error");

        Context* context = Context::GetContext(contextIndex);
        if (!context)
            return PyErr_Format(PyExc_IndexError, __FUNCTION__, "() : index % i is out of range(count = % i)", contextIndex, Context::GetContextCount());

        context->m_input.variable_FocalLength = value;

        Py_INCREF(Py_None);
        return Py_None;
    }

    inline PyObject* Set_LensRNGSource(PyObject* self, PyObject* args)
    {
        int contextIndex;
        int value;

        if (!PyArg_ParseTuple(args, "ii:Set_LensRNGSource", &contextIndex, &value))
            return PyErr_Format(PyExc_TypeError, "type error");

        Context* context = Context::GetContext(contextIndex);
        if (!context)
            return PyErr_Format(PyExc_IndexError, __FUNCTION__, "() : index % i is out of range(count = % i)", contextIndex, Context::GetContextCount());

        context->m_input.variable_LensRNGSource = (LensRNG)value;

        Py_INCREF(Py_None);
        return Py_None;
    }

    inline PyObject* Set_LensRNGExtend(PyObject* self, PyObject* args)
    {
        int contextIndex;
        int value;

        if (!PyArg_ParseTuple(args, "ii:Set_LensRNGExtend", &contextIndex, &value))
            return PyErr_Format(PyExc_TypeError, "type error");

        Context* context = Context::GetContext(contextIndex);
        if (!context)
            return PyErr_Format(PyExc_IndexError, __FUNCTION__, "() : index % i is out of range(count = % i)", contextIndex, Context::GetContextCount());

        context->m_input.variable_LensRNGExtend = (NoiseTexExtends)value;

        Py_INCREF(Py_None);
        return Py_None;
    }

    inline PyObject* Set_JitterNoiseTextures(PyObject* self, PyObject* args)
    {
        int contextIndex;
        bool value;

        if (!PyArg_ParseTuple(args, "ib:Set_JitterNoiseTextures", &contextIndex, &value))
            return PyErr_Format(PyExc_TypeError, "type error");

        Context* context = Context::GetContext(contextIndex);
        if (!context)
            return PyErr_Format(PyExc_IndexError, __FUNCTION__, "() : index % i is out of range(count = % i)", contextIndex, Context::GetContextCount());

        context->m_input.variable_JitterNoiseTextures = value;

        Py_INCREF(Py_None);
        return Py_None;
    }

    inline PyObject* Set_DOF(PyObject* self, PyObject* args)
    {
        int contextIndex;
        int value;

        if (!PyArg_ParseTuple(args, "ii:Set_DOF", &contextIndex, &value))
            return PyErr_Format(PyExc_TypeError, "type error");

        Context* context = Context::GetContext(contextIndex);
        if (!context)
            return PyErr_Format(PyExc_IndexError, __FUNCTION__, "() : index % i is out of range(count = % i)", contextIndex, Context::GetContextCount());

        context->m_input.variable_DOF = (DOFMode)value;

        Py_INCREF(Py_None);
        return Py_None;
    }

    inline PyObject* Set_ApertureRadius(PyObject* self, PyObject* args)
    {
        int contextIndex;
        float value;

        if (!PyArg_ParseTuple(args, "if:Set_ApertureRadius", &contextIndex, &value))
            return PyErr_Format(PyExc_TypeError, "type error");

        Context* context = Context::GetContext(contextIndex);
        if (!context)
            return PyErr_Format(PyExc_IndexError, __FUNCTION__, "() : index % i is out of range(count = % i)", contextIndex, Context::GetContextCount());

        context->m_input.variable_ApertureRadius = value;

        Py_INCREF(Py_None);
        return Py_None;
    }

    inline PyObject* Set_AnamorphicScaling(PyObject* self, PyObject* args)
    {
        int contextIndex;
        float2 value;

        if (!PyArg_ParseTuple(args, "iff:Set_AnamorphicScaling", &contextIndex, &value[0], &value[1]))
            return PyErr_Format(PyExc_TypeError, "type error");

        Context* context = Context::GetContext(contextIndex);
        if (!context)
            return PyErr_Format(PyExc_IndexError, __FUNCTION__, "() : index % i is out of range(count = % i)", contextIndex, Context::GetContextCount());

        context->m_input.variable_AnamorphicScaling = value;

        Py_INCREF(Py_None);
        return Py_None;
    }

    inline PyObject* Set_PetzvalScaling(PyObject* self, PyObject* args)
    {
        int contextIndex;
        float2 value;

        if (!PyArg_ParseTuple(args, "iff:Set_PetzvalScaling", &contextIndex, &value[0], &value[1]))
            return PyErr_Format(PyExc_TypeError, "type error");

        Context* context = Context::GetContext(contextIndex);
        if (!context)
            return PyErr_Format(PyExc_IndexError, __FUNCTION__, "() : index % i is out of range(count = % i)", contextIndex, Context::GetContextCount());

        context->m_input.variable_PetzvalScaling = value;

        Py_INCREF(Py_None);
        return Py_None;
    }

    inline PyObject* Set_OcclusionSettings(PyObject* self, PyObject* args)
    {
        int contextIndex;
        float3 value;

        if (!PyArg_ParseTuple(args, "ifff:Set_OcclusionSettings", &contextIndex, &value[0], &value[1], &value[2]))
            return PyErr_Format(PyExc_TypeError, "type error");

        Context* context = Context::GetContext(contextIndex);
        if (!context)
            return PyErr_Format(PyExc_IndexError, __FUNCTION__, "() : index % i is out of range(count = % i)", contextIndex, Context::GetContextCount());

        context->m_input.variable_OcclusionSettings = value;

        Py_INCREF(Py_None);
        return Py_None;
    }

    inline PyObject* Set_NoImportanceSampling(PyObject* self, PyObject* args)
    {
        int contextIndex;
        bool value;

        if (!PyArg_ParseTuple(args, "ib:Set_NoImportanceSampling", &contextIndex, &value))
            return PyErr_Format(PyExc_TypeError, "type error");

        Context* context = Context::GetContext(contextIndex);
        if (!context)
            return PyErr_Format(PyExc_IndexError, __FUNCTION__, "() : index % i is out of range(count = % i)", contextIndex, Context::GetContextCount());

        context->m_input.variable_NoImportanceSampling = value;

        Py_INCREF(Py_None);
        return Py_None;
    }

    inline PyObject* Set_SkyColor(PyObject* self, PyObject* args)
    {
        int contextIndex;
        float3 value;

        if (!PyArg_ParseTuple(args, "ifff:Set_SkyColor", &contextIndex, &value[0], &value[1], &value[2]))
            return PyErr_Format(PyExc_TypeError, "type error");

        Context* context = Context::GetContext(contextIndex);
        if (!context)
            return PyErr_Format(PyExc_IndexError, __FUNCTION__, "() : index % i is out of range(count = % i)", contextIndex, Context::GetContextCount());

        context->m_input.variable_SkyColor = value;

        Py_INCREF(Py_None);
        return Py_None;
    }

    inline PyObject* Set_SkyBrightness(PyObject* self, PyObject* args)
    {
        int contextIndex;
        float value;

        if (!PyArg_ParseTuple(args, "if:Set_SkyBrightness", &contextIndex, &value))
            return PyErr_Format(PyExc_TypeError, "type error");

        Context* context = Context::GetContext(contextIndex);
        if (!context)
            return PyErr_Format(PyExc_IndexError, __FUNCTION__, "() : index % i is out of range(count = % i)", contextIndex, Context::GetContextCount());

        context->m_input.variable_SkyBrightness = value;

        Py_INCREF(Py_None);
        return Py_None;
    }

    inline PyObject* Set_MaterialEmissiveMultiplier(PyObject* self, PyObject* args)
    {
        int contextIndex;
        float value;

        if (!PyArg_ParseTuple(args, "if:Set_MaterialEmissiveMultiplier", &contextIndex, &value))
            return PyErr_Format(PyExc_TypeError, "type error");

        Context* context = Context::GetContext(contextIndex);
        if (!context)
            return PyErr_Format(PyExc_IndexError, __FUNCTION__, "() : index % i is out of range(count = % i)", contextIndex, Context::GetContextCount());

        context->m_input.variable_MaterialEmissiveMultiplier = value;

        Py_INCREF(Py_None);
        return Py_None;
    }

    inline PyObject* Set_SmallLightBrightness(PyObject* self, PyObject* args)
    {
        int contextIndex;
        float value;

        if (!PyArg_ParseTuple(args, "if:Set_SmallLightBrightness", &contextIndex, &value))
            return PyErr_Format(PyExc_TypeError, "type error");

        Context* context = Context::GetContext(contextIndex);
        if (!context)
            return PyErr_Format(PyExc_IndexError, __FUNCTION__, "() : index % i is out of range(count = % i)", contextIndex, Context::GetContextCount());

        context->m_input.variable_SmallLightBrightness = value;

        Py_INCREF(Py_None);
        return Py_None;
    }

    inline PyObject* Set_SmallLightsColor(PyObject* self, PyObject* args)
    {
        int contextIndex;
        float3 value;

        if (!PyArg_ParseTuple(args, "ifff:Set_SmallLightsColor", &contextIndex, &value[0], &value[1], &value[2]))
            return PyErr_Format(PyExc_TypeError, "type error");

        Context* context = Context::GetContext(contextIndex);
        if (!context)
            return PyErr_Format(PyExc_IndexError, __FUNCTION__, "() : index % i is out of range(count = % i)", contextIndex, Context::GetContextCount());

        context->m_input.variable_SmallLightsColor = value;

        Py_INCREF(Py_None);
        return Py_None;
    }

    inline PyObject* Set_SmallLightsColorful(PyObject* self, PyObject* args)
    {
        int contextIndex;
        bool value;

        if (!PyArg_ParseTuple(args, "ib:Set_SmallLightsColorful", &contextIndex, &value))
            return PyErr_Format(PyExc_TypeError, "type error");

        Context* context = Context::GetContext(contextIndex);
        if (!context)
            return PyErr_Format(PyExc_IndexError, __FUNCTION__, "() : index % i is out of range(count = % i)", contextIndex, Context::GetContextCount());

        context->m_input.variable_SmallLightsColorful = value;

        Py_INCREF(Py_None);
        return Py_None;
    }

    inline PyObject* Set_SmallLightRadius(PyObject* self, PyObject* args)
    {
        int contextIndex;
        float value;

        if (!PyArg_ParseTuple(args, "if:Set_SmallLightRadius", &contextIndex, &value))
            return PyErr_Format(PyExc_TypeError, "type error");

        Context* context = Context::GetContext(contextIndex);
        if (!context)
            return PyErr_Format(PyExc_IndexError, __FUNCTION__, "() : index % i is out of range(count = % i)", contextIndex, Context::GetContextCount());

        context->m_input.variable_SmallLightRadius = value;

        Py_INCREF(Py_None);
        return Py_None;
    }

    inline PyObject* Set_GatherDOF_UseNoiseTextures(PyObject* self, PyObject* args)
    {
        int contextIndex;
        bool value;

        if (!PyArg_ParseTuple(args, "ib:Set_GatherDOF_UseNoiseTextures", &contextIndex, &value))
            return PyErr_Format(PyExc_TypeError, "type error");

        Context* context = Context::GetContext(contextIndex);
        if (!context)
            return PyErr_Format(PyExc_IndexError, __FUNCTION__, "() : index % i is out of range(count = % i)", contextIndex, Context::GetContextCount());

        context->m_input.variable_GatherDOF_UseNoiseTextures = value;

        Py_INCREF(Py_None);
        return Py_None;
    }

    inline PyObject* Set_GatherDOF_AnimateNoiseTextures(PyObject* self, PyObject* args)
    {
        int contextIndex;
        bool value;

        if (!PyArg_ParseTuple(args, "ib:Set_GatherDOF_AnimateNoiseTextures", &contextIndex, &value))
            return PyErr_Format(PyExc_TypeError, "type error");

        Context* context = Context::GetContext(contextIndex);
        if (!context)
            return PyErr_Format(PyExc_IndexError, __FUNCTION__, "() : index % i is out of range(count = % i)", contextIndex, Context::GetContextCount());

        context->m_input.variable_GatherDOF_AnimateNoiseTextures = value;

        Py_INCREF(Py_None);
        return Py_None;
    }

    inline PyObject* Set_GatherDOF_SuppressBokeh(PyObject* self, PyObject* args)
    {
        int contextIndex;
        bool value;

        if (!PyArg_ParseTuple(args, "ib:Set_GatherDOF_SuppressBokeh", &contextIndex, &value))
            return PyErr_Format(PyExc_TypeError, "type error");

        Context* context = Context::GetContext(contextIndex);
        if (!context)
            return PyErr_Format(PyExc_IndexError, __FUNCTION__, "() : index % i is out of range(count = % i)", contextIndex, Context::GetContextCount());

        context->m_input.variable_GatherDOF_SuppressBokeh = value;

        Py_INCREF(Py_None);
        return Py_None;
    }

    inline PyObject* Set_GatherDOF_FocalDistance(PyObject* self, PyObject* args)
    {
        int contextIndex;
        float value;

        if (!PyArg_ParseTuple(args, "if:Set_GatherDOF_FocalDistance", &contextIndex, &value))
            return PyErr_Format(PyExc_TypeError, "type error");

        Context* context = Context::GetContext(contextIndex);
        if (!context)
            return PyErr_Format(PyExc_IndexError, __FUNCTION__, "() : index % i is out of range(count = % i)", contextIndex, Context::GetContextCount());

        context->m_input.variable_GatherDOF_FocalDistance = value;

        Py_INCREF(Py_None);
        return Py_None;
    }

    inline PyObject* Set_GatherDOF_FocalRegion(PyObject* self, PyObject* args)
    {
        int contextIndex;
        float value;

        if (!PyArg_ParseTuple(args, "if:Set_GatherDOF_FocalRegion", &contextIndex, &value))
            return PyErr_Format(PyExc_TypeError, "type error");

        Context* context = Context::GetContext(contextIndex);
        if (!context)
            return PyErr_Format(PyExc_IndexError, __FUNCTION__, "() : index % i is out of range(count = % i)", contextIndex, Context::GetContextCount());

        context->m_input.variable_GatherDOF_FocalRegion = value;

        Py_INCREF(Py_None);
        return Py_None;
    }

    inline PyObject* Set_GatherDOF_FocalLength(PyObject* self, PyObject* args)
    {
        int contextIndex;
        float value;

        if (!PyArg_ParseTuple(args, "if:Set_GatherDOF_FocalLength", &contextIndex, &value))
            return PyErr_Format(PyExc_TypeError, "type error");

        Context* context = Context::GetContext(contextIndex);
        if (!context)
            return PyErr_Format(PyExc_IndexError, __FUNCTION__, "() : index % i is out of range(count = % i)", contextIndex, Context::GetContextCount());

        context->m_input.variable_GatherDOF_FocalLength = value;

        Py_INCREF(Py_None);
        return Py_None;
    }

    inline PyObject* Set_GatherDOF_NearTransitionRegion(PyObject* self, PyObject* args)
    {
        int contextIndex;
        float value;

        if (!PyArg_ParseTuple(args, "if:Set_GatherDOF_NearTransitionRegion", &contextIndex, &value))
            return PyErr_Format(PyExc_TypeError, "type error");

        Context* context = Context::GetContext(contextIndex);
        if (!context)
            return PyErr_Format(PyExc_IndexError, __FUNCTION__, "() : index % i is out of range(count = % i)", contextIndex, Context::GetContextCount());

        context->m_input.variable_GatherDOF_NearTransitionRegion = value;

        Py_INCREF(Py_None);
        return Py_None;
    }

    inline PyObject* Set_GatherDOF_FarTransitionRegion(PyObject* self, PyObject* args)
    {
        int contextIndex;
        float value;

        if (!PyArg_ParseTuple(args, "if:Set_GatherDOF_FarTransitionRegion", &contextIndex, &value))
            return PyErr_Format(PyExc_TypeError, "type error");

        Context* context = Context::GetContext(contextIndex);
        if (!context)
            return PyErr_Format(PyExc_IndexError, __FUNCTION__, "() : index % i is out of range(count = % i)", contextIndex, Context::GetContextCount());

        context->m_input.variable_GatherDOF_FarTransitionRegion = value;

        Py_INCREF(Py_None);
        return Py_None;
    }

    inline PyObject* Set_GatherDOF_Scale(PyObject* self, PyObject* args)
    {
        int contextIndex;
        float value;

        if (!PyArg_ParseTuple(args, "if:Set_GatherDOF_Scale", &contextIndex, &value))
            return PyErr_Format(PyExc_TypeError, "type error");

        Context* context = Context::GetContext(contextIndex);
        if (!context)
            return PyErr_Format(PyExc_IndexError, __FUNCTION__, "() : index % i is out of range(count = % i)", contextIndex, Context::GetContextCount());

        context->m_input.variable_GatherDOF_Scale = value;

        Py_INCREF(Py_None);
        return Py_None;
    }

    inline PyObject* Set_GatherDOF_DoFarField(PyObject* self, PyObject* args)
    {
        int contextIndex;
        bool value;

        if (!PyArg_ParseTuple(args, "ib:Set_GatherDOF_DoFarField", &contextIndex, &value))
            return PyErr_Format(PyExc_TypeError, "type error");

        Context* context = Context::GetContext(contextIndex);
        if (!context)
            return PyErr_Format(PyExc_IndexError, __FUNCTION__, "() : index % i is out of range(count = % i)", contextIndex, Context::GetContextCount());

        context->m_input.variable_GatherDOF_DoFarField = value;

        Py_INCREF(Py_None);
        return Py_None;
    }

    inline PyObject* Set_GatherDOF_DoFarFieldFloodFill(PyObject* self, PyObject* args)
    {
        int contextIndex;
        bool value;

        if (!PyArg_ParseTuple(args, "ib:Set_GatherDOF_DoFarFieldFloodFill", &contextIndex, &value))
            return PyErr_Format(PyExc_TypeError, "type error");

        Context* context = Context::GetContext(contextIndex);
        if (!context)
            return PyErr_Format(PyExc_IndexError, __FUNCTION__, "() : index % i is out of range(count = % i)", contextIndex, Context::GetContextCount());

        context->m_input.variable_GatherDOF_DoFarFieldFloodFill = value;

        Py_INCREF(Py_None);
        return Py_None;
    }

    inline PyObject* Set_GatherDOF_DoNearField(PyObject* self, PyObject* args)
    {
        int contextIndex;
        bool value;

        if (!PyArg_ParseTuple(args, "ib:Set_GatherDOF_DoNearField", &contextIndex, &value))
            return PyErr_Format(PyExc_TypeError, "type error");

        Context* context = Context::GetContext(contextIndex);
        if (!context)
            return PyErr_Format(PyExc_IndexError, __FUNCTION__, "() : index % i is out of range(count = % i)", contextIndex, Context::GetContextCount());

        context->m_input.variable_GatherDOF_DoNearField = value;

        Py_INCREF(Py_None);
        return Py_None;
    }

    inline PyObject* Set_GatherDOF_DoNearFieldFloodFill(PyObject* self, PyObject* args)
    {
        int contextIndex;
        bool value;

        if (!PyArg_ParseTuple(args, "ib:Set_GatherDOF_DoNearFieldFloodFill", &contextIndex, &value))
            return PyErr_Format(PyExc_TypeError, "type error");

        Context* context = Context::GetContext(contextIndex);
        if (!context)
            return PyErr_Format(PyExc_IndexError, __FUNCTION__, "() : index % i is out of range(count = % i)", contextIndex, Context::GetContextCount());

        context->m_input.variable_GatherDOF_DoNearFieldFloodFill = value;

        Py_INCREF(Py_None);
        return Py_None;
    }

    inline PyObject* Set_GatherDOF_KernelSize(PyObject* self, PyObject* args)
    {
        int contextIndex;
        float4 value;

        if (!PyArg_ParseTuple(args, "iffff:Set_GatherDOF_KernelSize", &contextIndex, &value[0], &value[1], &value[2], &value[3]))
            return PyErr_Format(PyExc_TypeError, "type error");

        Context* context = Context::GetContext(contextIndex);
        if (!context)
            return PyErr_Format(PyExc_IndexError, __FUNCTION__, "() : index % i is out of range(count = % i)", contextIndex, Context::GetContextCount());

        context->m_input.variable_GatherDOF_KernelSize = value;

        Py_INCREF(Py_None);
        return Py_None;
    }

    inline PyObject* Set_GatherDOF_BlurTapCount(PyObject* self, PyObject* args)
    {
        int contextIndex;
        uint value;

        if (!PyArg_ParseTuple(args, "iI:Set_GatherDOF_BlurTapCount", &contextIndex, &value))
            return PyErr_Format(PyExc_TypeError, "type error");

        Context* context = Context::GetContext(contextIndex);
        if (!context)
            return PyErr_Format(PyExc_IndexError, __FUNCTION__, "() : index % i is out of range(count = % i)", contextIndex, Context::GetContextCount());

        context->m_input.variable_GatherDOF_BlurTapCount = value;

        Py_INCREF(Py_None);
        return Py_None;
    }

    inline PyObject* Set_GatherDOF_FloodFillTapCount(PyObject* self, PyObject* args)
    {
        int contextIndex;
        uint value;

        if (!PyArg_ParseTuple(args, "iI:Set_GatherDOF_FloodFillTapCount", &contextIndex, &value))
            return PyErr_Format(PyExc_TypeError, "type error");

        Context* context = Context::GetContext(contextIndex);
        if (!context)
            return PyErr_Format(PyExc_IndexError, __FUNCTION__, "() : index % i is out of range(count = % i)", contextIndex, Context::GetContextCount());

        context->m_input.variable_GatherDOF_FloodFillTapCount = value;

        Py_INCREF(Py_None);
        return Py_None;
    }

    inline PyObject* Set_GaussBlur_Sigma(PyObject* self, PyObject* args)
    {
        int contextIndex;
        float value;

        if (!PyArg_ParseTuple(args, "if:Set_GaussBlur_Sigma", &contextIndex, &value))
            return PyErr_Format(PyExc_TypeError, "type error");

        Context* context = Context::GetContext(contextIndex);
        if (!context)
            return PyErr_Format(PyExc_IndexError, __FUNCTION__, "() : index % i is out of range(count = % i)", contextIndex, Context::GetContextCount());

        context->m_input.variable_GaussBlur_Sigma = value;

        Py_INCREF(Py_None);
        return Py_None;
    }

    inline PyObject* Set_GaussBlur_Disable(PyObject* self, PyObject* args)
    {
        int contextIndex;
        bool value;

        if (!PyArg_ParseTuple(args, "ib:Set_GaussBlur_Disable", &contextIndex, &value))
            return PyErr_Format(PyExc_TypeError, "type error");

        Context* context = Context::GetContext(contextIndex);
        if (!context)
            return PyErr_Format(PyExc_IndexError, __FUNCTION__, "() : index % i is out of range(count = % i)", contextIndex, Context::GetContextCount());

        context->m_input.variable_GaussBlur_Disable = value;

        Py_INCREF(Py_None);
        return Py_None;
    }

    inline PyObject* Set_TemporalAccumulation_Alpha(PyObject* self, PyObject* args)
    {
        int contextIndex;
        float value;

        if (!PyArg_ParseTuple(args, "if:Set_TemporalAccumulation_Alpha", &contextIndex, &value))
            return PyErr_Format(PyExc_TypeError, "type error");

        Context* context = Context::GetContext(contextIndex);
        if (!context)
            return PyErr_Format(PyExc_IndexError, __FUNCTION__, "() : index % i is out of range(count = % i)", contextIndex, Context::GetContextCount());

        context->m_input.variable_TemporalAccumulation_Alpha = value;

        Py_INCREF(Py_None);
        return Py_None;
    }

    inline PyObject* Set_TemporalAccumulation_Enabled(PyObject* self, PyObject* args)
    {
        int contextIndex;
        bool value;

        if (!PyArg_ParseTuple(args, "ib:Set_TemporalAccumulation_Enabled", &contextIndex, &value))
            return PyErr_Format(PyExc_TypeError, "type error");

        Context* context = Context::GetContext(contextIndex);
        if (!context)
            return PyErr_Format(PyExc_IndexError, __FUNCTION__, "() : index % i is out of range(count = % i)", contextIndex, Context::GetContextCount());

        context->m_input.variable_TemporalAccumulation_Enabled = value;

        Py_INCREF(Py_None);
        return Py_None;
    }

    inline PyObject* Set_ToneMap_ExposureFStops(PyObject* self, PyObject* args)
    {
        int contextIndex;
        float value;

        if (!PyArg_ParseTuple(args, "if:Set_ToneMap_ExposureFStops", &contextIndex, &value))
            return PyErr_Format(PyExc_TypeError, "type error");

        Context* context = Context::GetContext(contextIndex);
        if (!context)
            return PyErr_Format(PyExc_IndexError, __FUNCTION__, "() : index % i is out of range(count = % i)", contextIndex, Context::GetContextCount());

        context->m_input.variable_ToneMap_ExposureFStops = value;

        Py_INCREF(Py_None);
        return Py_None;
    }

    inline PyObject* Set_ToneMap_ToneMapper(PyObject* self, PyObject* args)
    {
        int contextIndex;
        int value;

        if (!PyArg_ParseTuple(args, "ii:Set_ToneMap_ToneMapper", &contextIndex, &value))
            return PyErr_Format(PyExc_TypeError, "type error");

        Context* context = Context::GetContext(contextIndex);
        if (!context)
            return PyErr_Format(PyExc_IndexError, __FUNCTION__, "() : index % i is out of range(count = % i)", contextIndex, Context::GetContextCount());

        context->m_input.variable_ToneMap_ToneMapper = (ToneMap_ToneMappingOperation)value;

        Py_INCREF(Py_None);
        return Py_None;
    }

    static PyMethodDef pythonModuleMethods[] = {
        {"LensRNGToString", LensRNGToString, METH_VARARGS, ""},
        {"DOFModeToString", DOFModeToString, METH_VARARGS, ""},
        {"PixelJitterTypeToString", PixelJitterTypeToString, METH_VARARGS, ""},
        {"MaterialSetsToString", MaterialSetsToString, METH_VARARGS, ""},
        {"NoiseTexExtendsToString", NoiseTexExtendsToString, METH_VARARGS, ""},
        {"GatherDOF_LensRNGToString", GatherDOF_LensRNGToString, METH_VARARGS, ""},
        {"GatherDOF_NoiseTexExtendsToString", GatherDOF_NoiseTexExtendsToString, METH_VARARGS, ""},
        {"ToneMap_ToneMappingOperationToString", ToneMap_ToneMappingOperationToString, METH_VARARGS, ""},
        {"Set_Reset", Set_Reset, METH_VARARGS, ""},
        {"Set_RenderSize", Set_RenderSize, METH_VARARGS, ""},
        {"Set_MaterialSet", Set_MaterialSet, METH_VARARGS, ""},
        {"Set_Accumulate", Set_Accumulate, METH_VARARGS, ""},
        {"Set_Animate", Set_Animate, METH_VARARGS, ""},
        {"Set_SamplesPerPixelPerFrame", Set_SamplesPerPixelPerFrame, METH_VARARGS, ""},
        {"Set_JitterPixels", Set_JitterPixels, METH_VARARGS, "Provides Antialiasing"},
        {"Set_NumBounces", Set_NumBounces, METH_VARARGS, "How many bounces the rays are allowed"},
        {"Set_AlbedoMode", Set_AlbedoMode, METH_VARARGS, "if true, returns albedo * AlbedoModeAlbedoMultiplier + emissive at primary hit"},
        {"Set_AlbedoModeAlbedoMultiplier", Set_AlbedoModeAlbedoMultiplier, METH_VARARGS, "How much to multiply albedo by in albedo mode, to darken it or lighten it"},
        {"Set_FocalLength", Set_FocalLength, METH_VARARGS, ""},
        {"Set_LensRNGSource", Set_LensRNGSource, METH_VARARGS, ""},
        {"Set_LensRNGExtend", Set_LensRNGExtend, METH_VARARGS, "How to extend the noise textures"},
        {"Set_JitterNoiseTextures", Set_JitterNoiseTextures, METH_VARARGS, "The noise textures are 8 bit unorms. This adds a random value between -0.5/255 and +0.5/255 to fill in the unset bits with white noise."},
        {"Set_DOF", Set_DOF, METH_VARARGS, ""},
        {"Set_ApertureRadius", Set_ApertureRadius, METH_VARARGS, ""},
        {"Set_AnamorphicScaling", Set_AnamorphicScaling, METH_VARARGS, "Defaults to 1.0, 1.0 for no anamorphic effects. Elongates the aperture, does not simulate anamorphic elements."},
        {"Set_PetzvalScaling", Set_PetzvalScaling, METH_VARARGS, "Scales bokeh on each axis depending on screen position. Fakes the effect. Defaults to 1.0, 1.0 for no elongation."},
        {"Set_OcclusionSettings", Set_OcclusionSettings, METH_VARARGS, "Pushes the bounding square of the lens outwards and clips against a unit circle. 1,1,1 means no occlusion. x is how far from the center of the screen to start moving the square. 0 is center, 1 is the corner.  y is how much to scale the lens bounding square by.  z is how far to move the square, as the pixel is farther from where the occlusion begins. Reasonable settings are 0, 0.1, 1.25."},
        {"Set_NoImportanceSampling", Set_NoImportanceSampling, METH_VARARGS, "If true, the FAST noise textures will not be used, and "},
        {"Set_SkyColor", Set_SkyColor, METH_VARARGS, ""},
        {"Set_SkyBrightness", Set_SkyBrightness, METH_VARARGS, ""},
        {"Set_MaterialEmissiveMultiplier", Set_MaterialEmissiveMultiplier, METH_VARARGS, ""},
        {"Set_SmallLightBrightness", Set_SmallLightBrightness, METH_VARARGS, ""},
        {"Set_SmallLightsColor", Set_SmallLightsColor, METH_VARARGS, ""},
        {"Set_SmallLightsColorful", Set_SmallLightsColorful, METH_VARARGS, "If true, makes the small lights colorful, else makes them all the same color"},
        {"Set_SmallLightRadius", Set_SmallLightRadius, METH_VARARGS, ""},
        {"Set_GatherDOF_UseNoiseTextures", Set_GatherDOF_UseNoiseTextures, METH_VARARGS, ""},
        {"Set_GatherDOF_AnimateNoiseTextures", Set_GatherDOF_AnimateNoiseTextures, METH_VARARGS, ""},
        {"Set_GatherDOF_SuppressBokeh", Set_GatherDOF_SuppressBokeh, METH_VARARGS, "If true, blurs out of focus areas, but reduces the Bokeh effect of small bright lights"},
        {"Set_GatherDOF_FocalDistance", Set_GatherDOF_FocalDistance, METH_VARARGS, "Anything closer than this is considered near field"},
        {"Set_GatherDOF_FocalRegion", Set_GatherDOF_FocalRegion, METH_VARARGS, "The size in world units of the middle range which is in focus"},
        {"Set_GatherDOF_FocalLength", Set_GatherDOF_FocalLength, METH_VARARGS, "Focal length in mm (Camera property e.g. 75mm)"},
        {"Set_GatherDOF_NearTransitionRegion", Set_GatherDOF_NearTransitionRegion, METH_VARARGS, "Fade distance in world units"},
        {"Set_GatherDOF_FarTransitionRegion", Set_GatherDOF_FarTransitionRegion, METH_VARARGS, "Fade distance in world units"},
        {"Set_GatherDOF_Scale", Set_GatherDOF_Scale, METH_VARARGS, "Camera property e.g. 0.5f, like aperture"},
        {"Set_GatherDOF_DoFarField", Set_GatherDOF_DoFarField, METH_VARARGS, "Whether or not to do the far field"},
        {"Set_GatherDOF_DoFarFieldFloodFill", Set_GatherDOF_DoFarFieldFloodFill, METH_VARARGS, "Whether to do flood fill on the far field"},
        {"Set_GatherDOF_DoNearField", Set_GatherDOF_DoNearField, METH_VARARGS, "Whether or not to do the near field"},
        {"Set_GatherDOF_DoNearFieldFloodFill", Set_GatherDOF_DoNearFieldFloodFill, METH_VARARGS, "Whether to do flood fill on the near field"},
        {"Set_GatherDOF_KernelSize", Set_GatherDOF_KernelSize, METH_VARARGS, "x = size of the bokeh blur radius in texel space. y = rotation in radians to apply to the bokeh shape. z = Number of edge of the polygon (number of blades). 0: circle. 4: square, 6: hexagon..."},
        {"Set_GatherDOF_BlurTapCount", Set_GatherDOF_BlurTapCount, METH_VARARGS, "8 for high quality, 6 for low quality. Used in a double for loop, so it's this number squared."},
        {"Set_GatherDOF_FloodFillTapCount", Set_GatherDOF_FloodFillTapCount, METH_VARARGS, "4 for high quality, 3 for low quality. Used in a double for loop, so it's this number squared."},
        {"Set_GaussBlur_Sigma", Set_GaussBlur_Sigma, METH_VARARGS, "Strength of blur. Standard deviation of gaussian distribution."},
        {"Set_GaussBlur_Disable", Set_GaussBlur_Disable, METH_VARARGS, ""},
        {"Set_TemporalAccumulation_Alpha", Set_TemporalAccumulation_Alpha, METH_VARARGS, "For exponential moving average. From 0 to 1. TAA commonly uses 0.1."},
        {"Set_TemporalAccumulation_Enabled", Set_TemporalAccumulation_Enabled, METH_VARARGS, ""},
        {"Set_ToneMap_ExposureFStops", Set_ToneMap_ExposureFStops, METH_VARARGS, ""},
        {"Set_ToneMap_ToneMapper", Set_ToneMap_ToneMapper, METH_VARARGS, ""},
        {nullptr, nullptr, 0, nullptr}
    };

    static PyModuleDef pythonModule = {
        PyModuleDef_HEAD_INIT, "FastBokeh", NULL, -1, pythonModuleMethods,
        NULL, NULL, NULL, NULL
    };

    PyObject* CreateModule()
    {
        PyObject* module = PyModule_Create(&pythonModule);
        PyModule_AddIntConstant(module, "LensRNG_UniformCircleWhite_PCG", 0);
        PyModule_AddIntConstant(module, "LensRNG_UniformCircleWhite", 1);
        PyModule_AddIntConstant(module, "LensRNG_UniformCircleBlue", 2);
        PyModule_AddIntConstant(module, "LensRNG_UniformHexagonWhite", 3);
        PyModule_AddIntConstant(module, "LensRNG_UniformHexagonBlue", 4);
        PyModule_AddIntConstant(module, "LensRNG_UniformHexagonICDF_White", 5);
        PyModule_AddIntConstant(module, "LensRNG_UniformHexagonICDF_Blue", 6);
        PyModule_AddIntConstant(module, "LensRNG_UniformStarWhite", 7);
        PyModule_AddIntConstant(module, "LensRNG_UniformStarBlue", 8);
        PyModule_AddIntConstant(module, "LensRNG_UniformStarICDF_White", 9);
        PyModule_AddIntConstant(module, "LensRNG_UniformStarICDF_Blue", 10);
        PyModule_AddIntConstant(module, "LensRNG_NonUniformStarWhite", 11);
        PyModule_AddIntConstant(module, "LensRNG_NonUniformStarBlue", 12);
        PyModule_AddIntConstant(module, "LensRNG_NonUniformStar2White", 13);
        PyModule_AddIntConstant(module, "LensRNG_NonUniformStar2Blue", 14);
        PyModule_AddIntConstant(module, "LensRNG_LKCP6White", 15);
        PyModule_AddIntConstant(module, "LensRNG_LKCP6Blue", 16);
        PyModule_AddIntConstant(module, "LensRNG_LKCP204White", 17);
        PyModule_AddIntConstant(module, "LensRNG_LKCP204Blue", 18);
        PyModule_AddIntConstant(module, "LensRNG_LKCP204ICDF_White", 19);
        PyModule_AddIntConstant(module, "LensRNG_LKCP204ICDF_Blue", 20);
        PyModule_AddIntConstant(module, "DOFMode_Off", 0);
        PyModule_AddIntConstant(module, "DOFMode_PathTraced", 1);
        PyModule_AddIntConstant(module, "DOFMode_PostProcessing", 2);
        PyModule_AddIntConstant(module, "PixelJitterType_None", 0);
        PyModule_AddIntConstant(module, "PixelJitterType_PerPixel", 1);
        PyModule_AddIntConstant(module, "PixelJitterType_Global", 2);
        PyModule_AddIntConstant(module, "MaterialSets_None", 0);
        PyModule_AddIntConstant(module, "MaterialSets_Interior", 1);
        PyModule_AddIntConstant(module, "MaterialSets_Exterior", 2);
        PyModule_AddIntConstant(module, "NoiseTexExtends_None", 0);
        PyModule_AddIntConstant(module, "NoiseTexExtends_White", 1);
        PyModule_AddIntConstant(module, "NoiseTexExtends_Shuffle1D", 2);
        PyModule_AddIntConstant(module, "NoiseTexExtends_Shuffle1DHilbert", 3);
        PyModule_AddIntConstant(module, "GatherDOF_LensRNG_UniformCircleWhite_PCG", 0);
        PyModule_AddIntConstant(module, "GatherDOF_LensRNG_UniformCircleWhite", 1);
        PyModule_AddIntConstant(module, "GatherDOF_LensRNG_UniformCircleBlue", 2);
        PyModule_AddIntConstant(module, "GatherDOF_LensRNG_UniformHexagonWhite", 3);
        PyModule_AddIntConstant(module, "GatherDOF_LensRNG_UniformHexagonBlue", 4);
        PyModule_AddIntConstant(module, "GatherDOF_LensRNG_UniformHexagonICDF_White", 5);
        PyModule_AddIntConstant(module, "GatherDOF_LensRNG_UniformHexagonICDF_Blue", 6);
        PyModule_AddIntConstant(module, "GatherDOF_LensRNG_UniformStarWhite", 7);
        PyModule_AddIntConstant(module, "GatherDOF_LensRNG_UniformStarBlue", 8);
        PyModule_AddIntConstant(module, "GatherDOF_LensRNG_UniformStarICDF_White", 9);
        PyModule_AddIntConstant(module, "GatherDOF_LensRNG_UniformStarICDF_Blue", 10);
        PyModule_AddIntConstant(module, "GatherDOF_LensRNG_NonUniformStarWhite", 11);
        PyModule_AddIntConstant(module, "GatherDOF_LensRNG_NonUniformStarBlue", 12);
        PyModule_AddIntConstant(module, "GatherDOF_LensRNG_NonUniformStar2White", 13);
        PyModule_AddIntConstant(module, "GatherDOF_LensRNG_NonUniformStar2Blue", 14);
        PyModule_AddIntConstant(module, "GatherDOF_LensRNG_LKCP6White", 15);
        PyModule_AddIntConstant(module, "GatherDOF_LensRNG_LKCP6Blue", 16);
        PyModule_AddIntConstant(module, "GatherDOF_LensRNG_LKCP204White", 17);
        PyModule_AddIntConstant(module, "GatherDOF_LensRNG_LKCP204Blue", 18);
        PyModule_AddIntConstant(module, "GatherDOF_LensRNG_LKCP204ICDF_White", 19);
        PyModule_AddIntConstant(module, "GatherDOF_LensRNG_LKCP204ICDF_Blue", 20);
        PyModule_AddIntConstant(module, "GatherDOF_NoiseTexExtends_None", 0);
        PyModule_AddIntConstant(module, "GatherDOF_NoiseTexExtends_White", 1);
        PyModule_AddIntConstant(module, "GatherDOF_NoiseTexExtends_Shuffle1D", 2);
        PyModule_AddIntConstant(module, "GatherDOF_NoiseTexExtends_Shuffle1DHilbert", 3);
        PyModule_AddIntConstant(module, "ToneMap_ToneMappingOperation_None", 0);
        PyModule_AddIntConstant(module, "ToneMap_ToneMappingOperation_Reinhard_Simple", 1);
        PyModule_AddIntConstant(module, "ToneMap_ToneMappingOperation_ACES_Luminance", 2);
        PyModule_AddIntConstant(module, "ToneMap_ToneMappingOperation_ACES", 3);
        return module;
    }
};
