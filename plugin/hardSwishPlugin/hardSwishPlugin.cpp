/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "hardSwishPlugin.h"
#include <cstring>
#include <iostream>
#include <vector>

using namespace nvinfer1;

namespace
{
const char* HARDSWISH_PLUGIN_VERSION{"1"};
const char* HARDSWISH_PLUGIN_NAME{"HardSwish"};
} // namespace

PluginFieldCollection HardSwishPluginCreator::mFC{};
std::vector<PluginField> HardSwishPluginCreator::mPluginAttributes;

HardSwishPlugin::HardSwishPlugin() {}

HardSwishPlugin::HardSwishPlugin(nvinfer1::DataType iType, int iC, int iH, int iW)
    : iType(iType)
    , iC(iC)
    , iH(iH)
    , iW(iW)
{}

HardSwishPlugin::HardSwishPlugin(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    iC = read<int>(d);
    iH = read<int>(d);
    iW = read<int>(d);
    ASSERT(d == a + length);
}

int HardSwishPlugin::getNbOutputs() const
{
    return 1;
}

int HardSwishPlugin::initialize()
{
    return STATUS_SUCCESS;
}

void HardSwishPlugin::terminate() {}

Dims HardSwishPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    nvinfer1::Dims dimsOutput;
    dimsOutput.nbDims = inputs->nbDims;
    dimsOutput.d[0] = inputs->d[0];
    dimsOutput.d[1] = inputs->d[1];
    dimsOutput.d[2] = inputs->d[2];
    return dimsOutput;
}

size_t HardSwishPlugin::getWorkspaceSize(int maxBatchSize) const
{
    return 0;
}

size_t HardSwishPlugin::getSerializationSize() const
{
    // iC, iH, iW, oC, oH, oW
    return sizeof(int) * 3;
}

void HardSwishPlugin::serialize(void* buffer) const
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, iC);
    write(d, iH);
    write(d, iW);

    ASSERT(d == a + getSerializationSize());
}

void HardSwishPlugin::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, nvinfer1::PluginFormat format, int maxBatchSize)
{
    ASSERT(nbInputs == 1);
    ASSERT(nbOutputs == 1);

    iC = inputDims->d[0];
    iH = inputDims->d[1];
    iW = inputDims->d[2];

    iType = inputTypes[0];
}

bool HardSwishPlugin::supportsFormat(DataType type, PluginFormat format) const
{
    return ((type == DataType::kFLOAT || type == DataType::kHALF) && format == PluginFormat::kNCHW);
}

const char* HardSwishPlugin::getPluginType() const
{
    return HARDSWISH_PLUGIN_NAME;
}

const char* HardSwishPlugin::getPluginVersion() const
{
    return HARDSWISH_PLUGIN_VERSION;
}

void HardSwishPlugin::destroy()
{
    delete this;
}

IPluginV2Ext* HardSwishPlugin::clone() const
{
    auto* plugin = new HardSwishPlugin(iType, iC, iH, iW);
    return plugin;
}

void HardSwishPlugin::setPluginNamespace(const char* pluginNamespace)
{
    mPluginNamespace = pluginNamespace;
}

const char* HardSwishPlugin::getPluginNamespace() const
{
    return mPluginNamespace;
}

nvinfer1::DataType HardSwishPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    return inputTypes[0];
}

bool HardSwishPlugin::isOutputBroadcastAcrossBatch(
    int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
{
    return false;
}

bool HardSwishPlugin::canBroadcastInputAcrossBatch(int inputIndex) const
{
    return false;
}

// Plugin creator
HardSwishPluginCreator::HardSwishPluginCreator() {}

const char* HardSwishPluginCreator::getPluginName() const
{
    return HARDSWISH_PLUGIN_NAME;
}

const char* HardSwishPluginCreator::getPluginVersion() const
{
    return HARDSWISH_PLUGIN_VERSION;
}

const PluginFieldCollection* HardSwishPluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2Ext* HardSwishPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    HardSwishPlugin* plugin = new HardSwishPlugin();
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

IPluginV2Ext* HardSwishPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    HardSwishPlugin* plugin = new HardSwishPlugin(serialData, serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}
