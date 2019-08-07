/**
MIT License

Copyright (c) 2018 NVIDIA CORPORATION. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*
*/

#ifndef _YOLO_H_
#define _YOLO_H_

#include "plugin_factory.h"
#include "trt_utils.h"

#include "NvInfer.h"

#include <stdint.h>
#include <string>
#include <vector>

using std::string;

/**
 * Holds information about an output tensor of the yolo network.
 */
struct TensorInfo
{
    std::string blobName;
    uint stride{0};
    uint gridSize_H{0};
    uint gridSize_W{0};
    uint numClasses{0};
    uint numBBoxes{0};
    uint64_t volume{0};
    std::vector<uint> masks;
    std::vector<float> anchors;
    int bindingIndex{-1};
    float* hostBuffer{nullptr};
};

class Yolo
{
public:
    Yolo(std::string configFilePath, std::string wtsFilePath, std::string precision, std::string enginePath, uint class_num, float probThresh);

    float getNMSThresh() const { return m_NMSThresh; }
    uint getInputH() const { return m_InputH; }
    uint getInputW() const { return m_InputW; }
    uint getClassNum() const {return m_ClassNum; }

    void doInference(const unsigned char* input, const uint batchSize = 1);
    std::vector<BBoxInfo> decodeDetections(const int& imageIdx, const int& imageH,
                                           const int& imageW);
    virtual ~Yolo();

protected:

    std::string m_EnginePath;
    const std::string m_ConfigFilePath;
    const std::string m_WtsFilePath;
    const std::string m_Precision;
    const std::string m_DeviceType;
    const std::string m_InputBlobName;
    const uint m_ClassNum;
    std::vector<TensorInfo> m_OutputTensors;
    std::vector<std::map<std::string, std::string>> m_configBlocks;
    uint m_InputH;
    uint m_InputW;
    uint m_InputC;
    uint64_t m_InputSize;
    const float m_ProbThresh;
    const float m_NMSThresh;

    Logger m_Logger;

    // TRT specific members
    const uint m_BatchSize;
    nvinfer1::INetworkDefinition* m_Network;
    nvinfer1::IBuilder* m_Builder;
    nvinfer1::IHostMemory* m_ModelStream;
    nvinfer1::ICudaEngine* m_Engine;
    nvinfer1::IExecutionContext* m_Context;
    std::vector<void*> m_DeviceBuffers;
    int m_InputBindingIndex;
    cudaStream_t m_CudaStream;
    PluginFactory* m_PluginFactory;
    std::unique_ptr<YoloTinyMaxpoolPaddingFormula> m_TinyMaxpoolPaddingFormula;

    std::vector<BBoxInfo> decodeTensor(const int imageIdx, const int imageH,
                                               const int imageW, const TensorInfo& tensor);

    inline void addBBoxProposal(const float bx, const float by, const float bw, const float bh,
                                const uint stride, const float scalingFactor, const float xOffset,
                                const float yOffset, const int maxIndex, const float maxProb,
                                std::vector<BBoxInfo>& binfo)
    {
        BBoxInfo bbi;
        bbi.box = convertBBoxNetRes(bx, by, bw, bh, stride, m_InputW, m_InputH);
        if ((bbi.box.x1 > bbi.box.x2) || (bbi.box.y1 > bbi.box.y2))
        {
            return;
        }
        convertBBoxImgRes(scalingFactor, xOffset, yOffset, bbi.box);
        bbi.label = maxIndex;
        bbi.prob = maxProb;
        binfo.push_back(bbi);
    };

private:
    void createYOLOEngine(const nvinfer1::DataType dataType = nvinfer1::DataType::kFLOAT);
    std::vector<std::map<std::string, std::string>> parseConfigFile(const std::string cfgFilePath);

    void parseConfigBlocks();
    void allocateBuffers();
    bool verifyYoloEngine();
    void destroyNetworkUtils(std::vector<nvinfer1::Weights>& trtWeights);
    void writePlanFileToDisk();
};

#endif // _YOLO_H_
