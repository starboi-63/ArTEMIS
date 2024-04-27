import cupy
import torch 
import re
# import math

# torch.cuda.current_stream() is a function that returns the current CUDA stream for the default CUDA device. 
# A CUDA stream is a sequence of operations that execute on the GPU in the order they were issued by the host (CPU). 
# This allows for concurrent execution of operations where possible, such as overlapping data transfers with computations.
# .cuda_stream is an attribute of the object returned by torch.cuda.current_stream(). 
# It likely represents the actual underlying CUDA stream object (often a pointer or a low-level handler) that interfaces directly with the CUDA driver API.

class Stream: 
    ptr = torch.cuda.current_stream().cuda_stream

kernel_Syn_updateOutput = '''
extern "C" __global__ void kernel_Syn_updateOutput(
        const int n,
        const float* input,
        const float* weight, 
        const float* offset_y,
        const float* offset_x,
        float* output
) 
{ 
    for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
        float dblOutput = 0.0;

        const int intSample = ( intIndex / SIZE_3(output) / SIZE_2(output) / SIZE_1(output) ) % SIZE_0(output);
        const int intDepth  = ( intIndex / SIZE_3(output) / SIZE_2(output)                  ) % SIZE_1(output);
        const int y         = ( intIndex / SIZE_3(output)                                   ) % SIZE_2(output);
        const int x         = ( intIndex                                                    ) % SIZE_3(output);
    
        for (int row = 0; row < F_SIZE; row += 1) {
            for (int col = 0; col < F_SIZE; col += 1) {
                float w         = VALUE_4(weight, intSample, row*F_SIZE+col, y, x);
                float alpha     = VALUE_4(offset_x, intSample, row*F_SIZE+col, y, x);
                float beta      = VALUE_4(offset_y, intSample, row*F_SIZE+col, y, x);
                int intAlpha    = (int)alpha;
                int intBeta     = (int)beta;

                int bottom = CLAMP(y + row*DILATION + intAlpha, SIZE_2(input) - 1);
                int left = CLAMP(x + col*DILATION + intBeta, SIZE_3(input) - 1);
                int top = CLAMP(y + row*DILATION + intAlpha + 1, SIZE_2(input) - 1);
                int right = CLAMP(x + col*DILATION + intBeta + 1, SIZE_3(input) - 1);

                float alphaTrunc = alpha - (float)intAlpha;
                float betaTrunc = beta - (float)intBeta;

                dblOutput += w * (
                    VALUE_4(input, intSample, intDepth, bottom, left)*(1 - alphaTrunc)*(1 - betaTrunc) + 
                    VALUE_4(input, intSample, intDepth, top, left)*alphaTrunc*(1 - betaTrunc) + 
                    VALUE_4(input, intSample, intDepth, bottom, right)*(1 - alphaTrunc)*betaTrunc + 
                    VALUE_4(input, intSample, intDepth, top, right)*alphaTrunc*betaTrunc
                );
            }
        }

        output[intIndex] = dblOutput;
    } 
}
'''

kernel_Syn_updateGradWeight = '''
extern "C" __global__ void kernel_AdaCoF_updateGradWeight(
    const int n,
    const float* gradLoss,
    const float* input,
    const float* offset_y,
    const float* offset_x,
    float* gradWeight
) 
{ 
    for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
        float floatOutput = 0.0;

        const int intSample  = ( intIndex / SIZE_3(gradWeight) / SIZE_2(gradWeight) / SIZE_1(gradWeight) ) % SIZE_0(gradWeight);
        const int intDepth   = ( intIndex / SIZE_3(gradWeight) / SIZE_2(gradWeight)                      ) % SIZE_1(gradWeight);
        const int y          = ( intIndex / SIZE_3(gradWeight)                                           ) % SIZE_2(gradWeight);
        const int x          = ( intIndex                                                                ) % SIZE_3(gradWeight);

        int row = intDepth / F_SIZE;
        int col = intDepth % F_SIZE;

        for (int depth = 0; depth < 3; depth++) {
            float delta     = VALUE_4(gradLoss, intSample, depth, y, x);
            float alpha     = VALUE_4(offset_y, intSample, row*F_SIZE+col, y, x);
            float beta      = VALUE_4(offset_x, intSample, row*F_SIZE+col, y, x);
            int intAlpha    = (int)alpha;
            int intBeta     = (int)beta;

            int bottom = CLAMP(y + row*DILATION + intAlpha, SIZE_2(input) - 1);
            int left = CLAMP(x + col*DILATION + intBeta, SIZE_3(input) - 1);
            int top = CLAMP(y + row*DILATION + intAlpha + 1, SIZE_2(input) - 1);
            int right = CLAMP(x + col*DILATION + intBeta + 1, SIZE_3(input) - 1);

            float alphaTrunc = alpha - (float)intAlpha;
            float betaTrunc = beta - (float)intBeta;
            
            floatOutput += delta * (
                VALUE_4(input, intSample, depth, bottom, left)*(1 - alphaTrunc)*(1 - betaTrunc) + 
                VALUE_4(input, intSample, depth, top, left)*alphaTrunc*(1 - betaTrunc) + 
                VALUE_4(input, intSample, depth, bottom, right)*(1 - alphaTrunc)*betaTrunc + 
                VALUE_4(input, intSample, depth, top, right)*alphaTrunc*betaTrunc
            );
        }

        gradWeight[intIndex] = floatOutput;
    } 
}
'''

kernel_Syn_updateGradAlpha = '''
extern "C" __global__ void kernel_AdaCoF_updateGradAlpha(
    const int n,
    const float* gradLoss,
    const float* input,
    const float* weight,
    const float* offset_y,
    const float* offset_x,
    float* gradOffset_y
) 
{ 
    for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
        float floatOutput = 0.0;

        const int intSample  = ( intIndex / SIZE_3(gradOffset_y) / SIZE_2(gradOffset_y) / SIZE_1(gradOffset_y) ) % SIZE_0(gradOffset_y);
        const int intDepth   = ( intIndex / SIZE_3(gradOffset_y) / SIZE_2(gradOffset_y)                        ) % SIZE_1(gradOffset_y);
        const int y          = ( intIndex / SIZE_3(gradOffset_y)                                               ) % SIZE_2(gradOffset_y);
        const int x          = ( intIndex                                                                      ) % SIZE_3(gradOffset_y);

        int row = intDepth / F_SIZE;
        int col = intDepth % F_SIZE;

        for (int depth = 0; depth < 3; depth++) {
            float delta     = VALUE_4(gradLoss, intSample, depth, y, x);
            float w         = VALUE_4(weight, intSample, row*F_SIZE + col, y, x);
            float alpha     = VALUE_4(offset_y, intSample, row*F_SIZE + col, y, x);
            float beta      = VALUE_4(offset_x, intSample, row*F_SIZE + col, y, x);
            int intAlpha    = (int)alpha;
            int intBeta     = (int)beta;

            int bottom = CLAMP(y + row*DILATION + intAlpha, SIZE_2(input) - 1);
            int left = CLAMP(x + col*DILATION + intBeta, SIZE_3(input) - 1);
            int top = CLAMP(y + row*DILATION + intAlpha + 1, SIZE_2(input) - 1);
            int right = CLAMP(x + col*DILATION + intBeta + 1, SIZE_3(input) - 1);

            betaTrunc = beta - (float)intBeta;

            floatOutput += delta * w * (
                - VALUE_4(input, intSample, depth, bottom, left)*(1 - betaTrunc) 
                + VALUE_4(input, intSample, depth, top, left)*(1 - betaTrunc) 
                - VALUE_4(input, intSample, depth, bottom, right)*betaTrunc 
                + VALUE_4(input, intSample, depth, top, right)*betaTrunc
            );
        }

        gradOffset_y[intIndex] = floatOutput;
    } 
}
'''

kernel_Syn_updateGradBeta = '''
extern "C" __global__ void kernel_AdaCoF_updateGradBeta(
    const int n,
    const float* gradLoss,
    const float* input,
    const float* weight,
    const float* offset_y,
    const float* offset_x,
    float* gradOffset_x
) 
{ 
    for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
        float floatOutput = 0.0;

        const int intSample  = ( intIndex / SIZE_3(gradOffset_x) / SIZE_2(gradOffset_x) / SIZE_1(gradOffset_x) ) % SIZE_0(gradOffset_x);
        const int intDepth   = ( intIndex / SIZE_3(gradOffset_x) / SIZE_2(gradOffset_x)                        ) % SIZE_1(gradOffset_x);
        const int y          = ( intIndex / SIZE_3(gradOffset_x)                                               ) % SIZE_2(gradOffset_x);
        const int x          = ( intIndex                                                                      ) % SIZE_3(gradOffset_x);

        int row = intDepth / F_SIZE;
        int col = intDepth % F_SIZE;

        for (int depth = 0; depth < 3; depth++) {
            float delta     = VALUE_4(gradLoss, intSample, depth, y, x);
            float w         = VALUE_4(weight, intSample, row*F_SIZE + col, y, x);
            float alpha     = VALUE_4(offset_y, intSample, row*F_SIZE + col, y, x);
            float beta      = VALUE_4(offset_x, intSample, row*F_SIZE + col, y, x);
            int intAlpha    = (int)alpha;
            int intBeta     = (int)beta;

            int bottom = CLAMP(y + row*DILATION + intAlpha, SIZE_2(input) - 1);
            int left = CLAMP(x + col*DILATION + intBeta, SIZE_3(input) - 1);
            int top = CLAMP(y + row*DILATION + intAlpha + 1, SIZE_2(input) - 1);
            int right = CLAMP(x + col*DILATION + intBeta + 1, SIZE_3(input) - 1);

            alphaTrunc = alpha - (float)intAlpha;

            floatOutput += delta * w * (
                - VALUE_4(input, intSample, depth, bottom, left)*(1 - alphaTrunc) 
                - VALUE_4(input, intSample, depth, top, left)*alphaTrunc 
                + VALUE_4(input, intSample, depth, bottom, right)*(1 - alphaTrunc) 
                + VALUE_4(input, intSample, depth, top, right)*alphaTrunc
            );
        }

        gradOffset_x[intIndex] = floatOutput;
    } 
}
'''

def cupy_kernel(strFunc, intFilterSize, intDilation, objVars): 
    strKernel = globals()[strFunc]

    # purpose: getting size of tensor axis
    while True: 
        objMatch = re.search('(SIZE_)([0-4])(\()([^\)]*)(\))', strKernel)

        if objMatch is None: 
            break

        intArg = int(objMatch.group(2))

        strTensor = objMatch.group(4)
        intSizes = objVars[strTensor].size()

        strKernel = strKernel.replace(objMatch.group(), str(intSizes[intArg]))

    # purpose: getting value at certain index of tensor, ex. tensor[index]
    while True:
        objMatch = re.search('(VALUE_)([0-4])(\()([^\)]+)(\))', strKernel)

        if objMatch is None:
            break

        intArgs = int(objMatch.group(2))
        strArgs = objMatch.group(4).split(',')

        strTensor = strArgs[0]
        intStrides = objVars[strTensor].stride()
        strIndex = ['((' + strArgs[intArg + 1].replace('{', '(').replace('}', ')').strip() + ')*' + str(intStrides[intArg]) + ')' for intArg in range(intArgs)]

        strKernel = strKernel.replace(objMatch.group(), strTensor + '[' + str.join('+', strIndex) + ']')

    # purpose: clamp a given value to the range [0, upperBound] 
    while True: 
        objMatch = re.search('(CLAMP)(\()([^\)]+)(\))', strKernel)

        if objMatch is None: 
            break

        strValue, strUpperBound = objMatch.group(3).split(',')
        value, upperBound = float(strValue), float(strUpperBound)

        clamped = min(max(value, 0), upperBound)

        strKernel = strKernel.replace(objMatch.group(), str(clamped))

    # setting macros
    strKernel = strKernel.replace('F_SIZE', str(intFilterSize))
    strKernel = strKernel.replace('DILATION', str(intDilation))

    return strKernel

@cupy.util.memoize(for_each_device=True)
def cupy_launch(strFunc, strKernel):
    return cupy.cuda.compile_with_cache(strKernel).get_function(strFunc)


