import cupy
import torch 
import re
import math

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

                int bottom = CLAMP(y + row*DILATION + intAlpha, 0, SIZE_2(input) - 1);
                int left = CLAMP(x + col*DILATION + intBeta, 0, SIZE_3(input) - 1);
                int top = CLAMP(y + row*DILATION + intAlpha + 1, 0, SIZE_2(input) - 1);
                int right = CLAMP(x + col*DILATION + intBeta + 1, 0, SIZE_3(input) - 1);

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

            int bottom = CLAMP(y + row*DILATION + intAlpha, 0, SIZE_2(input) - 1);
            int left = CLAMP(x + col*DILATION + intBeta, 0, SIZE_3(input) - 1);
            int top = CLAMP(y + row*DILATION + intAlpha + 1, 0, SIZE_2(input) - 1);
            int right = CLAMP(x + col*DILATION + intBeta + 1, 0, SIZE_3(input) - 1);

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

            int bottom = CLAMP(y + row*DILATION + intAlpha, 0, SIZE_2(input) - 1);
            int left = CLAMP(x + col*DILATION + intBeta, 0, SIZE_3(input) - 1);
            int top = CLAMP(y + row*DILATION + intAlpha + 1, 0, SIZE_2(input) - 1);
            int right = CLAMP(x + col*DILATION + intBeta + 1, 0, SIZE_3(input) - 1);

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

            int bottom = CLAMP(y + row*DILATION + intAlpha, 0, SIZE_2(input) - 1);
            int left = CLAMP(x + col*DILATION + intBeta, 0, SIZE_3(input) - 1);
            int top = CLAMP(y + row*DILATION + intAlpha + 1, 0, SIZE_2(input) - 1);
            int right = CLAMP(x + col*DILATION + intBeta + 1, 0, SIZE_3(input) - 1);

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

        strKernel = strKernel.replace(objMatch.group(0), str(intSizes[intArg]))

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

        strKernel = strKernel.replace(objMatch.group(0), strTensor + '[' + str.join('+', strIndex) + ']')

    # purpose: integer clamp a given value to the range [0, upperBound]
    while True: 
        objMatch = re.search('(CLAMP)(\()([^\)]*)(\))', strKernel)

        if objMatch is None: 
            break

        strArgs = objMatch.group(3).split(',')
        
        assert (len(strArgs) == 3)

        strValue, strLowerBound, strUpperBound = strArgs
        strReplacement = f"min(max({strValue}, {strLowerBound}), {strUpperBound})"

        strKernel = strKernel.replace(objMatch.group(0), strReplacement)

    # setting macros
    strKernel = strKernel.replace('F_SIZE', str(intFilterSize))
    strKernel = strKernel.replace('DILATION', str(intDilation))

    return strKernel

@cupy.util.memoize(for_each_device=True)
def cupy_launch(strFunc, strKernel):
    return cupy.cuda.compile_with_cache(strKernel).get_function(strFunc)

class FunctionSyn(torch.autograd.Function): 
    @staticmethod
    def forward(context, input, weight, offset_y, offset_x, dilation): 
        context.save_for_backward(input, weight, offset_y, offset_x)
        context.dilation = dilation

        intSample = input.size(0)
        intInputDepth = input.size(1)
        intInputHeight = input.size(2)
        intInputWidth = input.size(3)
        intFilterSize = int(math.sqrt(weight.size(1)))
        intOutputHeight = weight.size(2)
        intOutputWidth = weight.size(3)

        assert (intInputHeight - ((intFilterSize - 1) * dilation + 1) == intOutputHeight - 1)
        assert (intInputWidth - ((intFilterSize - 1) * dilation + 1) == intOutputWidth - 1)

        assert (input.is_contiguous() == True)
        assert (weight.is_contiguous() == True)
        assert (offset_y.is_contiguous() == True)
        assert (offset_x.is_contiguous() == True)

        output = input.new_zeros(intSample, intInputDepth, intOutputHeight, intOutputWidth)

        if input.is_cuda: 
            # torch.cuda.current_stream() is a function that returns the current CUDA stream for the default CUDA device. 
            # A CUDA stream is a sequence of operations that execute on the GPU in the order they were issued by the host (CPU). 
            # This allows for concurrent execution of operations where possible, such as overlapping data transfers with computations.
            # .cuda_stream is an attribute of the object returned by torch.cuda.current_stream(). 
            # It likely represents the actual underlying CUDA stream object (often a pointer or a low-level handler) that interfaces directly with the CUDA driver API.

            class Stream: 
                ptr = torch.cuda.current_stream().cuda_stream

            n = output.nelement()
            cupy_launch('kernel_Syn_updateOutput', cupy_kernel('kernel_Syn_updateOutput', intFilterSize, dilation, {
                'input': input,
                'weight': weight,
                'offset_y': offset_y,
                'offset_x': offset_x,
                'output': output
            }))(
                grid=tuple([math.ceil(n / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[n, input.data_ptr(), weight.data_ptr(), offset_y.data_ptr(), offset_x.data_ptr(), output.data_ptr()],
                stream=Stream
            )
        else: 
            raise NotImplementedError()

        return output
    
    @staticmethod
    def backward(context, gradOutput):
        input, weight, offset_y, offset_x = context.saved_tensors
        dilation = context.dilation

        intSample = input.size(0)
        intInputDepth = input.size(1)
        intInputHeight = input.size(2)
        intInputWidth = input.size(3)
        intFilterSize = int(math.sqrt(weight.size(1)))
        intOutputHeight = weight.size(2)
        intOutputWidth = weight.size(3)

        assert (intInputHeight - ((intFilterSize - 1) * dilation + 1) == intOutputHeight - 1)
        assert (intInputWidth - ((intFilterSize - 1) * dilation + 1) == intOutputWidth - 1)

        assert (gradOutput.is_contiguous() == True)

        gradInput = input.new_zeros(intSample, intInputDepth, intInputHeight, intInputWidth) if context.needs_input_grad[0] else None
        gradWeight = input.new_zeros(intSample, intFilterSize ** 2, intOutputHeight, intOutputWidth) if context.needs_input_grad[1] else None
        gradOffset_y = input.new_zeros(intSample, intFilterSize ** 2, intOutputHeight, intOutputWidth) if context.needs_input_grad[2] else None
        gradOffset_x = input.new_zeros(intSample, intFilterSize ** 2, intOutputHeight, intOutputWidth) if context.needs_input_grad[2] else None

        if input.is_cuda:
            class Stream:
                ptr = torch.cuda.current_stream().cuda_stream

            # weight grad
            n = gradWeight.nelement()
            cupy_launch('kernel_Syn_updateGradWeight', cupy_kernel('kernel_Syn_updateGradWeight', intFilterSize, dilation, {
                'gradLoss': gradOutput,
                'input': input,
                'offset_y': offset_y,
                'offset_x': offset_x,
                'gradWeight': gradWeight
            }))(
                grid=tuple([math.ceil(n / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[n, gradOutput.data_ptr(), input.data_ptr(), offset_y.data_ptr(), offset_x.data_ptr(), gradWeight.data_ptr()],
                stream=Stream
            )

            # alpha grad
            n = gradOffset_y.nelement()
            cupy_launch('kernel_Syn_updateGradAlpha', cupy_kernel('kernel_Syn_updateGradAlpha', intFilterSize, dilation, {
                'gradLoss': gradOutput,
                'input': input,
                'weight': weight,
                'offset_y': offset_y,
                'offset_x': offset_x,
                'gradOffset_y': gradOffset_y
            }))(
                grid=tuple([math.ceil(n / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[n, gradOutput.data_ptr(), input.data_ptr(), weight.data_ptr(), offset_y.data_ptr(), offset_x.data_ptr(), gradOffset_y.data_ptr()],
                stream=Stream
            )

            # beta grad
            n = gradOffset_x.nelement()
            cupy_launch('kernel_AdaCoF_updateGradBeta', cupy_kernel('kernel_AdaCoF_updateGradBeta', intFilterSize, dilation, {
                'gradLoss': gradOutput,
                'input': input,
                'weight': weight,
                'offset_y': offset_y,
                'offset_x': offset_x,
                'gradOffset_x': gradOffset_x
            }))(
                grid=tuple([math.ceil(n / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[n, gradOutput.data_ptr(), input.data_ptr(), weight.data_ptr(), offset_y.data_ptr(), offset_x.data_ptr(), gradOffset_x.data_ptr()],
                stream=Stream
            )
        else:
            raise NotImplementedError()

        return gradInput, gradWeight, gradOffset_y, gradOffset_x, None