import cupy
import torch
import re
import math

kernel_AdaCoF_updateOutput = '''
extern "C" __global__ void kernel_AdaCoF_updateOutput(
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

kernel_AdaCoF_updateGradWeight = '''
    extern "C" __global__ void kernel_AdaCoF_updateGradWeight(
        const int n,
        const float* gradLoss,
        const float* input,
        const float* offset_i,
        const float* offset_j,
        float* gradWeight
    ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
        float floatOutput = 0.0;

        const int intSample  = ( intIndex / SIZE_3(gradWeight) / SIZE_2(gradWeight) / SIZE_1(gradWeight) ) % SIZE_0(gradWeight);
        const int intDepth   = ( intIndex / SIZE_3(gradWeight) / SIZE_2(gradWeight)                      ) % SIZE_1(gradWeight);
        const int i          = ( intIndex / SIZE_3(gradWeight)                                           ) % SIZE_2(gradWeight);
        const int j          = ( intIndex                                                                ) % SIZE_3(gradWeight);

        int k = intDepth / F_SIZE;
        int l = intDepth % F_SIZE;

        for (int c = 0; c < 3; c++) 
        {
        float delta     = VALUE_4(gradLoss, intSample, c, i, j);
        float alpha     = VALUE_4(offset_i, intSample, k*F_SIZE+l, i, j);
        float beta      = VALUE_4(offset_j, intSample, k*F_SIZE+l, i, j);
        int A           = (int) alpha;
        int B           = (int) beta;

        int i_k_A = i+k*DILATION+A;
        if(i_k_A < 0)
            i_k_A = 0;
        if(i_k_A > SIZE_2(input) - 1)
            i_k_A = SIZE_2(input) - 1;

        int j_l_B = j+l*DILATION+B;
        if(j_l_B < 0)
            j_l_B = 0;
        if(j_l_B > SIZE_3(input) - 1)
            j_l_B = SIZE_3(input) - 1;

        int i_k_A_1 = i+k*DILATION+A+1;
        if(i_k_A_1 < 0)
            i_k_A_1 = 0;
        if(i_k_A_1 > SIZE_2(input) - 1)
            i_k_A_1 = SIZE_2(input) - 1;

        int j_l_B_1 = j+l*DILATION+B+1;
        if(j_l_B_1 < 0)
            j_l_B_1 = 0;
        if(j_l_B_1 > SIZE_3(input) - 1)
            j_l_B_1 = SIZE_3(input) - 1;
        
        floatOutput += delta * (
            VALUE_4(input, intSample, c, i_k_A, j_l_B)*(1-(alpha-(float)A))*(1-(beta-(float)B)) + 
            VALUE_4(input, intSample, c, i_k_A_1, j_l_B)*(alpha-(float)A)*(1-(beta-(float)B)) + 
            VALUE_4(input, intSample, c, i_k_A, j_l_B_1)*(1-(alpha-(float)A))*(beta-(float)B) + 
            VALUE_4(input, intSample, c, i_k_A_1, j_l_B_1)*(alpha-(float)A)*(beta-(float)B)
            );
        }

        gradWeight[intIndex] = floatOutput;
    } }
'''

kernel_AdaCoF_updateGradAlpha = '''
    extern "C" __global__ void kernel_AdaCoF_updateGradAlpha(
        const int n,
        const float* gradLoss,
        const float* input,
        const float* weight,
        const float* offset_i,
        const float* offset_j,
        float* gradOffset_i
    ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
        float floatOutput = 0.0;

        const int intSample  = ( intIndex / SIZE_3(gradOffset_i) / SIZE_2(gradOffset_i) / SIZE_1(gradOffset_i) ) % SIZE_0(gradOffset_i);
        const int intDepth   = ( intIndex / SIZE_3(gradOffset_i) / SIZE_2(gradOffset_i)                        ) % SIZE_1(gradOffset_i);
        const int i          = ( intIndex / SIZE_3(gradOffset_i)                                               ) % SIZE_2(gradOffset_i);
        const int j          = ( intIndex                                                                      ) % SIZE_3(gradOffset_i);

        int k = intDepth / F_SIZE;
        int l = intDepth % F_SIZE;

        for (int c = 0; c < 3; c++) 
        {
        float delta     = VALUE_4(gradLoss, intSample, c, i, j);
        float w         = VALUE_4(weight, intSample, k*F_SIZE+l, i, j);
        float alpha     = VALUE_4(offset_i, intSample, k*F_SIZE+l, i, j);
        float beta      = VALUE_4(offset_j, intSample, k*F_SIZE+l, i, j);
        int A           = (int) alpha;
        int B           = (int) beta;

        int i_k_A = i+k*DILATION+A;
        if(i_k_A < 0)
            i_k_A = 0;
        if(i_k_A > SIZE_2(input) - 1)
            i_k_A = SIZE_2(input) - 1;

        int j_l_B = j+l*DILATION+B;
        if(j_l_B < 0)
            j_l_B = 0;
        if(j_l_B > SIZE_3(input) - 1)
            j_l_B = SIZE_3(input) - 1;

        int i_k_A_1 = i+k*DILATION+A+1;
        if(i_k_A_1 < 0)
            i_k_A_1 = 0;
        if(i_k_A_1 > SIZE_2(input) - 1)
            i_k_A_1 = SIZE_2(input) - 1;

        int j_l_B_1 = j+l*DILATION+B+1;
        if(j_l_B_1 < 0)
            j_l_B_1 = 0;
        if(j_l_B_1 > SIZE_3(input) - 1)
            j_l_B_1 = SIZE_3(input) - 1;

        floatOutput += delta * w * (
            - VALUE_4(input, intSample, c, i_k_A, j_l_B)*(1-(beta-(float)B)) + 
            VALUE_4(input, intSample, c, i_k_A_1, j_l_B)*(1-(beta-(float)B)) - 
            VALUE_4(input, intSample, c, i_k_A, j_l_B_1)*(beta-(float)B) + 
            VALUE_4(input, intSample, c, i_k_A_1, j_l_B_1)*(beta-(float)B)
            );
        }

        gradOffset_i[intIndex] = floatOutput;
    } }
'''

kernel_AdaCoF_updateGradBeta = '''
    extern "C" __global__ void kernel_AdaCoF_updateGradBeta(
        const int n,
        const float* gradLoss,
        const float* input,
        const float* weight,
        const float* offset_i,
        const float* offset_j,
        float* gradOffset_j
    ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
        float floatOutput = 0.0;

        const int intSample  = ( intIndex / SIZE_3(gradOffset_j) / SIZE_2(gradOffset_j) / SIZE_1(gradOffset_j) ) % SIZE_0(gradOffset_j);
        const int intDepth   = ( intIndex / SIZE_3(gradOffset_j) / SIZE_2(gradOffset_j)                        ) % SIZE_1(gradOffset_j);
        const int i          = ( intIndex / SIZE_3(gradOffset_j)                                               ) % SIZE_2(gradOffset_j);
        const int j          = ( intIndex                                                                      ) % SIZE_3(gradOffset_j);

        int k = intDepth / F_SIZE;
        int l = intDepth % F_SIZE;

        for (int c = 0; c < 3; c++) 
        {
        float delta     = VALUE_4(gradLoss, intSample, c, i, j);
        float w         = VALUE_4(weight, intSample, k*F_SIZE+l, i, j);
        float alpha     = VALUE_4(offset_i, intSample, k*F_SIZE+l, i, j);
        float beta      = VALUE_4(offset_j, intSample, k*F_SIZE+l, i, j);
        int A           = (int) alpha;
        int B           = (int) beta;

        int i_k_A = i+k*DILATION+A;
        if(i_k_A < 0)
            i_k_A = 0;
        if(i_k_A > SIZE_2(input) - 1)
            i_k_A = SIZE_2(input) - 1;

        int j_l_B = j+l*DILATION+B;
        if(j_l_B < 0)
            j_l_B = 0;
        if(j_l_B > SIZE_3(input) - 1)
            j_l_B = SIZE_3(input) - 1;

        int i_k_A_1 = i+k*DILATION+A+1;
        if(i_k_A_1 < 0)
            i_k_A_1 = 0;
        if(i_k_A_1 > SIZE_2(input) - 1)
            i_k_A_1 = SIZE_2(input) - 1;

        int j_l_B_1 = j+l*DILATION+B+1;
        if(j_l_B_1 < 0)
            j_l_B_1 = 0;
        if(j_l_B_1 > SIZE_3(input) - 1)
            j_l_B_1 = SIZE_3(input) - 1;

        floatOutput += delta * w * (
            - VALUE_4(input, intSample, c, i_k_A, j_l_B)*(1-(alpha-(float)A)) - 
            VALUE_4(input, intSample, c, i_k_A_1, j_l_B)*(alpha-(float)A) + 
            VALUE_4(input, intSample, c, i_k_A, j_l_B_1)*(1-(alpha-(float)A)) + 
            VALUE_4(input, intSample, c, i_k_A_1, j_l_B_1)*(alpha-(float)A)
            );
        }

        gradOffset_j[intIndex] = floatOutput;
    } }
'''


def cupy_kernel(strFunction, intFilterSize, intDilation, objectVariables):
    strKernel = globals()[strFunction]

    while True:
        objectMatch = re.search('(SIZE_)([0-4])(\()([^\)]*)(\))', strKernel)

        if objectMatch is None:
            break
        # end

        intArg = int(objectMatch.group(2))

        strTensor = objectMatch.group(4)
        intSizes = objectVariables[strTensor].size()

        strKernel = strKernel.replace(objectMatch.group(), str(intSizes[intArg]))
    # end

    while True:
        objectMatch = re.search('(VALUE_)([0-4])(\()([^\)]+)(\))', strKernel)

        if objectMatch is None:
            break
        # end

        intArgs = int(objectMatch.group(2))
        strArgs = objectMatch.group(4).split(',')

        strTensor = strArgs[0]
        intStrides = objectVariables[strTensor].stride()
        strIndex = ['((' + strArgs[intArg + 1].replace('{', '(').replace('}', ')').strip() + ')*' + str(intStrides[intArg]) + ')' for intArg in range(intArgs)]

        strKernel = strKernel.replace(objectMatch.group(0), strTensor + '[' + str.join('+', strIndex) + ']')
    # end

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

    strKernel = strKernel.replace('F_SIZE', str(intFilterSize))
    strKernel = strKernel.replace('DILATION', str(intDilation))

    return strKernel


# end

@cupy._util.memoize(for_each_device=True)
def cupy_launch(strFunction, strKernel):
    return cupy.cuda.compile_with_cache(strKernel).get_function(strFunction)


# end

class FunctionSynth(torch.autograd.Function):
    # end
    @staticmethod
    def forward(ctx, input, weight, offset_i, offset_j, dilation):
        ctx.save_for_backward(input, weight, offset_i, offset_j)
        ctx.dilation = dilation

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
        assert (offset_i.is_contiguous() == True)
        assert (offset_j.is_contiguous() == True)

        output = input.new_zeros(intSample, intInputDepth, intOutputHeight, intOutputWidth)

        if input.is_cuda == True:
            class Stream:
                ptr = torch.cuda.current_stream().cuda_stream

            # end

            n = output.nelement()
            cupy_launch('kernel_AdaCoF_updateOutput', cupy_kernel('kernel_AdaCoF_updateOutput', intFilterSize, dilation, {
                'input': input,
                'weight': weight,
                'offset_y': offset_i,
                'offset_x': offset_j,
                'output': output
            }))(
                grid=tuple([int((n + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[n, input.data_ptr(), weight.data_ptr(), offset_i.data_ptr(), offset_j.data_ptr(), output.data_ptr()],
                stream=Stream
            )

        elif input.is_cuda == False:
            raise NotImplementedError()

        # end

        return output

    # end
    @staticmethod
    def backward(ctx, gradOutput):
        input, weight, offset_i, offset_j = ctx.saved_tensors
        dilation = ctx.dilation

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

        gradInput = input.new_zeros(intSample, intInputDepth, intInputHeight, intInputWidth) if ctx.needs_input_grad[0] == True else None
        gradWeight = input.new_zeros(intSample, intFilterSize ** 2, intOutputHeight, intOutputWidth) if ctx.needs_input_grad[1] == True else None
        gradOffset_i = input.new_zeros(intSample, intFilterSize ** 2, intOutputHeight, intOutputWidth) if ctx.needs_input_grad[2] == True else None
        gradOffset_j = input.new_zeros(intSample, intFilterSize ** 2, intOutputHeight, intOutputWidth) if ctx.needs_input_grad[2] == True else None

        if input.is_cuda == True:
            class Stream:
                ptr = torch.cuda.current_stream().cuda_stream

            # end

            # weight grad
            n_w = gradWeight.nelement()
            cupy_launch('kernel_AdaCoF_updateGradWeight', cupy_kernel('kernel_AdaCoF_updateGradWeight', intFilterSize, dilation, {
                'gradLoss': gradOutput,
                'input': input,
                'offset_i': offset_i,
                'offset_j': offset_j,
                'gradWeight': gradWeight
            }))(
                grid=tuple([int((n_w + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[n_w, gradOutput.data_ptr(), input.data_ptr(), offset_i.data_ptr(), offset_j.data_ptr(), gradWeight.data_ptr()],
                stream=Stream
            )

            # alpha grad
            n_i = gradOffset_i.nelement()
            cupy_launch('kernel_AdaCoF_updateGradAlpha', cupy_kernel('kernel_AdaCoF_updateGradAlpha', intFilterSize, dilation, {
                'gradLoss': gradOutput,
                'input': input,
                'weight': weight,
                'offset_i': offset_i,
                'offset_j': offset_j,
                'gradOffset_i': gradOffset_i
            }))(
                grid=tuple([int((n_i + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[n_i, gradOutput.data_ptr(), input.data_ptr(), weight.data_ptr(), offset_i.data_ptr(), offset_j.data_ptr(), gradOffset_i.data_ptr()],
                stream=Stream
            )

            # beta grad
            n_j = gradOffset_j.nelement()
            cupy_launch('kernel_AdaCoF_updateGradBeta', cupy_kernel('kernel_AdaCoF_updateGradBeta', intFilterSize, dilation, {
                'gradLoss': gradOutput,
                'input': input,
                'weight': weight,
                'offset_i': offset_i,
                'offset_j': offset_j,
                'gradOffset_j': gradOffset_j
            }))(
                grid=tuple([int((n_j + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[n_j, gradOutput.data_ptr(), input.data_ptr(), weight.data_ptr(), offset_i.data_ptr(), offset_j.data_ptr(), gradOffset_j.data_ptr()],
                stream=Stream
            )

        elif input.is_cuda == False:
            raise NotImplementedError()

        # end

        return gradInput, gradWeight, gradOffset_i, gradOffset_j, None

# end


# end
# import cupy
# import torch
# import re
# import math
#
# kernel_Synth_updateOutput = '''
# extern "C" __global__ void kernel_Synth_updateOutput(
#         const int n,
#         const float* input,
#         const float* weight,
#         const float* offset_y,
#         const float* offset_x,
#         float* output
# )
# {
#     for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
#         float dblOutput = 0.0;
#
#         const int intSample = ( intIndex / SIZE_3(output) / SIZE_2(output) / SIZE_1(output) ) % SIZE_0(output);
#         const int intDepth  = ( intIndex / SIZE_3(output) / SIZE_2(output)                  ) % SIZE_1(output);
#         const int y         = ( intIndex / SIZE_3(output)                                   ) % SIZE_2(output);
#         const int x         = ( intIndex                                                    ) % SIZE_3(output);
#
#         for (int row = 0; row < F_SIZE; row += 1) {
#             for (int col = 0; col < F_SIZE; col += 1) {
#                 float w         = VALUE_4(weight, intSample, row*F_SIZE+col, y, x);
#                 float alpha     = VALUE_4(offset_x, intSample, row*F_SIZE+col, y, x);
#                 float beta      = VALUE_4(offset_y, intSample, row*F_SIZE+col, y, x);
#                 int intAlpha    = (int)alpha;
#                 int intBeta     = (int)beta;
#
#                 int bottom = y + row*DILATION + intAlpha;
#                 if(bottom < 0)
#                     bottom = 0;
#                 if(bottom > SIZE_2(input) - 1)
#                     bottom = SIZE_2(input) - 1;
#
#                 int left = x + col*DILATION + intBeta;
#                 if(left < 0)
#                     left = 0;
#                 if(left > SIZE_3(input) - 1)
#                     left = SIZE_3(input) - 1;
#
#                 int top = y + row*DILATION + intAlpha + 1;
#                 if(top < 0)
#                     top = 0;
#                 if(top > SIZE_2(input) - 1)
#                     top = SIZE_2(input) - 1;
#
#                 int right = x + col*DILATION + intBeta + 1;
#                 if(right < 0)
#                     right = 0;
#                 if(right > SIZE_3(input) - 1)
#                     right = SIZE_3(input) - 1;
#
#                 float alphaTrunc = alpha - (float)intAlpha;
#                 float betaTrunc = beta - (float)intBeta;
#
#                 dblOutput += w * (
#                     VALUE_4(input, intSample, intDepth, bottom, left)*(1 - alphaTrunc)*(1 - betaTrunc) +
#                     VALUE_4(input, intSample, intDepth, top, left)*alphaTrunc*(1 - betaTrunc) +
#                     VALUE_4(input, intSample, intDepth, bottom, right)*(1 - alphaTrunc)*betaTrunc +
#                     VALUE_4(input, intSample, intDepth, top, right)*alphaTrunc*betaTrunc
#                 );
#             }
#         }
#
#         output[intIndex] = dblOutput;
#     }
# }
# '''
#
# kernel_Synth_updateGradWeight = '''
# extern "C" __global__ void kernel_Synth_updateGradWeight(
#     const int n,
#     const float* gradLoss,
#     const float* input,
#     const float* offset_y,
#     const float* offset_x,
#     float* gradWeight
# )
# {
#     for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
#         float floatOutput = 0.0;
#
#         const int intSample  = ( intIndex / SIZE_3(gradWeight) / SIZE_2(gradWeight) / SIZE_1(gradWeight) ) % SIZE_0(gradWeight);
#         const int intDepth   = ( intIndex / SIZE_3(gradWeight) / SIZE_2(gradWeight)                      ) % SIZE_1(gradWeight);
#         const int y          = ( intIndex / SIZE_3(gradWeight)                                           ) % SIZE_2(gradWeight);
#         const int x          = ( intIndex                                                                ) % SIZE_3(gradWeight);
#
#         int row = intDepth / F_SIZE;
#         int col = intDepth % F_SIZE;
#
#         for (int depth = 0; depth < 3; depth++) {
#             float delta     = VALUE_4(gradLoss, intSample, depth, y, x);
#             float alpha     = VALUE_4(offset_y, intSample, row*F_SIZE+col, y, x);
#             float beta      = VALUE_4(offset_x, intSample, row*F_SIZE+col, y, x);
#             int intAlpha    = (int)alpha;
#             int intBeta     = (int)beta;
#
#             int bottom = y + row*DILATION + intAlpha;
#             if(bottom < 0)
#                 bottom = 0;
#             if(bottom > SIZE_2(input) - 1)
#                 bottom = SIZE_2(input) - 1;
#
#             int left = x + col*DILATION + intBeta;
#             if(left < 0)
#                 left = 0;
#             if(left > SIZE_3(input) - 1)
#                 left = SIZE_3(input) - 1;
#
#             int top = y + row*DILATION + intAlpha + 1;
#             if(top < 0)
#                 top = 0;
#             if(top > SIZE_2(input) - 1)
#                 top = SIZE_2(input) - 1;
#
#             int right = x + col*DILATION + intBeta + 1;
#             if(right < 0)
#                 right = 0;
#             if(right > SIZE_3(input) - 1)
#                 right = SIZE_3(input) - 1;
#
#             float alphaTrunc = alpha - (float)intAlpha;
#             float betaTrunc = beta - (float)intBeta;
#
#             floatOutput += delta * (
#                 VALUE_4(input, intSample, depth, bottom, left)*(1 - alphaTrunc)*(1 - betaTrunc) +
#                 VALUE_4(input, intSample, depth, top, left)*alphaTrunc*(1 - betaTrunc) +
#                 VALUE_4(input, intSample, depth, bottom, right)*(1 - alphaTrunc)*betaTrunc +
#                 VALUE_4(input, intSample, depth, top, right)*alphaTrunc*betaTrunc
#             );
#         }
#
#         gradWeight[intIndex] = floatOutput;
#     }
# }
# '''
#
# kernel_Synth_updateGradAlpha = '''
# extern "C" __global__ void kernel_Synth_updateGradAlpha(
#     const int n,
#     const float* gradLoss,
#     const float* input,
#     const float* weight,
#     const float* offset_y,
#     const float* offset_x,
#     float* gradOffset_y
# )
# {
#     for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
#         float floatOutput = 0.0;
#
#         const int intSample  = ( intIndex / SIZE_3(gradOffset_y) / SIZE_2(gradOffset_y) / SIZE_1(gradOffset_y) ) % SIZE_0(gradOffset_y);
#         const int intDepth   = ( intIndex / SIZE_3(gradOffset_y) / SIZE_2(gradOffset_y)                        ) % SIZE_1(gradOffset_y);
#         const int y          = ( intIndex / SIZE_3(gradOffset_y)                                               ) % SIZE_2(gradOffset_y);
#         const int x          = ( intIndex                                                                      ) % SIZE_3(gradOffset_y);
#
#         int row = intDepth / F_SIZE;
#         int col = intDepth % F_SIZE;
#
#         for (int depth = 0; depth < 3; depth++) {
#             float delta     = VALUE_4(gradLoss, intSample, depth, y, x);
#             float w         = VALUE_4(weight, intSample, row*F_SIZE + col, y, x);
#             float alpha     = VALUE_4(offset_y, intSample, row*F_SIZE + col, y, x);
#             float beta      = VALUE_4(offset_x, intSample, row*F_SIZE + col, y, x);
#             int intAlpha    = (int)alpha;
#             int intBeta     = (int)beta;
#
#             int bottom = y + row*DILATION + intAlpha;
#             if(bottom < 0)
#                 bottom = 0;
#             if(bottom > SIZE_2(input) - 1)
#                 bottom = SIZE_2(input) - 1;
#
#             int left = x + col*DILATION + intBeta;
#             if(left < 0)
#                 left = 0;
#             if(left > SIZE_3(input) - 1)
#                 left = SIZE_3(input) - 1;
#
#             int top = y + row*DILATION + intAlpha + 1;
#             if(top < 0)
#                 top = 0;
#             if(top > SIZE_2(input) - 1)
#                 top = SIZE_2(input) - 1;
#
#             int right = x + col*DILATION + intBeta + 1;
#             if(right < 0)
#                 right = 0;
#             if(right > SIZE_3(input) - 1)
#                 right = SIZE_3(input) - 1;
#
#
#             float betaTrunc = beta - (float)intBeta;
#
#             floatOutput += delta * w * (
#                 - VALUE_4(input, intSample, depth, bottom, left)*(1 - betaTrunc)
#                 + VALUE_4(input, intSample, depth, top, left)*(1 - betaTrunc)
#                 - VALUE_4(input, intSample, depth, bottom, right)*betaTrunc
#                 + VALUE_4(input, intSample, depth, top, right)*betaTrunc
#             );
#         }
#
#         gradOffset_y[intIndex] = floatOutput;
#     }
# }
# '''
#
# kernel_Synth_updateGradBeta = '''
# extern "C" __global__ void kernel_Synth_updateGradBeta(
#     const int n,
#     const float* gradLoss,
#     const float* input,
#     const float* weight,
#     const float* offset_y,
#     const float* offset_x,
#     float* gradOffset_x
# )
# {
#     for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
#         float floatOutput = 0.0;
#
#         const int intSample  = ( intIndex / SIZE_3(gradOffset_x) / SIZE_2(gradOffset_x) / SIZE_1(gradOffset_x) ) % SIZE_0(gradOffset_x);
#         const int intDepth   = ( intIndex / SIZE_3(gradOffset_x) / SIZE_2(gradOffset_x)                        ) % SIZE_1(gradOffset_x);
#         const int y          = ( intIndex / SIZE_3(gradOffset_x)                                               ) % SIZE_2(gradOffset_x);
#         const int x          = ( intIndex                                                                      ) % SIZE_3(gradOffset_x);
#
#         int row = intDepth / F_SIZE;
#         int col = intDepth % F_SIZE;
#
#         for (int depth = 0; depth < 3; depth++) {
#             float delta     = VALUE_4(gradLoss, intSample, depth, y, x);
#             float w         = VALUE_4(weight, intSample, row*F_SIZE + col, y, x);
#             float alpha     = VALUE_4(offset_y, intSample, row*F_SIZE + col, y, x);
#             float beta      = VALUE_4(offset_x, intSample, row*F_SIZE + col, y, x);
#             int intAlpha    = (int)alpha;
#             int intBeta     = (int)beta;
#
#             int bottom = y + row*DILATION + intAlpha;
#             if(bottom < 0)
#                 bottom = 0;
#             if(bottom > SIZE_2(input) - 1)
#                 bottom = SIZE_2(input) - 1;
#
#             int left = x + col*DILATION + intBeta;
#             if(left < 0)
#                 left = 0;
#             if(left > SIZE_3(input) - 1)
#                 left = SIZE_3(input) - 1;
#
#             int top = y + row*DILATION + intAlpha + 1;
#             if(top < 0)
#                 top = 0;
#             if(top > SIZE_2(input) - 1)
#                 top = SIZE_2(input) - 1;
#
#             int right = x + col*DILATION + intBeta + 1;
#             if(right < 0)
#                 right = 0;
#             if(right > SIZE_3(input) - 1)
#                 right = SIZE_3(input) - 1;
#
#             float alphaTrunc = alpha - (float)intAlpha;
#
#             floatOutput += delta * w * (
#                 - VALUE_4(input, intSample, depth, bottom, left)*(1 - alphaTrunc)
#                 - VALUE_4(input, intSample, depth, top, left)*alphaTrunc
#                 + VALUE_4(input, intSample, depth, bottom, right)*(1 - alphaTrunc)
#                 + VALUE_4(input, intSample, depth, top, right)*alphaTrunc
#             );
#         }
#
#         gradOffset_x[intIndex] = floatOutput;
#     }
# }
# '''
#
# def cupy_kernel(strFunc, intFilterSize, intDilation, objVars):
#     strKernel = globals()[strFunc]
#
#     # purpose: getting size of tensor axis
#     while True:
#         objMatch = re.search('(SIZE_)([0-4])(\()([^\)]*)(\))', strKernel)
#
#         if objMatch is None:
#             break
#
#         intArg = int(objMatch.group(2))
#
#         strTensor = objMatch.group(4)
#         intSizes = objVars[strTensor].size()
#
#         strKernel = strKernel.replace(objMatch.group(0), str(intSizes[intArg]))
#
#     # purpose: getting value at certain index of tensor, ex. tensor[index]
#     while True:
#         objMatch = re.search('(VALUE_)([0-4])(\()([^\)]+)(\))', strKernel)
#
#         if objMatch is None:
#             break
#
#         intArgs = int(objMatch.group(2))
#         strArgs = objMatch.group(4).split(',')
#
#         strTensor = strArgs[0]
#         intStrides = objVars[strTensor].stride()
#         strIndex = ['((' + strArgs[intArg + 1].replace('{', '(').replace('}', ')').strip() + ')*' + str(intStrides[intArg]) + ')' for intArg in range(intArgs)]
#
#         strKernel = strKernel.replace(objMatch.group(0), strTensor + '[' + str.join('+', strIndex) + ']')
#
#     # setting macros
#     strKernel = strKernel.replace('F_SIZE', str(intFilterSize))
#     strKernel = strKernel.replace('DILATION', str(intDilation))
#
#     return strKernel
#
# @cupy._util.memoize(for_each_device=True)
# def cupy_launch(strFunc, strKernel):
#     module = cupy.RawModule(code = strKernel)
#     return module.get_function(strFunc)
#
# class FunctionSynth(torch.autograd.Function):
#     @staticmethod
#     def forward(context, input, weight, offset_y, offset_x, dilation):
#         context.save_for_backward(input, weight, offset_y, offset_x)
#         context.dilation = dilation
#
#         intSample = input.size(0)
#         intInputDepth = input.size(1)
#         intInputHeight = input.size(2)
#         intInputWidth = input.size(3)
#         intFilterSize = int(math.sqrt(weight.size(1)))
#         intOutputHeight = weight.size(2)
#         intOutputWidth = weight.size(3)
#
#         assert (intInputHeight - ((intFilterSize - 1) * dilation + 1) == intOutputHeight - 1)
#         assert (intInputWidth - ((intFilterSize - 1) * dilation + 1) == intOutputWidth - 1)
#
#         assert (input.is_contiguous() == True)
#         assert (weight.is_contiguous() == True)
#         assert (offset_y.is_contiguous() == True)
#         assert (offset_x.is_contiguous() == True)
#
#         output = input.new_zeros(intSample, intInputDepth, intOutputHeight, intOutputWidth)
#
#         if input.is_cuda:
#             # torch.cuda.current_stream() is a function that returns the current CUDA stream for the default CUDA device.
#             # A CUDA stream is a sequence of operations that execute on the GPU in the order they were issued by the host (CPU).
#             # This allows for concurrent execution of operations where possible, such as overlapping data transfers with computations.
#             # .cuda_stream is an attribute of the object returned by torch.cuda.current_stream().
#             # It likely represents the actual underlying CUDA stream object (often a pointer or a low-level handler) that interfaces directly with the CUDA driver API.
#
#             class Stream:
#                 ptr = torch.cuda.current_stream().cuda_stream
#
#             n = output.nelement()
#             cupy_launch('kernel_Synth_updateOutput', cupy_kernel('kernel_Synth_updateOutput', intFilterSize, dilation, {
#                 'input': input,
#                 'weight': weight,
#                 'offset_y': offset_y,
#                 'offset_x': offset_x,
#                 'output': output
#             }))(
#                 grid=tuple([math.ceil(n / 512), 1, 1]),
#                 block=tuple([512, 1, 1]),
#                 args=[n, input.data_ptr(), weight.data_ptr(), offset_y.data_ptr(), offset_x.data_ptr(), output.data_ptr()],
#                 stream=Stream
#             )
#         else:
#             raise NotImplementedError()
#
#         return output
#
#     @staticmethod
#     def backward(context, gradOutput):
#         input, weight, offset_y, offset_x = context.saved_tensors
#         dilation = context.dilation
#
#         intSample = input.size(0)
#         intInputDepth = input.size(1)
#         intInputHeight = input.size(2)
#         intInputWidth = input.size(3)
#         intFilterSize = int(math.sqrt(weight.size(1)))
#         intOutputHeight = weight.size(2)
#         intOutputWidth = weight.size(3)
#
#         assert (intInputHeight - ((intFilterSize - 1) * dilation + 1) == intOutputHeight - 1)
#         assert (intInputWidth - ((intFilterSize - 1) * dilation + 1) == intOutputWidth - 1)
#
#         assert (gradOutput.is_contiguous() == True)
#
#         gradInput = input.new_zeros(intSample, intInputDepth, intInputHeight, intInputWidth) if context.needs_input_grad[0] else None
#         gradWeight = input.new_zeros(intSample, intFilterSize ** 2, intOutputHeight, intOutputWidth) if context.needs_input_grad[1] else None
#         gradOffset_y = input.new_zeros(intSample, intFilterSize ** 2, intOutputHeight, intOutputWidth) if context.needs_input_grad[2] else None
#         gradOffset_x = input.new_zeros(intSample, intFilterSize ** 2, intOutputHeight, intOutputWidth) if context.needs_input_grad[2] else None
#
#         if input.is_cuda:
#             class Stream:
#                 ptr = torch.cuda.current_stream().cuda_stream
#
#             # weight grad
#             n = gradWeight.nelement()
#             cupy_launch('kernel_Synth_updateGradWeight', cupy_kernel('kernel_Synth_updateGradWeight', intFilterSize, dilation, {
#                 'gradLoss': gradOutput,
#                 'input': input,
#                 'offset_y': offset_y,
#                 'offset_x': offset_x,
#                 'gradWeight': gradWeight
#             }))(
#                 grid=tuple([math.ceil(n / 512), 1, 1]),
#                 block=tuple([512, 1, 1]),
#                 args=[n, gradOutput.data_ptr(), input.data_ptr(), offset_y.data_ptr(), offset_x.data_ptr(), gradWeight.data_ptr()],
#                 stream=Stream
#             )
#
#             # alpha grad
#             n = gradOffset_y.nelement()
#             cupy_launch('kernel_Synth_updateGradAlpha', cupy_kernel('kernel_Synth_updateGradAlpha', intFilterSize, dilation, {
#                 'gradLoss': gradOutput,
#                 'input': input,
#                 'weight': weight,
#                 'offset_y': offset_y,
#                 'offset_x': offset_x,
#                 'gradOffset_y': gradOffset_y
#             }))(
#                 grid=tuple([math.ceil(n / 512), 1, 1]),
#                 block=tuple([512, 1, 1]),
#                 args=[n, gradOutput.data_ptr(), input.data_ptr(), weight.data_ptr(), offset_y.data_ptr(), offset_x.data_ptr(), gradOffset_y.data_ptr()],
#                 stream=Stream
#             )
#
#             # beta grad
#             n = gradOffset_x.nelement()
#             cupy_launch('kernel_Synth_updateGradBeta', cupy_kernel('kernel_Synth_updateGradBeta', intFilterSize, dilation, {
#                 'gradLoss': gradOutput,
#                 'input': input,
#                 'weight': weight,
#                 'offset_y': offset_y,
#                 'offset_x': offset_x,
#                 'gradOffset_x': gradOffset_x
#             }))(
#                 grid=tuple([math.ceil(n / 512), 1, 1]),
#                 block=tuple([512, 1, 1]),
#                 args=[n, gradOutput.data_ptr(), input.data_ptr(), weight.data_ptr(), offset_y.data_ptr(), offset_x.data_ptr(), gradOffset_x.data_ptr()],
#                 stream=Stream
#             )
#         else:
#             raise NotImplementedError()
#
#         return gradInput, gradWeight, gradOffset_y, gradOffset_x, None
