extern "C" __global__ void kernel_Syn_updateOutput(
        const int n,
        const float* input,
        const float* weight, 
        const float* offset_x,
        const float* offset_y,
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

