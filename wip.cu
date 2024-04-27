
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


