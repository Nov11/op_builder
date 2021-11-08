//
// Created by c6s2 on 2021/11/8.
//

#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

class ZeroOutOp : public OpKernel {
public:
    explicit ZeroOutOp(OpKernelConstruction *context) : OpKernel(context) {}

    void Compute(OpKernelContext *context) override {
        const Tensor &input_tensor = context->input(0);
        auto input = input_tensor.flat<int32>();

        Tensor *output_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));
        auto output = output_tensor->template flat<int32>();

        for (int i = 1; i < input.size(); i++) {
            output(i) = 0;
        }
        if (input.size() > 0) output(0) = input(0);
    }
};

REGISTER_KERNEL_BUILDER(Name("ZeroOut").Device(DEVICE_CPU), ZeroOutOp);