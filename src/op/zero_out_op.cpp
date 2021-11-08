//
// Created by c6s2 on 2021/11/8.
//

#include <tensorflow/core/framework/op.h>

REGISTER_OP("ZeroOut")
                .Input("to_zero: int32")
                .Output("zeroed: int32");