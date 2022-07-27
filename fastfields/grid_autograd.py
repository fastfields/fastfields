__all__ = ['GridPull', 'GridPush', 'GridCount', 'GridGrad']

import torch
from .utils import custom_fwd, custom_bwd, convert_bound, convert_interpolation
from .ext.bind import (
    grid_pull, grid_pull_backward,
    grid_push, grid_push_backward,
    grid_count, grid_count_backward,
    grid_grad, grid_grad_backward,
    InterpolationType, BoundType)


def make_list(x):
    if not isinstance(x, (list, tuple)):
        x = [x]
    return list(x)


ENUM_TYPE = 'enum'


class GridPull(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, grid, interpolation, bound, extrapolate):

        bound = convert_bound(make_list(bound), as_type=ENUM_TYPE)
        interpolation = convert_interpolation(make_list(interpolation), as_type=ENUM_TYPE)
        extrapolate = int(extrapolate)
        opt = (bound, interpolation, extrapolate)

        # Pull
        output = grid_pull(input, grid, *opt)

        # Context
        ctx.opt = opt
        ctx.save_for_backward(input, grid)

        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        var = ctx.saved_tensors
        opt = ctx.opt
        grad_input = grad_grid = None
        grads = grid_pull_backward(grad, *var, *opt)
        if ctx.needs_input_grad[0]:
            grad_input = grads[0]
            if ctx.needs_input_grad[1]:
                grad_grid = grads[1]
        elif ctx.needs_input_grad[1]:
            grad_grid = grads[0]
        return grad_input, grad_grid, None, None, None


class GridPush(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, grid, shape, interpolation, bound, extrapolate):

        bound = convert_bound(make_list(bound), as_type=ENUM_TYPE)
        interpolation = convert_interpolation(make_list(interpolation), as_type=ENUM_TYPE)
        extrapolate = int(extrapolate)
        opt = (bound, interpolation, extrapolate)

        # Push
        output = grid_push(input, grid, shape, *opt)

        # Context
        ctx.opt = opt
        ctx.save_for_backward(input, grid)

        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        var = ctx.saved_tensors
        opt = ctx.opt
        grad_input = grad_grid = None
        grads = grid_push_backward(grad, *var, *opt)
        if ctx.needs_input_grad[0]:
            grad_input = grads[0]
            if ctx.needs_input_grad[1]:
                grad_grid = grads[1]
        elif ctx.needs_input_grad[1]:
            grad_grid = grads[0]
        return grad_input, grad_grid, None, None, None, None


class GridCount(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, grid, shape, interpolation, bound, extrapolate):

        bound = convert_bound(make_list(bound), as_type=ENUM_TYPE)
        interpolation = convert_interpolation(make_list(interpolation), as_type=ENUM_TYPE)
        extrapolate = int(extrapolate)
        opt = (bound, interpolation, extrapolate)

        # Push
        output = grid_count(grid, shape, *opt)

        # Context
        ctx.opt = opt
        ctx.save_for_backward(grid)

        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        var = ctx.saved_tensors
        opt = ctx.opt
        grad_grid = None
        if ctx.needs_input_grad[0]:
            grad_grid = grid_count_backward(grad, *var, *opt)
        return grad_grid, None, None, None, None


class GridGrad(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, grid, interpolation, bound, extrapolate):

        bound = convert_bound(make_list(bound), as_type=ENUM_TYPE)
        interpolation = convert_interpolation(make_list(interpolation), as_type=ENUM_TYPE)
        extrapolate = int(extrapolate)
        opt = (bound, interpolation, extrapolate)

        # Pull
        output = grid_grad(input, grid, *opt)

        # Context
        ctx.opt = opt
        ctx.save_for_backward(input, grid)

        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        var = ctx.saved_tensors
        opt = ctx.opt
        grad_input = grad_grid = None
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            grads = grid_grad_backward(grad, *var, *opt)
            if ctx.needs_input_grad[0]:
                grad_input = grads[0]
                if ctx.needs_input_grad[1]:
                    grad_grid = grads[1]
            elif ctx.needs_input_grad[1]:
                grad_grid = grads[0]
        return grad_input, grad_grid, None, None, None


