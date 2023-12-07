from torch.autograd import Function
from torch.utils.cpp_extension import load
import torch


class AddFunction(Function):
    @staticmethod
    def forward(ctx, a, b):
        assert a.is_cuda and b.is_cuda
        c = torch.empty_like(a)
        add_module.add(a, b, c)
        ctx.c = c
        return c

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output


class SimpleNet(torch.nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.linear1 = torch.nn.Linear(1000, 1000)

    def forward(self, x):
        x = self.linear1(x)
        x = AddFunction.apply(x, x)
        return x


add_module = load(
    name='add_module',
    sources=['autograd_test_kernel_interface.cpp', 'autograd_test_kernel.cu'],
    verbose=True)

model = SimpleNet().cuda()
input = torch.randn(1, 1000).cuda()
output = model(input)
print(output)

