import torch


def test_torch_can_be_imported():
    assert torch.__version__


def test_matrix_multiplication_shape():
    left = torch.randn(2, 3)
    right = torch.randn(3, 4)

    output = left @ right

    assert output.shape == (2, 4)


def test_autograd_creates_correct_gradient():
    x = torch.tensor(2.0, requires_grad=True)
    y = x**2

    y.backward()

    assert x.grad is not None
    assert x.grad.item() == 4.0