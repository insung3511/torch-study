import torch

tensor1 = torch.tensor([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]])
print(tensor1)

tensor2 = torch.tensor([[[9., 10.], [11., 12.]], [[13., 14], [15., 16.]]])
print(tensor2)

print("#######################")

sum_tensor = tensor1 + tensor2
print(sum_tensor)

sub_tensor = tensor1 - tensor2
print(sub_tensor)

mul_tensor = tensor1 * tensor2
print(mul_tensor)

div_tensor = tensor1 / tensor2
print(div_tensor)

print("#######################")
print(torch.add(tensor1, tensor2))
print(torch.sub(tensor1, tensor2))
print(torch.mul(tensor1, tensor2))
print(torch.div(tensor1, tensor2))
print(torch.matmul(tensor1, tensor2))