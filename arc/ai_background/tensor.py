import torch

tensor1 = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
print(tensor1)

tensor2 = torch.tensor([[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14], [15.0, 16.0]]])
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
