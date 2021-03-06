from audioop import mul
import torch

matrix1 = torch.tensor([[1., 2.], 
                        [3., 4.]])
print(matrix1)

matrix2 = torch.tensor([[5., 6.], 
                        [7., 8.]])
print(matrix2)

# Calculating each matrix
print("###########################")

sum_matrix = matrix1 + matrix2
print(sum_matrix)

sub_matrix = matrix1 / matrix2
print(sub_matrix)

div_matrix = matrix1 + matrix2
print(div_matrix)

mul_matrix = matrix1 * matrix2
print(mul_matrix)

print("###########################")

print(torch.add(matrix1, matrix2))
print(torch.sub(matrix1, matrix2))
print(torch.mul(matrix1, matrix2))
print(torch.div(matrix1, matrix2))
print(torch.matmul(matrix1, matrix2))

# matmul is matrix multiple 