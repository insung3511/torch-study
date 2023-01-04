import torch

# Create scalar: 1
scalar1 = torch.tensor([1.0])
print(scalar1)

# Create scalar: 3
scalar2 = torch.tensor([3.0])
print(scalar2)

# [Text] Okay, here is the caculating scalar by scalar.
# [Text] Just a simple are here.
# Get caculating plus scalars
add_scalar = scalar1 + scalar2
print(add_scalar)

# Get caculating product scalars
sub_scalar = scalar1 * scalar2
print(add_scalar)

# Get caculating dividing scalars
div_scalar = scalar1 / scalar2
print(div_scalar)

print("############################")

# [Text] And got other solution.
# Using the torch fuctions.
# P.s. caculating function can only give 2 positional
print(torch.add(scalar1, scalar2))
print(torch.sub(scalar1, scalar2))
print(torch.mul(scalar1, scalar2))
print(torch.div(scalar1, scalar2))
