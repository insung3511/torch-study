import torch

# In torch tensor using static number like...
# '3' is '3.'. You have to use write dot after number
vector1 = torch.tensor([1.0, 2.0, 3.0])
print(vector1)

vector2 = torch.tensor([4.0, 5.0, 6.0])
print(vector2)

print("###########################")

# And now we gonna do caculating vectors
add_vector = vector1 + vector2
print(add_vector)

sub_vector = vector1 - vector2
print(sub_vector)

mul_vector = vector1 * vector2
print(mul_vector)

div_vector = vector1 / vector2
print(div_vector)

print("###########################")
# Okay, nice. got a other solution
print(torch.add(vector1, vector2))
print(torch.sub(vector1, vector2))
print(torch.mul(vector1, vector2))
print(torch.div(vector1, vector2))
print(torch.dot(vector1, vector2))

"""torch dot
dot product is scalar product. solving answer its like this.
vector(a) = (a, b, c), vector(b) = (d, e, f)
dot_vector(a, b) = a*d + b*e + c*f
"""

vector3 = torch.tensor([4.0, 29.0, 23.0])
print(torch.add(vector1, vector3))
