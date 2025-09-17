import torch

#Tensors
x = torch.arange(12,dtype=torch.float32)
print('printing the x vector: ',x)

print('number of elements: ', x.numel())

print('x shape: ',x.shape)

x.reshape(3,4)
print('x shape: ',x.shape)

X = x.reshape(3,4)
print('X shape: ',X.shape)

#zeros, ones, rand
torch.zeros((2, 3, 4)) # this creates a 3-d tensor of shape (2, 3, 4) filled with zeros
torch.ones((2, 3, 4)) # this creates a 3rd order tensor of shape (2, 3, 4) filled with ones

# Note: this is an example of how an RGB image can be represented as a 3rd order tensor: (channels, width, height)
# In CNN, the input images are typically represented as 4th order tensors: (batch_size, channels, width, height)
torch.randn(3, 4) # this creates a 2nd order tensor (matrix) of shape (3, 4) filled with random numbers from a normal distribution with mean 0 and variance 1
torch.tensor([[2, 1, 4, 3], 
              [1, 2, 3, 4], 
              [4, 3, 2, 1]]).shape # Create a 2nd order tensor (matrix) with specific values

