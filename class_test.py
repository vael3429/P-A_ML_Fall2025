import torch


x = torch.arange(12, dtype=torch.float32)
print('printing the x vector:', x)



x.numel() 

x.shape 
x.reshape(3, 4) # Reshape to 3 rows and 4 columns

print(x.shape) #note that this does not change the original tensor! you need to assign it to a new variable or overwrite the original one
X = x.reshape(3, 4) # Now x is reshaped
print(X.shape)


torch.zeros((2, 3, 4)) # this creates a 3-d tensor of shape (2, 3, 4) filled with zeros

torch.ones((2, 3, 4)) # this creates a 3-d tensor of shape (2, 3, 4) filled with ones

torch.randn(3, 4)

torch.tensor([[2, 1, 4, 3], 
              [1, 2, 3, 4], 
              [4, 3, 2, 1]]).shape

print(X)
print(X[0])
print(X[-1])
print(X[1:3])

X[1,2] = 17
print(X)

#write values in first two rows to be zero
X[0:2] = 0 
print(X)

print(X[:,1:2]) #columns not rows

#element wise operations

print(torch.exp(X))

y = torch.randn(5)
z = torch.randn(5)

print(y + z)
print(y-z)
print(y*z)
print(y/z)
print(y**z) 


#broadcasting

a=torch.arange(3).reshape((3,1))
b=torch.arange(2).reshape((1,2))

