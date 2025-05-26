import tinygrad
from tinygrad.tensor import Tensor
from tinygrad.nn.optim import Adam
import tinygrad.nn as nn
from tinygrad.nn.datasets import mnist 

from tinygrad import TinyJit

from tinygrad import Device
print(Device.DEFAULT)

from tinygrad import Tensor, nn

class SimpleCNN:
  def __init__(self):
    self.l1 = nn.Conv2d(1, 32, kernel_size=(3,3))
    self.l2 = nn.Conv2d(32, 64, kernel_size=(3,3))
    self.l3 = nn.Linear(1600, 10)

  def __call__(self, x:Tensor) -> Tensor:
    x = self.l1(x).relu().max_pool2d((2,2))
    x = self.l2(x).relu().max_pool2d((2,2))
    return self.l3(x.flatten(1).dropout(0.5))
  

X_train, Y_train, X_test, Y_test = mnist()
model = SimpleCNN()

optim = nn.optim.Adam(nn.state.get_parameters(model))
batch_size = 128
def step():
  Tensor.training = True  # makes dropout work
  samples = Tensor.randint(batch_size, high=X_train.shape[0])
  X, Y = X_train[samples], Y_train[samples]
  optim.zero_grad()
  loss = model(X).sparse_categorical_crossentropy(Y).backward()
  optim.step()
  return loss


jit_step = TinyJit(step)


for step in range(300):
  loss = jit_step()
  if step%100 == 0:
    Tensor.training = False
    acc = (model(X_test).argmax(axis=1) == Y_test).mean().item()
    print(f"step {step:4d}, loss {loss.item():.2f}, acc {acc*100.:.2f}%")