from lib import *

configs = [
    ([256, 10], 10),
    ([256, 100, 10], 10),
    ([256, 100, 100, 10], 20),
    ([256, 1000, 10], 20)
    ]

for l, _ in configs:
    nn = FullyConnectedNeuralNetwork(l)

    pytorch_total_params = sum(p.numel() for p in nn.parameters() if p.requires_grad)
    print(l, pytorch_total_params)
