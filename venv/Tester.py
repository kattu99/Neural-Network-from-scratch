import NeuralNetwork as neuralnet

def main():
    topology = [4, 3, 1]
    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    outputs = [[0], [1], [1], [1]]
    net = neuralnet.Network(topology)
    net.train(inputs, outputs,0.009,0.0015)
    print(net.predict([0,0]))

if __name__ == '__main__':
    main()