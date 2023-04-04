class Neuron:
    def __init__(self, inputs, learningRate=0.01, start=False, end=False):
        self.inputs = inputs # single val if start else array
        self.bias = 1
        self.start = start
        self.end = end
        self.learningRate = learningRate
        
        self.weights = []
        self.value = 0.0 # weighted sum
        self.output = 0.0
        self.grad = []
        
        self.activation = self.tanh
        self.activationPrime = self.tanh_Prime
        
        if self.start:
            # no weights
            self.value = inputs # input will be a single value
            self.output = self.activation(self.value) # no activation required for inputs ??
            # self.output = self.value
        else:
            # not the starting layer
            self.weights = np.array([np.random.uniform(-1.0, 1.0) for _ in range(len(self.inputs))]) # [0.2, 0.8, 0.34]
            self.value = np.dot(self.weights, self.inputs) + self.bias # raw result, before activation func

            if self.end:
                # activation = softmax
                # calculated at the layer level
                # thus no activation needed
                self.output = self.value
            else:
                # activation = sigmoid/ReLU
                self.output = self.activation(self.value)
    
    def tanh(self, z):
        return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def ReLU(self, z):
        return max(0, z)
    
    def tanh_Prime(self, z):
        return 1 - np.power(self.tanh(z), 2)
    
    def sigmoid_Prime(self, z):
        val = self.sigmoid(z)
        return val * (1 - val)
    
    def ReLU_Prime(self, z):
        return 0 if z < 0 else 1
    
    def forward(self, X):
        self.inputs = X
        self.grad = []
        if self.end:
            # no activation
            self.value = np.dot(self.weights, self.inputs) + self.bias
            # activation softmax
            self.output = self.value
        elif self.start:
            return
        else:
            self.value = np.dot(self.weights, self.inputs) + self.bias
            self.output = self.activation(self.value)
                
    def update_weights(self, grad_array):
        if self.start:
            return
        
        if self.end:
            # grad_array -> int val
            z_grad = grad_array
            self.grad = np.dot(z_grad, self.weights)
            self.weights -= self.learningRate * sum(self.grad)
            return
        
        g_next = sum(grad_array)
        z_grad = g_next * self.activationPrime(self.value)
        self.grad = np.dot(z_grad, self.weights) # [] array of grads for a specific weight
        self.weights -= self.learningRate * sum(self.grad) # better to sum up when updating
    
class Layer:
    def __init__(self, size, input_from_prev, learningRate=0.01, start=False, end=False):
        self.size = size
        self.inputs = input_from_prev
        self.start = start
        self.end = end
        
        if self.start:
            # Individual Neuron
            self.neurons = [Neuron(inp, learningRate=0.01, start=True) for inp in self.inputs]
        else:
            # Array from prev passed as input
            self.neurons = [Neuron(self.inputs, learningRate=0.01, end=self.end) for _ in range(self.size)]
    
    def activation_Softmax(self):
        # converts outputs into a probabilistic distribution
        # values can grow extremely fast, x - np.max(x) will keep the max value at 0
        # and other vals below 0
        # thus the resulting exonentiation is in the range 0-1
        x = [n.output for n in self.neurons]
        exps = np.exp(x - np.max(x)) # to prevent overflow
        return np.array(exps / np.sum(exps))
    
    def backpass(self, next_layer_grads):
        for grad, n in zip(next_layer_grads, self.neurons):
            n.update_weights(grad)
            
    def forward(self, X):
        for n in self.neurons:
            n.forward(X)
            
    def getValues(self):
        # weighted sum
        return [n.value for n in self.neurons]
    
    def getOutput(self):
        return [n.output for n in self.neurons]
    
    def getGrads(self):
        return np.array([n.grad for n in self.neurons])

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
    
class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        
        # scaler values
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods  
    
class NN:
    def __init__(self, inputs, y, n_hidden, hidden_size=[3], activation="tanh"):
        self.X = self.normalizer(inputs)
        self.y_true = y
        
        self.learning_rate = 0.1
        self.epochs = 100
        self.cost_history = []
        
        self.hidden_layer_size = 5
        self.hidden_layers = n_hidden
        self.layers = []
        self.start_L = Layer(len(self.X), self.X, self.learning_rate, start=True)
        self.layers.append(self.start_L)
        
        layer_out = self.start_L.getOutput()
        for num in range(self.hidden_layers):
            layer = Layer(self.hidden_layer_size, layer_out, self.learning_rate)
            self.layers.append(layer)
            layer_out = layer.getOutput()
        
        self.end_L = Layer(len(self.y_true), self.layers[-1].getOutput(), self.learning_rate, end=True)
        self.layers.append(self.end_L)
        
    def normalizer(self, arr: list):
        return np.array([x/sum(arr) for x in arr])
    
    def getTrainingPredictions(self):
        return self.end_L.activation_Softmax()
    
    def getArgMax(self):
        return np.argmax(self.getTrainingPredictions())
    
    def forward(self):
        X = self.start_L.getOutput()
        for layer in self.layers[1:]:
            layer.forward(X)
            X = layer.getOutput()
    
    def error_Prime(self):
        y_pred = self.getTrainingPredictions()
        error = y_pred - self.y_true
        return error
    
    def backpass(self):
        y_pred = self.error_Prime()
        y_pred = np.clip(y_pred, 1e-7, 1-1e-7)
        layer_grads = y_pred
        
        for layer in reversed(self.layers):
            layer.backpass(layer_grads)
            layer_grads = layer.getGrads().T
            
    def getError(self):
        y_pred = self.getTrainingPredictions()
        return 0.5 * np.square(y_pred - self.y_true)
    
    def train(self):
        for epoch in range(self.epochs):
            error = sum(self.getError())
            self.cost_history.append(error)
            self.backpass()
            self.forward()
