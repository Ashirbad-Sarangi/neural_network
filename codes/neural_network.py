import numpy as maths

class neural_network :
    
    def __init__(self,
                 df_data , 
                 number_of_neurons = [2,1], 
                 epsilon = 5e-2, 
                 alpha = 0.2, 
                 hidden_function = 'relu', 
                 output_function = 'sigmoid' , 
                 initialisation_technique = 'xavier_normal', 
                 batch_size = 15,
                 optimiser = 'momentum', 
                 gamma = 0.3
                ) :
        self.number_of_hidden_layers = len(number_of_neurons) - 1
        self.number_of_neurons = number_of_neurons
        self.df_data = df_data
        self.epsilon = epsilon 
        self.N = df_data.shape[0]
        self.alpha = alpha
        self.batch_size = batch_size
        self.optimiser = optimiser
        self.gamma = gamma

        functions_list = {'relu':self.relu , 'sigmoid' : self.sigmoid , 'identity' : self.identity}
        intialisation_list = {'xavier_normal':self.xavier_normal , 'he_normal':self.he_normal, 'xavier_uniform' : self.xavier_uniform, 'he_uniform' : self.he_uniform, 'uniform' : self.uniform }
        loss_function_list = {'sigmoid' : self.cross_entropy , 'identity' : self.rmse }
    
        self.hidden = functions_list[hidden_function]
        self.output = functions_list[output_function]
        self.loss = loss_function_list[output_function]
        self.initialise = intialisation_list[initialisation_technique]

        if self.output == self.sigmoid and self.number_of_neurons[-1] != 1 :
            print("Sigmoid must have only one neuron in the output. Please check the structure !")
            raise ValueError('Sigmoid must have only one neuron in the output. Please check the structure !')

        if self.output == self.identity and self.number_of_neurons[-1] != 1 :
            print("Regression must have only one neuron in the output. Please check the structure !")
            raise ValueError('Regression must have only one neuron in the output. Please check the structure !')


    def relu(self,x) :
        """Hidden Activation Function"""
        return max(0,x)
    
    def sigmoid(self,x):
        """Output Function"""
        return 1 / (1 + maths.exp(-x))

    def identity(self,x):
        return x

    def calculate_derivative(self, data, func):
        if func == self.relu :
            if data > 0 : data = 1
            derivative = func(data)

        if func == self.sigmoid :
            derivative = func(data) * ( 1- func(data))

        if func == self.identity :
            derivative = 1

        if func == self.rmse:
            derivative = data

        if func == self.cross_entropy :
            y = data[0]
            fx = data[1]
            x = data[2]
            derivative = (y-fx)*x

        return derivative

    def xavier_normal(self,fan_in, fan_out):
        mu = 0 
        sigma = (2 / (fan_in + fan_out))**0/5

        weight = maths.random.normal(mu, sigma,(fan_out, fan_in))
        bias = maths.random.normal(mu,sigma,(fan_out,1))

        return weight , bias

    def he_normal(self,fan_in, fan_out):
        mu = 0
        sigma = (2/fan_in) ** 0.5
        
        weight = maths.random.normal(0,sigma,(fan_out,fan_in))
        bias = maths.random.normal(0,sigma,(fan_out,1))

        return weight , bias

    def xavier_uniform(self, fan_in, fan_out) :
        weight = maths.random.uniform(-( 6 ** 0.5 / (fan_in + fan_out)) ** 0.5 , (6 / (fan_in + fan_out)) ** 0.5, (fan_out,fan_in))
        bias = maths.random.uniform(-( 6 ** 0.5 / (fan_in + fan_out)) ** 0.5 , (6 / (fan_in + fan_out)) ** 0.5, (fan_out,1))

        return weight, bias
        
    def he_uniform(self, fan_in, fan_out) :
        weight = maths.random.uniform(-( 6 ** 0.5 / fan_in) ** 0.5 , (6 / fan_out) ** 0.5, (fan_out,fan_in))
        bias = maths.random.uniform(-( 6 ** 0.5 / fan_in) ** 0.5 , (6 / fan_out) ** 0.5, (fan_out,1))

        return weight, bias

    
    def uniform(self, fan_in, fan_out) :
        weight = maths.random.uniform(-1/fan_in**0.5 , 1/fan_in**0.5,(fan_out,fan_in))
        bias = maths.random.uniform(-1/fan_in**0.5 , 1/fan_in**0.5,(fan_out,1))

        return weight, bias

    def rmse(y,x):
        return (y-x)**2

    def cross_entropy(y,x):
        return -(y*maths.log(abs(x)) + (1-y)*maths.log(abs(1-x)))
    
    
    def initialisation(self):
        weights = []
        bias = []
        update_weights = []
        update_bias = []
        for l in range(self.number_of_hidden_layers + 1):
            if l == 0 : previous_layer = self.df_data.shape[1] - 1
            else : previous_layer = self.number_of_neurons[l-1]
            present_layer = self.number_of_neurons[l]

            
            weight , bia = self.initialise(previous_layer , present_layer)   
            update_weight = maths.zeros((present_layer, previous_layer))
            update_bia = maths.zeros((present_layer,1))
            
            weights.append(weight)
            bias.append(bia)

            update_weights.append(update_weight)
            update_bias.append(update_bia)
        
        self.weights = weights
        self.bias = bias

        self.momentum_update_weights = update_weights
        self.momentum_update_bias = update_bias


    def forward_propagation(self,x):
        activations = []
        activations.append(x)   # inputdata point is the initial activation
        
        hidden_layers = []
        
        for l in range(self.number_of_hidden_layers):
            neurons = self.weights[l] @ activations[l] + self.bias[l]   # it actually should be activations[l+1] but because indexing in python starts from 0, so it is activations[l]
            activation = maths.matrix([self.hidden(float(neuron)) for neuron in neurons]).reshape(-1,1)
            hidden_layers.append(neurons)
            activations.append(activation)
   
        neurons = self.weights[-1] @ activations[-1] + self.bias[-1]
        outputs = maths.matrix([float(self.output(x)) for x in neurons]).reshape(-1,1)
        hidden_layers.append(neurons)
        activations.append(outputs)
        
        self.hidden_layers = hidden_layers
        self.activations = activations


    def backward_propagation(self,x,y) :
      
        deltas = []
        
        if self.loss == self.rmse :
            t = 2*(self.activations[-1] - y)
            
        if self.loss == self.cross_entropy :
            t = (y, self.activations[-1] , self.hidden_layers[-1])

        
        loss_change_wrt_output = self.calculate_derivative(t,self.loss)

        outputs = maths.matrix([float(self.calculate_derivative(x,self.output)) for x in self.hidden_layers[-1]]).reshape(loss_change_wrt_output.shape)
        delta = maths.multiply(loss_change_wrt_output , outputs)   # change in loss wrt output layer
        deltas.append(delta)

        grad_weights = []
        grad_biases = []

        for l in range(self.number_of_hidden_layers , -1, -1 ): # it is actually running from Output layer to the first hidden layer. Due to indexing convention, loop starts from number_of_hidden_layers and goes till 0
            grad_weight = deltas[0] @ self.activations[l].T
            grad_bias = deltas[0]

            
            info_passed_to_weights = self.weights[l].T @ deltas[0]   # information from present layer passed on to the weights connecting present and previous layers 
            if maths.any(maths.isinf(info_passed_to_weights)) :
                print("Infinite values encountered while passing information to weights ! Exploding gradient !!!")
                raise ValueError("Infinite values encountered while passing information to weights ! Exploding gradient !!!")
            
            if l > 0 : layer = self.hidden_layers[l-1]
            else : layer = self.activations[0]

            change_in_neurons = maths.matrix([self.calculate_derivative(float(i),self.hidden) for i in layer ]).reshape(-1,1)  # change of neurons of the previous layer

            
            delta =  maths.multiply(info_passed_to_weights , change_in_neurons ).reshape(-1,1)      # changes in loss wrt the previous layer's neurons
            if maths.any(maths.isnan(delta)) :
                print("NaN values encountered in delta")
                raise ValueError("NaN values encountered in delta")
            deltas.insert(0,delta)

            grad_weights.insert(0,grad_weight)
            grad_biases.insert(0,grad_bias)
    
        return grad_weights, grad_biases


    def has_converged(self, prev_weights, prev_bias) :
        self._converged = (maths.linalg.norm(maths.matrix(self.weights[0]) - maths.matrix(prev_weights[0])) < self.epsilon) and (maths.linalg.norm(maths.matrix(self.bias[0]) - maths.matrix(prev_bias[0])) < self.epsilon)


    def update_weights(self):
        
        prev_weights = [w + 1 for w in self.weights]
        prev_bias = [b + 1 for b in self.bias]
        
        self._converged = False

        t = 0 
        while not self._converged:
            prev_weights = self.weights.copy()
            prev_bias = self.bias.copy()

            number_of_batches = len(self.df_data) // self.batch_size
            t = t+1 

            for M in range(number_of_batches) :
                batch_grad_weights = []
                batch_grad_bias = []
                for m in range(self.batch_size) :
                    if min(M*self.batch_size + m , self.N) != self.N :
                        x = maths.matrix(self.df_data.iloc[M*self.batch_size + m][:-1]).reshape(-1,1)
                        y = self.df_data.iloc[M*self.batch_size + m][-1]
                        self.forward_propagation(x)
                        grad_weights, grad_bias = self.backward_propagation(x,y)
                        batch_grad_weights.append(grad_weights)
                        batch_grad_bias.append(grad_bias)

                self.sum_of_grad_weights = batch_grad_weights[0].copy()
                self.sum_of_grad_bias = batch_grad_bias[0].copy()


                for i in range(1,len(batch_grad_bias)):
                    for j in range(len(batch_grad_bias[i])):
                        self.sum_of_grad_bias[j] = self.sum_of_grad_bias[j] + batch_grad_bias[i][j]

                for i in range(1,len(batch_grad_weights)):
                    for j in range(len(batch_grad_weights[i])):
                        self.sum_of_grad_weights[j] = self.sum_of_grad_weights[j] + batch_grad_weights[i][j]

                if self.optimiser == 'momentum' :
                    for l in range(self.number_of_hidden_layers + 1):
                        self.momentum_update_weights[l] = self.gamma * self.momentum_update_weights[l] + float(self.alpha/self.batch_size) * self.sum_of_grad_weights[l]
                        self.momentum_update_bias[l] = self.gamma * self.momentum_update_bias[l] + float(self.alpha/self.batch_size) * self.sum_of_grad_bias[l]
                    
                    for l in range(self.number_of_hidden_layers + 1) :
                        self.weights[l] = self.weights[l] - self.momentum_update_weights[l]
                        self.bias[l] = self.bias[l] - self.momentum_update_bias[l]
                

                elif self.optimiser == 'adagrad' or self.optimiser == 'rmsprop':
                    self.gamma = int(self.optimiser == 'rmsprop') * self.gamma

                    for l in range(self.number_of_hidden_layers + 1) :
                        squared_weights = 0
                        squared_bias = 0
                        squared_weights = (self.gamma * squared_weights) + (1 - self.gamma) * maths.diagonal(maths.matmul( self.sum_of_grad_weights[l] , self.sum_of_grad_weights[l].T )).reshape(-1,1)
                        squared_bias = (self.gamma * squared_bias) + (1 - self.gamma) * maths.diagonal(maths.matmul( self.sum_of_grad_bias[l] , self.sum_of_grad_bias[l].T )).reshape(-1,1)

                        squared_weights = (squared_weights + 1e-8) ** 0.5
                        squared_bias = (squared_bias + 1e-8) ** 0.5

                        
                        self.weights[l] = self.weights[l] - self.alpha * maths.multiply(squared_weights , self.sum_of_grad_weights[l])
                        self.bias[l] = self.bias[l] - self.alpha * maths.multiply(squared_bias , self.sum_of_grad_bias[l])


                elif self.optimiser == 'adam' :

                    beta1 = 0.4
                    beta2 = 0.3 
                    
                    for l in range(self.number_of_hidden_layers + 1) :
                        m_weights = 0
                        m_bias = 0

                        m_weights = ( beta1 * m_weights + (1-beta1) * self.sum_of_grad_weights[l] ) / ( 1 - beta1 ** t )
                        m_bias = beta2 * m_bias + (1-beta2) * self.sum_of_grad_bias[l] / ( 1 - beta2 ** t )

                        squared_weights = 0
                        squared_bias = 0
                        squared_weights = ( (beta1 * squared_weights) + (1 - beta1) * maths.diagonal(maths.matmul( self.sum_of_grad_weights[l] , self.sum_of_grad_weights[l].T )).reshape(-1,1) ) / ( 1 - beta1 ** t)
                        squared_bias = ( (beta2 * squared_bias) + (1 - beta2) * maths.diagonal(maths.matmul( self.sum_of_grad_bias[l] , self.sum_of_grad_bias[l].T )).reshape(-1,1) ) / ( 1 - beta2 ** t)

                        squared_weights = (squared_weights + 1e-8) ** 0.5
                        squared_bias = (squared_bias + 1e-8) ** 0.5

                        
                        self.weights[l] = self.weights[l] - self.alpha * maths.multiply(squared_weights , self.sum_of_grad_weights[l])
                        self.bias[l] = self.bias[l] - self.alpha * maths.multiply(squared_bias , self.sum_of_grad_bias[l])

                else : 
                    raise ValueError("Choose optimiser among Momentum, AdaGrad, RMSProp or Adam")
                        
            
            self.has_converged(prev_weights, prev_bias)