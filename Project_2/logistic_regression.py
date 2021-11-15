import numpy as np

class SGDRegressor:
    def __init__(self, eta0, max_iter=200, penalty = None):
        self.max_iter = max_iter
        self.penalty = penalty
        self.eta0 = eta0

    def SGD(self, X, y, G):        
        n_epochs = 500
        t0, t1 = 5, 50
        
        def learning_schedule(t):
            return t0/(t + t1)

        self.beta = np.random.random([X.shape[1], 1])
        m = X.shape[0]

        for epoch in range(n_epochs):
            for i in range(m):
                random_index = np.random.randint(m) #random number
                #compte gradient at different spots..!
                xi = X[random_index:random_index+1] #1 row from x
                yi = y[random_index:random_index+1] #corresponding y
                gradients = G(xi, yi, self.beta) #compute gradient at this point
                eta = learning_schedule(epoch*m + i) #eta decreases over time
                self.beta = self.beta - eta*gradients #do one step

    def GD(self, X, y, G):
        """
        Used to test SGD algorithm
        """

        self.beta = np.random.random([X.shape[1], 1])
        
        eta = 0.01
        Niterations = 1000
        for i in range(Niterations):
            grad = G(X, y, self.beta)
            self.beta -= eta*grad

class LogisticRegression(SGDRegressor):
    def __init__ (self, eta0=0.1, max_iter=200, lamb = 0):
            SGDRegressor.__init__(self, eta0, max_iter)
            self.lamb = lamb # regularization parameter

    def p_sigmoid(self, xb):
        return np.exp(xb)/(1 + np.exp(xb)) 

    def fit(self, x, y):

        X = np.c_[np.ones((x.shape[0],1)), x] #designmatrix

        
        def G(X, y, beta):
            return - X.T @ (y - self.p_sigmoid(X @ beta)) + self.lamb*beta
         
        self.SGD(X, y, G)

    def predict(self, x):

        X = np.c_[np.ones((x.shape[0],1)), x] #designmatrix
        return (self.p_sigmoid(X @ self.beta) > 0.5).astype(int)




