import numpy as np
import matplotlib.pyplot as plt
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_loss(y, y_pred):
    m = len(y)
    loss = -(1/m) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    return loss

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0
        losses = []        
        for i in range(self.num_iterations):
            # Compute predictions
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = sigmoid(linear_model)
            
            # Compute gradients
            dw = (1 / num_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / num_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Compute and record loss
            if i % 100 == 0:
                loss = compute_loss(y, y_pred)
                losses.append(loss)
                print(f"Iteration {i}, Loss: {loss}")
        
        # Optional: plot the loss
        plt.plot(range(0, self.num_iterations, 100), losses)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Loss over Iterations")
        plt.show()

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_model)
        return np.where(y_pred >= 0.5, 1, 0)

# Example usage
if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(42)
    num_samples = 100
    X = np.random.rand(num_samples, 2)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)
    
    # Split into training and testing sets
    train_size = int(0.8 * num_samples)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Train logistic regression model
    model = LogisticRegression(learning_rate=0.1, num_iterations=1000)
    model.fit(X_train, y_train)
    
    # Evaluate model
    predictions = model.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")
