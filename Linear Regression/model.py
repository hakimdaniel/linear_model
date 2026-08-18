import pandas as pd

df = pd.read_csv("data.csv")

x = df["input"].tolist()
y = df["target"].tolist()

print("x:",x)
print("y:",y)

w = 0
b = 0
loss = 0
learning_rate = 0.01

print("Start Train model")

for e in range(100):
    print(f"\n(Epoch {e+1})=============")
    
    for i in range(len(x)):
        prediction = w*x[i]+b
        error = prediction - y[i]
        loss = error ** 2
        
        dw = 2 * error * x[i]
        db = 2 * error
        w = w - learning_rate * dw
        b = b - learning_rate * db
      
    print("Current w:",w)
    print("Current b",b)
    print("Current loss:",loss)

def predict(x):
    prediction = w * x + b
    return prediction

print("\n--------RESULT---------")
print("[ Trained Linear Regression Model ]")
print("Final weight:",w)
print("Final bias:",b)
print("Final loss:",loss)

print("\nTest model")
print("Input 2 :",predict(2))
print("Input 5 :",predict(5))
print("Input 10 :",predict(10))
print("Input 12 :",predict(12))
