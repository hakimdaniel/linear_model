import csv

x1 = []
x2 = []
x3 = []
y = []

with open("data.csv","r") as file:
    reader = csv.DictReader(file)

    for row in reader:
        x1.append(float(row["total_mana"]))
        x2.append(float(row["user_age"]))
        x3.append(float(row["experience_days"]))
        y.append(float(row["magic_output"]))

w1 = 0
w2 = 0
w3 = 0
b = 0
loss = 0
learning_rate = 0.00001

print("Start Train model")

for e in range(100):
    print(f"\n(Epoch {e+1})=============")
    
    for i in range(len(y)):
        prediction = (w1*x1[i])+(w2*x2[i])+(w3*x3[i])+b
        error = prediction - y[i]
        loss = error ** 2
        
        dw1 = 2 * error * x1[i]
        dw2 = 2 * error * x2[i]
        dw3 = 2 * error * x3[i]
        db = 2 * error
        
        w1 -= learning_rate * dw1
        w2 -= learning_rate * dw2
        w3 -= learning_rate * dw3
        b = b - learning_rate * db

def predict(x1,x2,x3):
    prediction = (w1 * x1) + (w2 * x2) + (w3 * x3)+ b
    return prediction

print("\n--------RESULT---------")
print("[ Trained Multiple Linear Regression Model ]")
print("Final weight 1:",w1)
print("Final weight 2:",w2)
print("Final weight 3:",w3)
print("Final bias:",b)
print("Final loss:",loss)

print("\nTest model")
print("Mana | Age | Experience")
print("Input 50 12 10:",predict(50,12,10))
print("Input 100 17 30:",predict(100,17,30))
print("Input 150 20 30:",predict(150,20,30))
print("Input 330 80 380:",predict(330,80,380))
