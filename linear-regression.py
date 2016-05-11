import numpy as np

## define training data
#X = np.array([[1.0,2104.0,5.0,1.0,45.0],[1.0,1416.0,3.0,2.0,40.0],[1.0,1534.0,3.0,2.0,30.0],[1.0,852.0,2.0,1.0,36.0]])
X = np.array([[1.0,2104.0,5.0,1.0],[1.0,1416.0,3.0,2.0],[1.0,1534.0,3.0,2.0],[1.0,852.0,2.0,1.0]])
Y = np.array([460.0,232.0,315.0,178.0])

print(X)

## scaling
XT = X.T
for h in range(len(XT)):
    x = XT[h]
    mean = np.mean(x)
    max = x[np.where(np.amax(x) == x)[0][0]]
    min = x[np.where(np.amin(x) == x)[0][0]]
    r = (max - min) / 2
    if r != 0:
        XT[h] = (XT[h] - mean) / r

## gradient decent
m = len(X)
n = len(X[0])
th = np.zeros(n)

alpha = 0.01
change = 1.0
steps = 0

#while change > 0.00001:
for k in range(300000):

    change = 0.0
    steps += 1

    for i in range(m):
        x = X[i]
        y = Y[i]
        for j in range(n):
            theta = th[j]
            j_th = (1/(2*m)) * (np.dot(x,th.T)-y) * x[j]
            new_theta = theta - alpha * j_th
            change += new_theta - theta
            th[j] = new_theta

    print(steps,change)

# normalized equation
A = np.dot(X.T,X)
th_n = np.dot(np.dot(np.linalg.inv(A),X.T),Y)


print("Closed test in GD:",np.dot(X,th.T))
print("Closed test in NE:",np.dot(X,th_n.T))
print("Closed test Gold: ",Y)

print(th)
print(th_n)

