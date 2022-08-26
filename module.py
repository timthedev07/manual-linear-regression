from math import fabs
EPOCHS = 50

def MSE(W, b, dataPoints):
    """
    `trainingData` is a list of tuples of numbers, e.g.:
        [
            (1, 3),
            (2.4, 7),
            (5, 19),
        ]
    """
    s = 0
    n = len(dataPoints)
    for x, y in dataPoints:
        s += (y - (W * x + b))**2
    return s / n
  
def GD(dataPoints, learningRate = 0.01, epochs = EPOCHS, logStep = 5, earlyStopping = True, earlyStoppingDelta = 0.001):
    W, b = 0, 0
    n = len(dataPoints)
    prevLoss = MSE(W, b, dataPoints)

    for epoch in range(1, epochs + 1):
        p = q = 0
        for x, y in dataPoints:
            yPred = W * x + b
            diff = (y - yPred)
            p += x * diff
            q += diff
        W -= (-2 / n) * learningRate * p
        b -= (-2 / n) * learningRate * q

        loss = MSE(W, b, dataPoints)
        if epoch % logStep == 0:
            print(f"Epoch {epoch} - loss: {loss}")
        
        lossDelta = fabs(loss - prevLoss)
        if lossDelta <= earlyStoppingDelta:
            print(f"Delta loss is below threshold {earlyStoppingDelta}; stopping.")
            return W, b
        prevLoss = loss
    return W, b
