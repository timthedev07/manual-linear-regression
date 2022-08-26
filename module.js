const EPOCHS = 50;

const MSE = (W, b, dataPoints) => {
  let s = 0;
  const n = dataPoints.length;

  for (let i = 0; i < n; i++) {
    let [x, y] = dataPoints[i];
    s += (y - (W * x + b))**2;
  }

  return s / n;
}
  
const GD = (dataPoints, learningRate = 0.01, epochs = EPOCHS, logStep = 5, earlyStopping = true, earlyStoppingDelta = 0.001) => {
  let W = 0;
  let b = 0;

  const n = dataPoints.length;
  let prevLoss = MSE(W, b, dataPoints);

  for (let epoch = 1; epoch <= EPOCHS, epoch++;) {
    let p = 0;
    let q = 0;

    for (let i = 0; i < n; i++) {
      let [x, y] = dataPoints[i];
      const yPred = W * x + b;
      const diff = (y - yPred);
      p += x * diff;
      q += diff;
    }
  
    W -= (-2 / n) * learningRate * p
    b -= (-2 / n) * learningRate * q

    let loss = MSE(W, b, dataPoints);
  
    if (epoch % logStep == 0) {
        console.log(`Epoch ${epoch} - loss: ${loss}`)
    }

    let lossDelta = Math.abs(loss - prevLoss)
    if (lossDelta <= earlyStoppingDelta) {
        console.log(`Delta loss is below threshold ${earlyStoppingDelta}; stopping.`)
        return W, b
    }
    
    prevLoss = loss;
  }
  return W, b
}

function predict() {

}
