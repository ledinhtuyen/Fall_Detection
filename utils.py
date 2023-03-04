import numpy as np

def softmax(x):
    x = x[0].numpy()
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x
    print("Fall Probality: ", y[1])
    
    if y[1] > 0.7:
        return "Fall"
    else:
        return "Up"