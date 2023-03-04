from flask import Flask, render_template, request, jsonify
import torch
from torch import nn
from torch.nn import Sequential
import numpy as np

def softmax(x):
    x = x[0]
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x
    if y[1] > 0.7:
        return "Fall"
    else:
        return "Up"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc = Sequential(
            nn.Linear(99, 10),
            nn.ReLU(),
            nn.Linear(10,2)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

model=MLP()
model.load_state_dict(torch.load('../checkpoint/fall_vs_up_dict.pth', map_location=device))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/fall_detection', methods=['POST'])
def fall_detection():
    request_data = request.get_json()
    keypoints = torch.tensor(request_data['keypoints'], dtype=torch.float32)
    output = model(keypoints.unsqueeze(0))
    output = output.detach().numpy()
    return softmax(output)

if __name__ == '__main__':
    app.run(debug=False)
