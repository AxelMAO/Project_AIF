import argparse
import torch
import torchvision.transforms as transforms
from flask import Flask, jsonify, request
from PIL import Image
import io
import torchvision.models as models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

app = Flask(__name__)

parser = argparse.ArgumentParser()
# add an argument '--model_path'
parser.add_argument('--model_path', type=str, help='path to the model', default='./weights/movie_net.pth')
model_path = parser.parse_args().model_path

model = models.mobilenet_v3_small(pretrained=False)
model.classifier = torch.nn.Linear(576, 10)

# Load the model
model.load_state_dict(torch.load(model_path))
#model.eval()

model.to(device)

transform = transforms.Compose(
        [transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        ])

genres_dict = {0 : 'action',1 :'animation', 2 : 'comedy', 3 : 'documentary', 
                4 : 'drama', 5 : 'fantasy', 6 : 'horror', 7 : 'romance', 
                8 : 'science Fiction',9 : 'thriller'}

@app.route('/predict', methods=['POST'])
def predict():
    img_binary = request.data
    img_pil = Image.open(io.BytesIO(img_binary))

    # Transform the PIL image
    tensor = transform(img_pil).to(device)
    tensor = tensor.unsqueeze(0)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        outputs = model(tensor)
        predicted = outputs.max(1)

    key = int(predicted[0])

    return jsonify({"prediction": genres_dict[key]})

    
@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    # Get the image data from the request
    images_binary = request.files.getlist("images[]")

    tensors = []

    for img_binary in images_binary:
        img_pil = Image.open(img_binary.stream)
        tensor = transform(img_pil)
        tensors.append(tensor)

    # Stack tensors to form a batch tensor
    batch_tensor = torch.stack(tensors, dim=0)

    # Make prediction
    with torch.no_grad():
        outputs = model(batch_tensor.to(device))
        _, predictions = outputs.max(1)

    return jsonify({"predictions": predictions.tolist()})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)