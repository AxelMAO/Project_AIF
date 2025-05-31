import argparse
import torch
import torchvision.transforms as transforms
import io
import os
import base64
import cv2
import torchvision.models as models
import pandas as pd
import numpy as np
import pickle
from flask import Flask, jsonify, request
from PIL import Image
from torchvision import datasets
from annoy import AnnoyIndex
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import DistilBertTokenizer, DistilBertModel
from models import ClassifieurMovie, ClassifieurResNet18, ClassifieurResNet34
from lime import lime_image
from skimage.segmentation import mark_boundaries
from torchvision.transforms import ToPILImage

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

app = Flask(__name__)

parser = argparse.ArgumentParser()
# add an argument '--model_path'
parser.add_argument('--model_path', type=str, help='path to the model', default='./weights/movie_net.pth')
model_path = parser.parse_args().model_path

parser.add_argument('--feature_extraction_path', type=str, help='path to the feature extraction folder', default='./feature_extraction/')
feature_extraction_path = parser.parse_args().feature_extraction_path

#model_Prediction = models.mobilenet_v3_small(pretrained=False)
model_Prediction = models.mobilenet_v3_small(weights=None)
model_Prediction.classifier = torch.nn.Linear(576, 10)
model_Prediction = ClassifieurMovie()
# Load the model for the first Part
#model.load_state_dict(torch.load(model_path))
model_Prediction.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
#model_Prediction.eval()

model_Prediction.to(device)

transform = transforms.Compose(
        [transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        ])

genres_dict = {0 : 'action',1 :'animation', 2 : 'comedy', 3 : 'documentary', 
                4 : 'drama', 5 : 'fantasy', 6 : 'horror', 7 : 'romance', 
                8 : 'science Fiction',9 : 'thriller'}


##############################################################
#Preparation for the Part 2 for the recommendation system ####
##############################################################

df_path = pd.read_parquet(os.path.join(feature_extraction_path, 'images_paths.parquet'))
annoy_index = AnnoyIndex(576, 'angular')
annoy_index.load(os.path.join(feature_extraction_path, 'rec_imdb.ann'))


#functions

def search(query_vector, k=5):
    indices = annoy_index.get_nns_by_vector(query_vector, k)
    paths = df_path['path'].iloc[indices]
    #paths = df_path['path'].iloc[indices].apply(lambda p: os.path.join(feature_extraction_path, p)).tolist()
    return paths


#Create the model for feature extraction
mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
modelFeatureExtraction  = torch.nn.Sequential(
    mobilenet.features,
    mobilenet.avgpool,
    torch.nn.Flatten(),
).cpu()

######################################################################################################
###Preparation for the Part 3 for the recommendation system based on the plot of the movie ###########
######################################################################################################

bert_index = AnnoyIndex(768, 'angular')
bert_index.load(os.path.join(feature_extraction_path, 'bert_index.ann'))
bagOfWords_index = AnnoyIndex(5000, 'angular')
bagOfWords_index.load(os.path.join(feature_extraction_path, 'tfidf_index5000.ann'))

bagOfWords_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

with open(os.path.join(feature_extraction_path, 'tfidf_vectorizer5000.pkl'), 'rb') as f:
    bagOfWords_vectorizer = pickle.load(f)


df_movies_TO = pd.read_csv(os.path.join(feature_extraction_path, 'movies_overview_titles.csv'))



tokenizer_bert = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model_bert = DistilBertModel.from_pretrained('distilbert-base-uncased')

def get_embeddings_bert(text):
    inputs = tokenizer_bert(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model_bert(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()



def search_overview_title(query_vector, annotated_index, k=5):
    indices = annotated_index.get_nns_by_vector(query_vector, k)
    titles_and_overviews = df_movies_TO.iloc[indices][['title', 'overview']]. to_dict(orient='records')
    return titles_and_overviews


#############################################
#############  PART4 ########################
#############################################

def image_array_to_base64(arr):
    """Transforme un tableau d'image numpy (RGB ou grayscale) en image PNG encodée base64."""
    arr_uint8 = (255 * (arr - arr.min()) / (arr.max() - arr.min())).astype(np.uint8)
    if arr.ndim == 2:
        img = Image.fromarray(arr_uint8, mode='L')
    else:
        img = Image.fromarray(arr_uint8)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')

class HookFeatures():
    def __init__(self, module):
        self.feature_hook = module.register_forward_hook(self.feature_hook_fn)
        self.features = None
        self.gradients = None

    def feature_hook_fn(self, module, input, output):
        self.features = output.clone().detach()
        self.gradient_hook = output.register_hook(self.gradient_hook_fn)

    def gradient_hook_fn(self, grad):
        self.gradients = grad

    def close(self):
        self.feature_hook.remove()
        self.gradient_hook.remove()

def generate_gradcam(model, input_tensor, np_img):
    # Hook sur la dernière couche convolutive de MobileNetV3
    hook = HookFeatures(model.features[12])

    input_tensor = input_tensor
    input_tensor.requires_grad = True  # Nécessaire pour le calcul des gradients
    input_tensor = input_tensor.to(device)
    # Forward pass
    output = model(input_tensor)
    pred_idx = output.argmax().item()
    output_max = output[0, pred_idx]
    output_max.backward()

    gradients = hook.gradients        # [B, C, H, W]
    activations = hook.features       # [B, C, H, W]
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])  # [C]

    # Pondérer chaque canal des activations
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]

    # Moyenne sur les canaux
    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = np.maximum(heatmap.detach().cpu(), 0)
    heatmap /= torch.max(heatmap)

    # Redimensionner et convertir en image couleur
    heatmap = cv2.resize(heatmap.numpy(), (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_RAINBOW)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB) / 255.0

    # Superposition sur l’image d’origine (normalisée entre 0-1)
    superposed_img = np.clip(heatmap_color * 0.4 + np_img, 0, 1)

    # Encode en base64
    img = Image.fromarray((superposed_img * 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    gradcam_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    hook.close()
    return gradcam_b64

def predict_and_explain(img_pil):
    tensor = transform(img_pil.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model_Prediction(tensor)
        prediction = output.argmax(dim=1).item()

    np_img = tensor.squeeze().cpu().numpy().transpose(1, 2, 0)

    # === LIME ===
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        np_img,
        lambda x: model_Prediction(torch.tensor(x.transpose(0, 3, 1, 2)).float().to(device)).softmax(1).detach().cpu().numpy(),
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )
    lime_img, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False)
    lime_vis = mark_boundaries(lime_img, mask)
    lime_base64 = image_array_to_base64(lime_vis)

    # === SmoothGrad ===
    tensor.requires_grad_()
    model_Prediction.zero_grad()
    output = model_Prediction(tensor)
    output[0, prediction].backward()
    gradients = tensor.grad.data.abs().squeeze().cpu().numpy().mean(axis=0)
    smooth_base64 = image_array_to_base64(gradients)

    gradcam_base64 = generate_gradcam(model_Prediction, tensor, np_img)

    return prediction, lime_base64, smooth_base64, gradcam_base64

@app.route('/predict', methods=['POST'])
def predict():
    img_binary = request.data
    img_pil = Image.open(io.BytesIO(img_binary))

    # Transform the PIL image
    #tensor = transform(img_pil).to(device)
    #tensor = tensor.unsqueeze(0)  # Add batch dimension

    # Make prediction
    #with torch.no_grad():
    #    outputs = model_Prediction(tensor)
    #    predicted = torch.argmax(outputs, dim=1).item()

    #key = int(predicted[0])

    prediction, lime_b64, smooth_b64, gradcam_b64 = predict_and_explain(img_pil)

    #return jsonify({"prediction": genres_dict[predicted]})
    return jsonify({
        "prediction": genres_dict[prediction],
        "lime_base64": lime_b64,
        "smoothgrad_base64": smooth_b64,
        "gradcam_base64": gradcam_b64
    })
    
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
        outputs = model_Prediction(batch_tensor.to(device))
        predictions = torch.max(outputs, dim=1).cpu()

    genres = [genres_dict[pred.item()] for pred in predictions] 
    
    return jsonify({"predictions": genres})



@app.route('/recommend_poster', methods=['POST'])
def recommend_poster():


    #img_binary = request.data
    #img_pil = Image.open(io.BytesIO(img_binary)).convert("RGB")
    
    #tensor = transform(img_pil).unsqueeze(0)  # Batch dim
    
    img_binary = request.data
    img_pil = Image.open(io.BytesIO(img_binary))

    # Transform the PIL image
    tensor = transform(img_pil).to(device)
    tensor = tensor.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        features = modelFeatureExtraction(tensor).squeeze().numpy()
    
    paths = search(features, k=5).tolist()
    

    #print(paths)
    
    return jsonify({"recommendations": paths})


@app.route('/recommend_plot_movie', methods=['POST'])
def recommend_plot_movie():
    # Get the plot from the request
    data = request.get_json()
    text = data.get("description", "")
    method = data.get("method", "Bert")  # default to Bert

    if not text:
        return jsonify({"error": "No description provided"}), 400

    if method == "Bag of Words":
        query_vector = bagOfWords_vectorizer.transform([text]).toarray()[0]
        results = search_overview_title(query_vector, bagOfWords_index, k=5)
    elif method == "Bert":
        query_vector = get_embeddings_bert(text)
        results = search_overview_title(query_vector, bert_index, k=5)
    else:
        return jsonify({"error": "Unknown method"}), 400

    return jsonify({"recommendations": results})



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)