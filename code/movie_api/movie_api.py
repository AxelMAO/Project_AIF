import argparse
import torch
import torchvision.transforms as transforms
import io
import os
import torchvision.models as models
import pandas as pd
import pickle
from flask import Flask, jsonify, request
from PIL import Image
from torchvision import datasets
from annoy import AnnoyIndex
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import DistilBertTokenizer, DistilBertModel



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

app = Flask(__name__)

parser = argparse.ArgumentParser()
# add an argument '--model_path'
parser.add_argument('--model_path', type=str, help='path to the model', default='./weights/movie_net.pth')
model_path = parser.parse_args().model_path

parser.add_argument('--feature_extraction_path', type=str, help='path to the feature extraction folder', default='./feature_extraction/')
feature_extraction_path = parser.parse_args().feature_extraction_path

model_Prediction = models.mobilenet_v3_small(pretrained=False)
model_Prediction.classifier = torch.nn.Linear(576, 10)

# Load the model for the first Part
#model.load_state_dict(torch.load(model_path))
model_Prediction.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
#model.eval()

model_Prediction.to(device)

transform = transforms.Compose(
        [transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        ])

genres_dict = {0 : 'action',1 :'animation', 2 : 'comedy', 3 : 'documentary', 
                4 : 'drama', 5 : 'fantasy', 6 : 'horror', 7 : 'romance', 
                8 : 'science Fiction',9 : 'thriller'}



#Preparation for the Part 2 for the recommendation system


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


#Preparation for the Part 3 for the recommendation system based on the plot of the movie

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



@app.route('/predict', methods=['POST'])
def predict():
    img_binary = request.data
    img_pil = Image.open(io.BytesIO(img_binary))

    # Transform the PIL image
    tensor = transform(img_pil).to(device)
    tensor = tensor.unsqueeze(0)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        outputs = model_Prediction(tensor)
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
        outputs = model_Prediction(batch_tensor.to(device))
        _, predictions = outputs.max(1)

    return jsonify({"predictions": predictions.tolist()})



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
    elif method == "bert":
        query_vector = get_embeddings_bert(text)
        results = search_overview_title(query_vector, bert_index, k=5)
    else:
        return jsonify({"error": "Unknown method"}), 400

    return jsonify({"recommendations": results})



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)