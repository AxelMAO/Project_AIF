const imageInput = document.getElementById("imageInput");
const imagePreview = document.getElementById("imagePreview");
const predictionText = document.getElementById("predictionText");
const recommendationImages = document.getElementById("recommendationImages");

const plotInput = document.getElementById("plotInput");
const plotRecommendations = document.getElementById("plotRecommendations");

// Prévisualisation de l'image
imageInput.addEventListener("change", function (event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
            imagePreview.src = e.target.result;
            imagePreview.style.display = "block";
        };
        reader.readAsDataURL(file);
    }
});

// Prédiction de genre
document.getElementById("uploadForm").addEventListener("submit", function (event) {
    event.preventDefault();

    const file = imageInput.files[0];
    if (!file) {
        alert("Veuillez choisir une image.");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    fetch("/", {
        method: "POST",
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            predictionText.innerText = "Prédiction : " + data.prediction;
        })
        .catch(error => console.error("Erreur de prédiction :", error));
});

// Recommandation par affiche
document.getElementById("recommendButton").addEventListener("click", function () {
    const file = imageInput.files[0];
    if (!file) {
        alert("Veuillez choisir une image.");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    fetch("/recommend_poster", {
        method: "POST",
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            recommendationImages.innerHTML = "";
            data.recommendations.forEach(path => {
                const img = document.createElement("img");
                img.src = path;
                img.className = "recommendation-thumb";
                recommendationImages.appendChild(img);
            });
        })
        .catch(error => console.error("Erreur de recommandation (poster) :", error));
});

// Recommandation par résumé
document.getElementById("recommendPlotButton").addEventListener("click", function () {
    const plot = plotInput.value.trim();
    const method = document.getElementById("embeddingMethod").value;

    if (!plot) {
        alert("Veuillez saisir un résumé de film.");
        return;
    }

    fetch("/recommend_plot", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ description: plot,
            method : method
         })
    })
        .then(response => response.json())
        .then(data => {
            plotRecommendations.innerHTML = "";

            data.recommendations.forEach(film => {
                const div = document.createElement("div");
                div.className = "plot-recommendation-item";

                const title = document.createElement("h3");
                title.innerText = film.title;

                const overview = document.createElement("p");
                overview.innerText = film.overview;

                div.appendChild(title);
                div.appendChild(overview);
                plotRecommendations.appendChild(div);
            });
        })
        .catch(error => console.error("Erreur de recommandation (plot) :", error));
});
