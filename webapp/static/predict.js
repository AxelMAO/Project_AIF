const imageInput = document.getElementById("imageInput");
const imagePreview = document.getElementById("imagePreview");
const predictionText = document.getElementById("predictionText");

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

document.getElementById("uploadForm").addEventListener("submit", function (event) {
    event.preventDefault();

    const file = imageInput.files[0];
    if (!file) {
        alert("Veuillez choisir une image.");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    fetch("/predict", {
        method: "POST",
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            predictionText.innerText = "Prédiction : " + data.prediction;
        })
        .catch(error => console.error("Erreur de prédiction :", error));
});
