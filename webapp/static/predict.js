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

        document.getElementById("explanationContainer").style.display = "none";
        document.getElementById("limeImage").src = "";
        document.getElementById("gradcamImage").src = "";
        document.getElementById("smoothgradImage").src = "";
        predictionText.innerText = "";
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
    
            // Affiche le container
            document.getElementById("explanationContainer").style.display = "block";
    
            if (data.lime_base64) {
                const limeImg = document.getElementById("limeImage");
                limeImg.src = "data:image/png;base64," + data.lime_base64;
                limeImg.style.display = "block";
            }
            if (data.gradcam_base64) {
                const gradcamImg = document.getElementById("gradcamImage");
                gradcamImg.src = "data:image/png;base64," + data.gradcam_base64;
                gradcamImg.style.display = "block";
            }
            if (data.smoothgrad_base64) {
                const smoothgradImg = document.getElementById("smoothgradImage");
                smoothgradImg.src = "data:image/png;base64," + data.smoothgrad_base64;
                smoothgradImg.style.display = "block";
            }
        })
        
        .catch(error => console.error("Erreur de prédiction :", error));
});
