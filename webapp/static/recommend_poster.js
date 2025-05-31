const imageInput = document.getElementById("imageInput");
const imagePreview = document.getElementById("imagePreview");
const recommendationImages = document.getElementById("recommendationImages");

imageInput.addEventListener("change", function (event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
            imagePreview.src = e.target.result;
            imagePreview.style.display = "block";
        };
        reader.readAsDataURL(file);

        document.getElementById("recommendationImages").style.display = "none";
    }
});

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
            document.getElementById("recommendationImages").style.display = "block";
            recommendationImages.innerHTML = "";
            data.recommendations.forEach(path => {
                const img = document.createElement("img");
                img.src = path;
                img.className = "recommendation-thumb";
                recommendationImages.appendChild(img);
            });
        })
        .catch(error => console.error("Erreur de recommandation :", error));
});
