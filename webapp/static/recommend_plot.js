const plotInput = document.getElementById("plotInput");
const plotRecommendations = document.getElementById("plotRecommendations");

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
        body: JSON.stringify({ description: plot, method: method })
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
        .catch(error => console.error("Erreur de recommandation :", error));
});
