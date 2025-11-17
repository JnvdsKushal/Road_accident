document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("predictionForm");
    const resultDiv = document.getElementById("result");

    if (form) {
        form.addEventListener("submit", async (event) => {
            event.preventDefault();

            const formData = new FormData(form);
            const data = {};

            formData.forEach((value, key) => {
                data[key] = value;
            });

            try {
                const response = await fetch("/predict_json/", {
                    method: "POST",
                    headers: {
                        "X-CSRFToken": getCSRFToken(),
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify(data)
                });

                if (!response.ok) {
                    throw new Error("Network error");
                }

                const result = await response.json();
                showPrediction(result);
            } catch (error) {
                console.error("Error:", error);
                resultDiv.innerHTML = `<p class="text-danger">Error predicting risk.</p>`;
            }
        });
    }
});

function showPrediction(result) {
    const resultDiv = document.getElementById("result");
    let message = "";

    if (result.prediction === 1) {
        message = "⚠️ High Risk! Drive cautiously.";
    } else if (result.prediction === 2) {
        message = "⚠️ Moderate Risk. Stay alert.";
    } else {
        message = "✅ Low Risk. Safe to drive.";
    }

    resultDiv.innerHTML = `
        <div class="alert alert-info">
            <h4>${message}</h4>
            <p>Confidence: ${result.confidence ? result.confidence + "%" : "N/A"}</p>
        </div>
    `;
}

function getCSRFToken() {
    const name = "csrftoken";
    const cookies = document.cookie.split(";");
    for (let cookie of cookies) {
        cookie = cookie.trim();
        if (cookie.startsWith(name + "=")) {
            return cookie.substring(name.length + 1);
        }
    }
    return "";
}
