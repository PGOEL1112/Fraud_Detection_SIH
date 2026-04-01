// 🔐 AUTH STATE
let currentUser = null;

// Listen auth state
document.addEventListener("DOMContentLoaded", () => {

    auth.onAuthStateChanged((user) => {
        if (user) {
            currentUser = user;
            document.getElementById("loginBtn").style.display = "none";
            document.getElementById("logoutBtn").style.display = "inline-block";
            document.getElementById("userInfo").innerText = "👤 " + user.email;
        } else {
            currentUser = null;
            document.getElementById("loginBtn").style.display = "inline-block";
            document.getElementById("logoutBtn").style.display = "none";
            document.getElementById("userInfo").innerText = "";
        }
    });

});


document.addEventListener("DOMContentLoaded", () => {

    document.getElementById("loginBtn").addEventListener("click", async () => {
        const provider = new firebase.auth.GoogleAuthProvider();
        try {
            await auth.signInWithPopup(provider);
        } catch (err) {
            alert(err.message);
        }
    });

    document.getElementById("logoutBtn").addEventListener("click", () => {
        auth.signOut();
    });

});

function recalculateStats() {
    const rows = document.querySelectorAll("#historyTable tbody tr");

    safeCount = 0;
    fraudCount = 0;

    rows.forEach(row => {
        const resultText = row.cells[6].innerText.toLowerCase();

        if (resultText.includes("fraud")) {
            fraudCount++;
        } else {
            safeCount++;
        }
    });

    const total = safeCount + fraudCount;

    document.getElementById("totalCount").textContent = total;
    document.getElementById("safeCount").textContent = safeCount;
    document.getElementById("fraudCount").textContent = fraudCount;

    const percent = total > 0 ? ((fraudCount / total) * 100).toFixed(1) : 0;
    document.getElementById("fraudPercent").textContent = percent + "%";

    updatePieChart();
    updateBarChart();
}

document.addEventListener("DOMContentLoaded", function () {

    // ---------- Prediction ----------
    const predictForm = document.getElementById("predictForm");
    const predictMessage = document.getElementById("predictMessage");
    const predictText = document.getElementById("predictText");
    const nextPredictionBtn = document.getElementById("nextPredictionBtn");
    const historyTableBody = document.querySelector("#historyTable tbody");
    const savedHistory = localStorage.getItem("historyTable");
    if (savedHistory) {
        historyTableBody.innerHTML = savedHistory;
        recalculateStats();
    }
    let safeCount = 0;
    let fraudCount = 0;
    let pieChart;
    let barChart;
    let confidenceChart;
    let confidenceData = [];
    let labels = [];

    predictForm?.addEventListener("submit", function (e) {
        // ❌ Block if not logged in
        if (!currentUser) {
            alert("⚠ Please login first!");
            return;
        }
        e.preventDefault();
        const formData = new FormData(predictForm);

        const jsonData = {
            herb_type: formData.get("herb_type"),
            quality_score: formData.get("quality_score"),
            moisture_level: formData.get("moisture_level"),
            stock_before: formData.get("stock_before"),
            stock_after: formData.get("stock_after"),
            amount: formData.get("amount"),
            email: currentUser.email,
        };

        fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(jsonData)
        })
            .then(res => res.json())
            .then(data => {
                if (data.status === "success") {
                    const result = data.result;

                    // Update counts
                    if (result.toLowerCase().includes("fraud")) fraudCount++;
                    else safeCount++;

                    const total = safeCount + fraudCount;
                    document.getElementById("totalCount").textContent = total;
                    document.getElementById("safeCount").textContent = safeCount;
                    document.getElementById("fraudCount").textContent = fraudCount;

                    const percent = total > 0 ? ((fraudCount / total) * 100).toFixed(1) : 0;
                    document.getElementById("fraudPercent").textContent = percent + "%";

                    // ---------- Confidence tracking ----------
                    confidenceData.push(data.confidence);
                    labels.push("T" + (labels.length + 1));

                    // limit size (last 10)
                    if (confidenceData.length > 10) {
                        confidenceData.shift();
                        labels.shift();
                    }

                    // Show result with reasons
                    predictText.innerHTML = `
                    ${result} <br>
                    🔥 Confidence: ${data.confidence}% <br>
                    ⚡ Risk Level: ${data.risk_level} <br>
                    🤖 RF: ${data.rf_prediction} | XGB: ${data.xgb_prediction}
                `;

                    if (data.reasons && data.reasons.length > 0) {
                        predictText.innerHTML += `<br>⚠ Reason: ${data.reasons.join(", ")}`;
                    }
                    predictMessage.style.display = "block";
                    predictMessage.classList.remove("safe", "fraud");

                    if (result.toLowerCase().includes("fraud")) {
                        predictMessage.classList.add("fraud");
                    } else {
                        predictMessage.classList.add("safe");
                    }

                    updatePieChart();                 
                    updateBarChart();
                    updateConfidenceChart();

                    // Add to history table
                    const row = document.createElement("tr");
                    row.innerHTML = `
                    <td>${formData.get("herb_type")}</td>
                    <td>${formData.get("quality_score")}</td>
                    <td>${formData.get("moisture_level")}</td>
                    <td>${formData.get("stock_before")}</td>
                    <td>${formData.get("stock_after")}</td>
                    <td>${formData.get("amount")}</td>
                    <td>${result}${data.reasons && data.reasons.length > 0 ? "<br><small>⚠ " + data.reasons.join(", ") + "</small>" : ""}</td>
                `;
                    historyTableBody.prepend(row);
                    localStorage.setItem("historyTable", historyTableBody.innerHTML);
                    Array.from(predictForm.elements).forEach(el => el.disabled = true);

                } else {
                    // Show validation errors nicely
                    if (data.details) {
                        predictText.textContent = "❌ Invalid Input: " + data.details.join(", ");
                        predictMessage.style.display = "block";
                    } else {
                        alert("Error: " + data.result);
                    }
                }
            })
            .catch(err => alert("Prediction error: " + err));
    });

    nextPredictionBtn?.addEventListener("click", function () {
        predictMessage.style.display = "none";
        predictText.textContent = "";
        predictForm.reset();
        Array.from(predictForm.elements).forEach(el => el.disabled = false);
    });

    // ---------- Pie Chart ----------
    function updatePieChart() {
        const ctx = document.getElementById('pieChart').getContext('2d');
        const data = {
            labels: ['Safe', 'Fraud'],
            datasets: [{
                label: 'Transactions',
                data: [safeCount, fraudCount],
                backgroundColor: ['#27ae60', '#c0392b'],
            }]
        };
        const config = { type: 'pie', data, options: { responsive: true, plugins: { legend: { position: 'bottom' } } } };
        if (pieChart) pieChart.destroy();
        pieChart = new Chart(ctx, config);
    }

    window.uploadCSV = function () {
        const fileInput = document.getElementById("bulkFile");
        const file = fileInput.files[0];

        if (!file) {
            alert("Please select CSV file");
            return;
        }

        const formData = new FormData();
        formData.append("file", file);

        fetch("/api/bulk_predict", {
            method: "POST",
            body: formData
        })
            .then(res => res.json())
            .then(data => {
                if (data.status === "success") {
                    document.getElementById("bulkResult").innerText =
                        `Fraud: ${data.fraud}, Safe: ${data.safe}`;
                }
            });
    }

    function updateBarChart() {
        const ctx = document.getElementById('barChart').getContext('2d');

        if (barChart) barChart.destroy();

        barChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Safe', 'Fraud'],
                datasets: [{
                    label: 'Transactions',
                    data: [safeCount, fraudCount],
                    backgroundColor: ['#27ae60', '#c0392b']
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { display: false }
                }
            }
        });
    }

    function updateConfidenceChart() {
        const ctx = document.getElementById('confidenceChart').getContext('2d');

        if (confidenceChart) confidenceChart.destroy();

        confidenceChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Confidence %',
                    data: confidenceData,
                    borderColor: '#2980b9',
                    fill: false,
                    tension: 0.3
                }]
            },
            options: {
                responsive: true
            }
        });
    }

    // ---------- Retrain ----------
    const retrainForm = document.getElementById("retrainForm");
    const retrainMessage = document.getElementById("retrainMessage");

    retrainForm?.addEventListener("submit", function (e) {
        e.preventDefault();
        const formData = new FormData(retrainForm);

        fetch("/retrain", { method: "POST", body: formData })
            .then(res => res.json())
            .then(data => {
                retrainMessage.textContent = data.message || "No response";
                retrainMessage.style.color = data.status === 'success' ? 'green' : 'red';
            })
            .catch(err => {
                retrainMessage.textContent = "❌ Error: " + err;
                retrainMessage.style.color = 'red';
            });
    });

});
