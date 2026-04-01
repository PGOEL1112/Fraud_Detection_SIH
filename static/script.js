// 🔐 AUTH STATE
let currentUser = null;
let safeCount = 0;
let fraudCount = 0;

let pieChart, barChart, confidenceChart;
let confidenceData = [];
let labels = [];

document.addEventListener("DOMContentLoaded", () => {
    // 🔐 AUTH STATE
    auth.onAuthStateChanged(async (user) => {

        const tbody = document.querySelector("#historyTable tbody");
        if (user) {
            currentUser = user;

            document.getElementById("loginBtn").style.display = "none";
            document.getElementById("logoutBtn").style.display = "inline-block";
            document.getElementById("userInfo").innerText = "👤 " + user.email;

            // 🔥 LOAD USER DATA FROM DB
            const res = await fetch("/api/my_data", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({ email: user.email })
            });

            const data = await res.json();

            tbody.innerHTML = ""; // Clear existing data

            if (data.status === "success") {
                data.data.forEach(item => {
                    const row = document.createElement("tr");
                    row.innerHTML = `
                        <td>${item.herb_type}</td>
                        <td>${item.quality_score}</td>
                        <td>${item.moisture_level}</td>
                        <td>${item.stock_before}</td>
                        <td>${item.stock_after}</td>
                        <td>${item.amount}</td>
                        <td>${item.result}</td>
                    `;
                    tbody.appendChild(row);
                });

                recalculateStats(); // 🔥 important
            }

        } else {
            currentUser = null;

            document.getElementById("loginBtn").style.display = "inline-block";
            document.getElementById("logoutBtn").style.display = "none";
            document.getElementById("userInfo").innerText = "";

            tbody.innerHTML = ""; // Clear table
            safeCount = 0;
            fraudCount = 0;
            updateUIStats();
        }
    });

    // 🔐 LOGIN
    document.getElementById("loginBtn").addEventListener("click", async () => {
        const provider = new firebase.auth.GoogleAuthProvider();
        await auth.signInWithPopup(provider);
    });

    // 🔐 LOGOUT
    document.getElementById("logoutBtn").addEventListener("click", () => {
        auth.signOut();
    });


    // ---------- Prediction ----------
    const predictForm = document.getElementById("predictForm");
    const predictMessage = document.getElementById("predictMessage");
    const predictText = document.getElementById("predictText");
    const nextPredictionBtn = document.getElementById("nextPredictionBtn");
    const historyTableBody = document.querySelector("#historyTable tbody");
    

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
                    
                    predictMessage.className = "prediction-card " +
                        (result.toLowerCase().includes("fraud") ? "fraud" : "safe");

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

                    recalculateStats(); 
                    confidenceData.push(data.confidence);
                    labels.push(`Tx ${labels.length + 1}`);

                    if(confidenceData.length > 10) {
                        confidenceData.shift();
                        labels.shift();
                    }

                    updateConfidenceChart();
                    predictForm.reset();

                } else {
                      alert("Error: " + data.result);
                    }
            })
            .catch(err => alert("Prediction error: " + err));
    });

    nextPredictionBtn.addEventListener("click", () => {
            predictMessage.style.display = "none";
        });


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

    // ---------- STATS ----------
function recalculateStats() {
    const rows = document.querySelectorAll("#historyTable tbody tr");

    safeCount = 0;
    fraudCount = 0;

    rows.forEach(row => {
        const text = row.cells[row.cells.length - 1].innerText.toLowerCase();
        if (text.includes("fraud")) fraudCount++;
        else safeCount++;
    });

    updateUIStats();
}

function updateUIStats() {
    const total = safeCount + fraudCount;

    document.getElementById("totalCount").textContent = total;
    document.getElementById("safeCount").textContent = safeCount;
    document.getElementById("fraudCount").textContent = fraudCount;

    const percent = total ? ((fraudCount / total) * 100).toFixed(1) : 0;
    document.getElementById("fraudPercent").textContent = percent + "%";


    if (safeCount === 0 && fraudCount === 0) {
        document.getElementById("totalCount").textContent = 0;
    }

    updatePieChart();
    updateBarChart();
}


// ---------- CHARTS ----------
function updatePieChart() {
    const ctx = document.getElementById("pieChart").getContext("2d");

    if (pieChart) pieChart.destroy();

    pieChart = new Chart(ctx, {
        type: "pie",
        data: {
            labels: ["Safe", "Fraud"],
            datasets: [{
                data: [safeCount, fraudCount],
                backgroundColor: ["#27ae60", "#c0392b"]
            }]
        }
    });
}

function updateBarChart() {
    const ctx = document.getElementById("barChart").getContext("2d");

    if (barChart) barChart.destroy();

    barChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels: ["Safe", "Fraud"],
            datasets: [{
                data: [safeCount, fraudCount],
                backgroundColor: ["#27ae60", "#c0392b"]
            }]
        }
    });
}

function updateConfidenceChart() {
    const ctx = document.getElementById("confidenceChart").getContext("2d");

    if (confidenceChart) confidenceChart.destroy();

    confidenceChart = new Chart(ctx, {
        type: "line",
        data: {
            labels: labels,
            datasets: [{
                label: "Confidence %",
                data: confidenceData,
                borderColor: "#2980b9",
                fill: false
            }]
        }
    });
}


// ---------- BULK CSV ----------
function uploadCSV() {
    const file = document.getElementById("bulkFile").files[0];

    if (!file) {
        alert("Upload CSV first");
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
            document.getElementById("bulkResult").innerText =
                `Fraud: ${data.fraud}, Safe: ${data.safe}`;
        });
}

