document.addEventListener("DOMContentLoaded", function () {

    // ---------- Prediction ----------
    const predictForm = document.getElementById("predictForm");
    const predictMessage = document.getElementById("predictMessage");
    const predictText = document.getElementById("predictText");
    const nextPredictionBtn = document.getElementById("nextPredictionBtn");
    const historyTableBody = document.querySelector("#historyTable tbody");

    let safeCount = 0;
    let fraudCount = 0;
    let pieChart;

    predictForm?.addEventListener("submit", function (e) {
        e.preventDefault();
        const formData = new FormData(predictForm);

        fetch("/predict", { method: "POST", body: formData })
            .then(res => res.json())
            .then(data => {
                const result = data.result;

                // Show result
                predictText.textContent = result;
                predictMessage.style.display = "block";

                // Update counts
                if (result.toLowerCase().includes("fraud")) fraudCount++;
                else safeCount++;

                updatePieChart();

                // Add to history table
                const row = document.createElement("tr");
                row.innerHTML = `
                    <td>${formData.get("herb_type")}</td>
                    <td>${formData.get("quality_score")}</td>
                    <td>${formData.get("moisture_level")}</td>
                    <td>${formData.get("stock_before")}</td>
                    <td>${formData.get("stock_after")}</td>
                    <td>${formData.get("amount")}</td>
                    <td>${result}</td>
                `;
                historyTableBody.prepend(row);

                Array.from(predictForm.elements).forEach(el => el.disabled = true);
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
        const config = { type: 'pie', data, options: { responsive:true, plugins:{legend:{position:'bottom'}} }};
        if(pieChart) pieChart.destroy();
        pieChart = new Chart(ctx, config);
    }

    // ---------- Retrain ----------
    const retrainForm = document.getElementById("retrainForm");
    const retrainMessage = document.getElementById("retrainMessage");

    retrainForm?.addEventListener("submit", function(e){
        e.preventDefault();
        const formData = new FormData(retrainForm);

        fetch("/retrain", { method:"POST", body:formData })
            .then(res => res.json())
            .then(data => {
                retrainMessage.textContent = data.message || data.status;
                retrainMessage.style.color = data.status === 'success' ? 'green' : 'red';
            })
            .catch(err => alert("Retrain error: " + err));
    });
});
