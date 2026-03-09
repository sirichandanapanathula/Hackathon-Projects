function handleEnter(event) {
    if (event.key === "Enter") {
        event.preventDefault();
        searchMedicine();
    }
}

function searchMedicine() {
    let issue = document.getElementById("searchBox").value.trim().toLowerCase();
    let medicines = {
        "flu": "Paracetamol, Ibuprofen, Rest and Hydration", 
        "diabetes": "Metformin, Insulin, Glimepiride",
        "hypertension": "Amlodipine, Lisinopril, Losartan",
        "asthma": "Salbutamol, Budesonide, Montelukast",
        "covid-19": "Remdesivir, Paracetamol, Oxygen Therapy",
        "fever": "Dolo 650, Crocin, Paracetamol",
        "cold": "Cetirizine, Allegra, Steam Inhalation",
        "headache": "Aspirin, Ibuprofen, Paracetamol",
        "migraine": "Sumatriptan, Rizatriptan, Naproxen"
    };

    let output = document.getElementById("medicine-output");

    if (medicines[issue]) {
        output.innerHTML = `✅ Recommended Medicines: <strong>${medicines[issue]}</strong>`;
        output.style.color = "green";
    } else {
        output.innerHTML = "❌ No recommendations available. Try another disease.";
        output.style.color = "red";
    }
}
