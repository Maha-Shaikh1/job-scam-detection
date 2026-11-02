// LOGIN
function loginUser() {
    fetch("http://127.0.0.1:5000/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            username: username.value,
            password: password.value
        })
    })
    .then(res => res.json())
    .then(data => {
        if (data.status === "success") {
            if (data.role === "admin") {
                window.location.href = "admin.html";
            } else {
                window.location.href = "dashboard.html";
            }
        } else {
            alert("Invalid Login");
        }
    });
}

// TEXT MODEL
function predictText() {
    fetch("http://127.0.0.1:5000/predict_text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            description: jobdesc.value
        })
    })
    .then(res => res.json())
    .then(data => {
        result.innerText = data.prediction;
    });
}

// NUMERIC MODEL
function predictNumeric() {
    fetch("http://127.0.0.1:5000/predict_numeric", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            salary: salary.value,
            experience: experience.value,
            rating: rating.value
        })
    })
    .then(res => res.json())
    .then(data => {
        result.innerText = data.prediction;
    });
}
