function classifyImage() {
    const fileInput = document.getElementById("imageUpload");

    if (!fileInput.files.length) {
        alert("Please upload an image");
        return;
    }

    const file = fileInput.files[0];
    const reader = new FileReader();

    reader.onload = function () {
        fetch("/classify_image", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                image: reader.result
            })
        })
        .then(res => res.json())
        .then(data => {
            const rows = document.querySelectorAll("#probability-body tr");

            rows.forEach(row => {
                const playerName = row.children[0].innerText;
                const probCell = row.children[1];

                if (data.probabilities[playerName] !== undefined) {
                    probCell.innerText =
                        (data.probabilities[playerName]).toFixed(2) + "%";
                } else {
                    probCell.innerText = "0.00%";
                }
            });
        })
        .catch(err => {
            console.error(err);
            alert("Classification failed. Check console.");
        });
    };

    reader.readAsDataURL(file);
}
