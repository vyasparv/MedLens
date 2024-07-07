document.getElementById('upload-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    const formData = new FormData(event.target);
    
    const response = await fetch('/predict', {
        method: 'POST',
        body: formData
    });
    
    const result = await response.json();
    document.getElementById('result').innerText = `Health Status: ${result.health_status} (Score: ${result.prediction_score.toFixed(2)})`;
});
