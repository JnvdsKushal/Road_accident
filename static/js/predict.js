// Handle form submission
qForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  qSpin.style.display = 'inline-block';
  qRes.style.display = 'none';

  const data = {
    Weather_Conditions: Number(document.getElementById('q_weather').value || 1),
    Road_Surface_Conditions: Number(document.getElementById('q_road').value || 1),
    Speed_limit: Number(document.getElementById('q_speed').value || 30),
    Traffic: Number(document.getElementById('q_traffic').value || 3),
    Time: Number(document.getElementById('q_time').value || 12),
  };

  try {
    const response = await fetch('/predict_json/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const result = await response.json();

    qSpin.style.display = 'none';
    qRes.style.display = 'block';

    // Display the backend's prediction result
    qBadge.textContent = result.predicted_label || 'Unknown';
    qText.textContent = `Prediction Code: ${result.predicted_code || 'N/A'}`;
  } catch (err) {
    console.error('Prediction error:', err);
    qSpin.style.display = 'none';
    qRes.style.display = 'block';
    qBadge.textContent = 'Error';
    qText.textContent = 'Failed to fetch prediction. Check console for details.';
  }
});
