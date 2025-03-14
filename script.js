{
    "handleSubmit": "function() {
        const form = document.getElementById('input-form');
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            const textInput = document.getElementById('text-input').value;
            const audioInput = document.getElementById('audio-input').files[0];

            let formData = new FormData();
            if (textInput) {
                formData.append('text', textInput);
            } else if (audioInput) {
                formData.append('audio', audioInput);
            } else {
                alert('Please enter text or upload audio.');
                return;
            }

            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            const result = await response.json();
            document.getElementById('response').innerText = result.response || result.error;
        });
    }"
}
