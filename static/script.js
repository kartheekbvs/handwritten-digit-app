document.addEventListener('DOMContentLoaded', () => {
    // Canvas Setup
    const canvas = document.getElementById('drawingCanvas');
    const ctx = canvas.getContext('2d');
    const clearBtn = document.getElementById('clearBtn');
    const predictBtn = document.getElementById('predictBtn');

    // UI Elements
    const steps = {
        cnn: document.getElementById('step-cnn'),
        pca: document.getElementById('step-pca'),
        lr: document.getElementById('step-lr')
    };
    const resultContainer = document.getElementById('result-container');
    const predictedDigitEl = document.getElementById('predicted-digit');
    const confidenceList = document.getElementById('confidence-list');

    // Canvas State
    let isDrawing = false;
    let hasDrawing = false;

    // Make line thicker and smoother for digits
    ctx.lineWidth = 18;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = '#000000'; // Black ink

    // Start drawing
    function startPosition(e) {
        isDrawing = true;
        hasDrawing = true;
        draw(e);

        // Reset results if modifying
        resetPipelineUI();
    }

    // Stop drawing
    function endPosition() {
        isDrawing = false;
        ctx.beginPath();
    }

    // Get correct mouse/touch position
    function getPos(e) {
        const rect = canvas.getBoundingClientRect();
        const clientX = e.clientX || (e.touches && e.touches[0].clientX);
        const clientY = e.clientY || (e.touches && e.touches[0].clientY);
        return {
            x: clientX - rect.left,
            y: clientY - rect.top
        };
    }

    // Draw function
    function draw(e) {
        if (!isDrawing) return;

        e.preventDefault(); // Prevent scrolling on touch
        const pos = getPos(e);

        ctx.lineTo(pos.x, pos.y);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(pos.x, pos.y);
    }

    // Event Listeners for Canvas
    canvas.addEventListener('mousedown', startPosition);
    canvas.addEventListener('mouseup', endPosition);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseout', endPosition);

    // Touch support for mobile
    canvas.addEventListener('touchstart', startPosition, { passive: false });
    canvas.addEventListener('touchend', endPosition);
    canvas.addEventListener('touchmove', draw, { passive: false });

    // Clear Canvas
    function clearCanvas() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        hasDrawing = false;
        ctx.beginPath();
        resetPipelineUI();
    }
    clearBtn.addEventListener('click', clearCanvas);

    // Pipeline Animation Logic
    async function animatePipelineStep(stepElement, duration) {
        stepElement.classList.remove('completed');
        stepElement.classList.add('active');

        const progressBar = stepElement.querySelector('.progress-bar');
        progressBar.style.width = '0%';

        // Animate progress bar
        return new Promise(resolve => {
            let start = null;
            function step(timestamp) {
                if (!start) start = timestamp;
                const progress = timestamp - start;
                const percentage = Math.min((progress / duration) * 100, 100);
                progressBar.style.width = `${percentage}%`;

                if (progress < duration) {
                    window.requestAnimationFrame(step);
                } else {
                    stepElement.classList.remove('active');
                    stepElement.classList.add('completed');
                    resolve();
                }
            }
            window.requestAnimationFrame(step);
        });
    }

    // Generate fake confidences (mock prediction)
    function generateMockPrediction() {
        const predictedDigit = Math.floor(Math.random() * 10);
        let confidences = [];
        let remainingConf = 100;

        for (let i = 0; i < 10; i++) {
            if (i === predictedDigit) {
                const conf = Math.floor(Math.random() * 20) + 75; // 75-94%
                confidences.push({ digit: i, conf: conf });
                remainingConf -= conf;
            } else {
                confidences.push({ digit: i, conf: 0 });
            }
        }

        // Randomly distribute remaining
        for (let i = 0; i < remainingConf; i++) {
            let randIdx = Math.floor(Math.random() * 10);
            if (randIdx !== predictedDigit) {
                confidences[randIdx].conf += 1;
            } else {
                // If it hits the predicted digit randomly, give it to a neighbor
                let altIdx = (predictedDigit + 1) % 10;
                confidences[altIdx].conf += 1;
            }
        }

        return {
            digit: predictedDigit,
            confidenceList: confidences.sort((a, b) => b.conf - a.conf)
        };
    }

    function renderResults(predictionData) {
        predictedDigitEl.textContent = predictionData.digit;
        confidenceList.innerHTML = '';

        // Take top 4 for neatness
        const top4 = predictionData.confidenceList.slice(0, 4);

        top4.forEach((item, index) => {
            const isTop = index === 0;
            const el = document.createElement('div');
            el.className = `confidence-item ${isTop ? 'top-prediction' : ''}`;

            el.innerHTML = `
                <span class="conf-label">${item.digit}</span>
                <div class="conf-bar-wrapper">
                    <div class="conf-bar" style="width: 0%"></div>
                </div>
                <span class="conf-value">${item.conf}%</span>
            `;

            confidenceList.appendChild(el);

            // Trigger animation after render
            setTimeout(() => {
                el.querySelector('.conf-bar').style.width = `${item.conf}%`;
            }, 50);
        });

        resultContainer.classList.add('show');
    }

    function resetPipelineUI() {
        Object.values(steps).forEach(step => {
            step.classList.remove('active', 'completed');
            const pb = step.querySelector('.progress-bar');
            if (pb) pb.style.width = '0%';
        });
        resultContainer.classList.remove('show');
        predictedDigitEl.textContent = '-';
        confidenceList.innerHTML = '';
    }

    // Predict Action
    predictBtn.addEventListener('click', async () => {
        if (!hasDrawing) {
            alert("Please draw a digit first!");
            return;
        }

        predictBtn.disabled = true;
        clearBtn.disabled = true;
        resetPipelineUI();

        try {
            // Start the visual animations instantly so the user feels immediate feedback
            const pCnn = animatePipelineStep(steps.cnn, 800);

            // In parallel, make the API call to Render Flask App
            const imageData = canvas.toDataURL('image/png');

            const apiPromise = fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ image: imageData })
            }).then(res => {
                if (!res.ok) {
                    throw new Error(`HTTP error! status: ${res.status}`);
                }
                return res.json();
            });

            // Wait for CNN visual to finish
            await pCnn;

            // Then do PCA visual
            await animatePipelineStep(steps.pca, 600);

            // Then do LR visual
            await animatePipelineStep(steps.lr, 400);

            // By now the API call should be finished, wait for it if not
            const result = await apiPromise;

            if (result.error) {
                console.error("Backend Error:", result.error);
                alert(`Backend Error: ${result.error}`);
            } else {
                // Formatting result from Render API to match older mock format
                const formattedResult = {
                    prediction: result.prediction,
                    confidences: result.confidences // array of {digit, conf}
                };

                // Show Result
                renderResults(formattedResult);
            }

        } catch (error) {
            console.error("Prediction failed:", error);
            alert("Failed to connect to the backend ML pipeline API.");
        } finally {
            predictBtn.disabled = false;
            clearBtn.disabled = false;
        }
    });
});
