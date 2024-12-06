document.getElementById("train-form").addEventListener("submit", async (event) => {
    event.preventDefault();  // Prevent the form from submitting normally

    const trainButton = document.getElementById("train-btn");
    const progressBar = document.getElementById("progress-bar");
    const trainStatus = document.getElementById("train-status");

    // Disable the button and show the progress bar
    trainButton.disabled = true;
    progressBar.style.display = "block";
    progressBar.value = 0; // Reset progress bar
    trainStatus.innerHTML = ""; // Clear status message

    try {
        // Send POST request to start training
        const response = await fetch("/train-model-api", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
        });

        if (!response.ok) {
            throw new Error("Failed to start training");
        }

        // Poll for progress updates
        const interval = setInterval(async () => {
            const progressResponse = await fetch("/progress");
            const progressData = await progressResponse.json();
            const progress = progressData.progress;

            // Update the progress bar
            progressBar.value = progress;

            if (progress >= 100) {
                clearInterval(interval); // Stop polling when complete
                trainStatus.innerHTML = `<p class="text-success">Training complete!</p>`;
            }
        }, 1000); // Check progress every second

    } catch (error) {
        trainStatus.innerHTML = `<p class="text-danger">Error: ${error.message}</p>`;
    } finally {
        trainButton.disabled = false; // Re-enable the button
    }
});