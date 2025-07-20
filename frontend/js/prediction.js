class LaptopPricePredictor {
  constructor() {
    this.form = document.getElementById("predictionForm");
    this.resultDiv = document.getElementById("predictionResult");
    this.chart = null;
    this.init();
  }

  init() {
    this.form.addEventListener("submit", (e) => this.handleSubmit(e));
    this.initializeTooltips();
    this.setupDynamicValidation();
  }

  handleSubmit(e) {
    e.preventDefault();

    // Show loading state
    this.showLoading();

    // Get form data
    const formData = new FormData(this.form);
    const specs = Object.fromEntries(formData.entries());

    // Validate data
    if (!this.validateSpecs(specs)) {
      this.showError("Please fill in all required fields correctly");
      return;
    }

    fetch("http://localhost:4000/api/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(specs),
    })
      .then((response) => response.json())
      .then((prediction) => {
        if (prediction.error) {
          this.showError(prediction.error);
        } else {
          this.showPrediction(prediction);
          this.updateChart(prediction); // Optional if using chart
        }
      })
      .catch((error) => {
        this.showError("Error contacting server: " + error.message);
      });
  }
   
  showPrediction(prediction) {
    this.resultDiv.innerHTML = `
            <div class="prediction-card">
                <h3>Estimated Price</h3>
                <div class="price">$${prediction.estimatedPrice.toLocaleString()}</div>
                <div class="confidence">
                    Confidence: ${prediction.confidence}%
                    <div class="confidence-bar">
                        <div class="confidence-level" style="width: ${
                          prediction.confidence
                        }%"></div>
                    </div>
                </div>
                <div class="price-range">
                    Price Range: $${prediction.priceRange.low.toLocaleString()} - $${prediction.priceRange.high.toLocaleString()}
                </div>
            </div>
        `;
  }

  showLoading() {
    this.resultDiv.innerHTML = `
            <div class="loading">
                <div class="spinner"></div>
                <p>Calculating prediction...</p>
            </div>
        `;
  }

  showError(message) {
    this.resultDiv.innerHTML = `
            <div class="error-message">
                <i class="fas fa-exclamation-circle"></i>
                ${message}
            </div>
        `;
  }

  validateSpecs(specs) {
    // Add validation rules here
    return true;
  }

  initializeTooltips() {
    const tooltips = document.querySelectorAll("[data-tooltip]");
    tooltips.forEach((element) => {
      const tooltip = document.createElement("div");
      tooltip.className = "tooltip";
      tooltip.textContent = element.dataset.tooltip;
      element.appendChild(tooltip);
    });
  }

  setupDynamicValidation() {
    const inputs = this.form.querySelectorAll("input, select");
    inputs.forEach((input) => {
      input.addEventListener("input", () => {
        this.validateInput(input);
      });
    });
  }

  validateInput(input) {
    const validationRules = {
      ram: (value) => value >= 2 && value <= 128,
      storage: (value) => value >= 128 && value <= 4000,
      screenSize: (value) => value >= 11 && value <= 17,
    };

    const rule = validationRules[input.name];
    if (rule && !rule(input.value)) {
      input.classList.add("invalid");
    } else {
      input.classList.remove("invalid");
    }
  }
}

// Initialize predictor when document is ready
document.addEventListener("DOMContentLoaded", () => {
  new LaptopPricePredictor();
});
