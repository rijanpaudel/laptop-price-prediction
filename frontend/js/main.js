// Loader
window.addEventListener('load', () => {
    const loader = document.getElementById('loader');
    setTimeout(() => {
        loader.style.display = 'none';
    }, 1000);
});

// Navigation Toggle
const navToggle = document.getElementById('navToggle');
const navMenu = document.querySelector('.nav-menu');

navToggle.addEventListener('click', () => {
    navMenu.classList.toggle('active');
});

// Smooth Scrolling
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
        e.preventDefault();
        document.querySelector(this.getAttribute('href')).scrollIntoView({
            behavior: 'smooth'
        });
    });
});

// Animation on Scroll
const animateOnScroll = () => {
    const elements = document.querySelectorAll('.feature-card, .step');
    elements.forEach(element => {
        const elementPosition = element.getBoundingClientRect().top;
        const screenPosition = window.innerHeight;
        
        if(elementPosition < screenPosition) {
            element.classList.add('animate');
        }
    });
};

window.addEventListener('scroll', animateOnScroll);

// Form Validation
const validateForm = (formId) => {
    const form = document.getElementById(formId);
    if (!form) return;

    form.addEventListener('submit', (e) => {
        e.preventDefault();
        
        // Reset errors
        const errors = form.querySelectorAll('.error-message');
        errors.forEach(error => error.remove());

        // Validate fields
        let isValid = true;
        const inputs = form.querySelectorAll('input, select');
        
        inputs.forEach(input => {
            if (input.hasAttribute('required') && !input.value) {
                isValid = false;
                showError(input, 'This field is required');
            }

            if (input.type === 'email' && input.value) {
                const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
                if (!emailRegex.test(input.value)) {
                    isValid = false;
                    showError(input, 'Please enter a valid email address');
                }
            }
        });

        if (isValid) {
            // Show success message
            showSuccess(form);
            
            // Simulate form submission
            simulateSubmission(form);
        }
    });
};

const showError = (input, message) => {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.textContent = message;
    input.parentNode.appendChild(errorDiv);
    input.classList.add('error');
};

const showSuccess = (form) => {
    const successDiv = document.createElement('div');
    successDiv.className = 'success-message';
    successDiv.textContent = 'Form submitted successfully!';
    form.appendChild(successDiv);
};

const simulateSubmission = (form) => {
    const submitButton = form.querySelector('button[type="submit"]');
    submitButton.disabled = true;
    submitButton.textContent = 'Processing...';
    
    setTimeout(() => {
        submitButton.disabled = false;
        submitButton.textContent = 'Submit';
        form.reset();
    }, 2000);
};

// Initialize form validation
validateForm('loginForm');
validateForm('registerForm');
validateForm('predictionForm');

// Dark Mode Toggle
const darkModeToggle = document.getElementById('darkModeToggle');
if (darkModeToggle) {
    darkModeToggle.addEventListener('click', () => {
        document.body.classList.toggle('dark-mode');
        localStorage.setItem('darkMode', document.body.classList.contains('dark-mode'));
    });

    // Check for saved dark mode preference
    if (localStorage.getItem('darkMode') === 'true') {
        document.body.classList.add('dark-mode');
    }
}