// Login form handling
document.getElementById('loginForm')?.addEventListener('submit', function(e) {
    e.preventDefault();
    
    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;

    // Simple validation
    if (!email || !password) {
        alert('Please fill in all fields');
        return;
    }

    // In a real application, this would make an API call to authenticate
    // For demo purposes, we'll just simulate a successful login
    if (email === 'admin@example.com' && password === 'admin') {
        window.location.href = 'admin.html';
    } else {
        window.location.href = 'prediction.html';
    }
});

// Registration form handling (if exists on page)
document.getElementById('registerForm')?.addEventListener('submit', function(e) {
    e.preventDefault();
    
    const name = document.getElementById('name').value;
    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;
    const confirmPassword = document.getElementById('confirmPassword').value;

    // Simple validation
    if (!name || !email || !password || !confirmPassword) {
        alert('Please fill in all fields');
        return;
    }

    if (password !== confirmPassword) {
        alert('Passwords do not match');
        return;
    }

    // In a real application, this would make an API call to register the user
    // For demo purposes, we'll just redirect to login
    alert('Registration successful! Please login.');
    window.location.href = 'login.html';
});