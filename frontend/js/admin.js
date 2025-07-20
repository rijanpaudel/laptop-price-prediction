// Populate recent predictions table
document.addEventListener('DOMContentLoaded', function() {
    const mockPredictions = [
        {
            date: '2025-06-26',
            user: 'john.doe@example.com',
            specs: 'Dell, i7, 16GB RAM, 512GB SSD, 15.6"',
            price: '$1,299'
        },
        {
            date: '2025-06-25',
            user: 'jane.smith@example.com',
            specs: 'HP, i5, 8GB RAM, 256GB SSD, 14"',
            price: '$899'
        },
        {
            date: '2025-06-25',
            user: 'mike.wilson@example.com',
            specs: 'Lenovo, i9, 32GB RAM, 1TB SSD, 17"',
            price: '$2,499'
        }
    ];

    const tbody = document.getElementById('recentPredictions');
    if (tbody) {
        mockPredictions.forEach(pred => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${pred.date}</td>
                <td>${pred.user}</td>
                <td>${pred.specs}</td>
                <td>${pred.price}</td>
            `;
            tbody.appendChild(row);
        });
    }
});

// Update stats periodically
function updateStats() {
    // In a real application, this would fetch real-time stats from the backend
    const totalUsers = Math.floor(Math.random() * 1000) + 1000;
    const predictions = Math.floor(Math.random() * 5000) + 5000;
    const accuracy = (Math.random() * 5 + 90).toFixed(1);

    document.querySelector('.stat-card:nth-child(1) p').textContent = totalUsers.toLocaleString();
    document.querySelector('.stat-card:nth-child(2) p').textContent = predictions.toLocaleString();
    document.querySelector('.stat-card:nth-child(3) p').textContent = accuracy + '%';
}

// Update stats every 30 seconds
setInterval(updateStats, 30000);