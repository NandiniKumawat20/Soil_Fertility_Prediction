// Navbar Active Link
document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', function() {
        document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
        this.classList.add('active');
    });
});

// Smooth Scrolling
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth'
            });
        }
    });
});

// Hamburger Menu
const hamburger = document.querySelector('.hamburger');
const navMenu = document.querySelector('.nav-menu');

if (hamburger) {
    hamburger.addEventListener('click', () => {
        navMenu.style.display = navMenu.style.display === 'flex' ? 'none' : 'flex';
        navMenu.style.position = 'absolute';
        navMenu.style.top = '100%';
        navMenu.style.left = '0';
        navMenu.style.right = '0';
        navMenu.style.background = 'white';
        navMenu.style.flexDirection = 'column';
        navMenu.style.padding = '20px';
        navMenu.style.gap = '10px';
        navMenu.style.boxShadow = 'var(--shadow)';
    });
}

// Feature Items Toggle Details
const featureItems = document.querySelectorAll('.feature-item');
featureItems.forEach(item => {
    const title = item.querySelector('h3');
    title.style.cursor = 'pointer';
    
    title.addEventListener('click', () => {
        const details = item.querySelector('.feature-details');
        details.classList.toggle('active');
    });
});

// Nutrient Chart
const nutrientCtx = document.getElementById('nutrientChart');
if (nutrientCtx) {
    const nutrientChart = new Chart(nutrientCtx, {
        type: 'bar',
        data: {
            labels: ['Nitrogen', 'Phosphorus', 'Potassium', 'Sulfur', 'Calcium'],
            datasets: [{
                label: 'Current Levels (mg/kg)',
                data: [42.3, 28.7, 35.8, 15.2, 45.1],
                backgroundColor: [
                    'rgba(34, 197, 94, 0.8)',
                    'rgba(59, 130, 246, 0.8)',
                    'rgba(245, 158, 11, 0.8)',
                    'rgba(168, 85, 247, 0.8)',
                    'rgba(236, 72, 153, 0.8)'
                ],
                borderColor: [
                    'rgb(34, 197, 94)',
                    'rgb(59, 130, 246)',
                    'rgb(245, 158, 11)',
                    'rgb(168, 85, 247)',
                    'rgb(236, 72, 153)'
                ],
                borderWidth: 2,
                borderRadius: 8
            }, {
                label: 'Optimal Range',
                data: [50, 30, 40, 20, 50],
                type: 'line',
                borderColor: '#d1d5db',
                borderWidth: 2,
                borderDash: [5, 5],
                fill: false,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 60,
                    ticks: {
                        callback: function(value) {
                            return value + ' mg/kg';
                        }
                    }
                }
            }
        }
    });
}

// Trends Chart
const trendsCtx = document.getElementById('trendsChart');
if (trendsCtx) {
    const trendsChart = new Chart(trendsCtx, {
        type: 'line',
        data: {
            labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
            datasets: [{
                label: 'Fertility Score',
                data: [70, 72, 75, 73, 76, 78, 79, 81, 80, 79, 78, 78],
                borderColor: 'rgb(34, 197, 94)',
                backgroundColor: 'rgba(34, 197, 94, 0.1)',
                borderWidth: 3,
                fill: true,
                tension: 0.4,
                pointBackgroundColor: 'rgb(34, 197, 94)',
                pointBorderColor: '#fff',
                pointBorderWidth: 2,
                pointRadius: 5,
                pointHoverRadius: 7
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                }
            }
        }
    });
}

// Animate Numbers on Scroll
const animateNumbers = () => {
    const statValues = document.querySelectorAll('.stat-value');
    
    const options = {
        threshold: 0.5
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const target = entry.target;
                const finalValue = parseFloat(target.textContent);
                
                animateValue(target, 0, finalValue, 1500);
                observer.unobserve(target);
            }
        });
    }, options);
    
    statValues.forEach(value => observer.observe(value));
};

const animateValue = (element, start, end, duration) => {
    let startTimer = Date.now();
    let previousValue = start;
    
    const timer = setInterval(() => {
        const now = Date.now();
        const progress = now - startTimer;
        const result = Math.floor((progress / duration) * (end - start) + start);
        
        if (progress >= duration) {
            element.textContent = end;
            clearInterval(timer);
        } else {
            element.textContent = result.toFixed(1);
        }
    }, 50);
};

// Call animation on page load
window.addEventListener('load', animateNumbers);

// Button Functions
document.getElementById('generateReport')?.addEventListener('click', () => {
    showNotification('Report generated successfully! Downloading...', 'success');
    // In a real application, this would generate a PDF
});

document.getElementById('downloadData')?.addEventListener('click', () => {
    showNotification('Data downloaded successfully!', 'success');
    // In a real application, this would trigger a download
});

document.getElementById('shareResults')?.addEventListener('click', () => {
    showNotification('Share link copied to clipboard!', 'success');
    // In a real application, this would copy a share link
});

// Notification System
const showNotification = (message, type = 'info') => {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <div style="display: flex; align-items: center; gap: 10px;">
            <i class="fas fa-${type === 'success' ? 'check-circle' : 'info-circle'}"></i>
            <span>${message}</span>
        </div>
    `;
    
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 15px 20px;
        background: ${type === 'success' ? '#22c55e' : '#3b82f6'};
        color: white;
        border-radius: 8px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        z-index: 9999;
        animation: slideIn 0.3s ease;
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => {
            notification.remove();
        }, 300);
    }, 3000);
};

// Add CSS animations for notifications
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(400px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(400px);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// Active Navigation Link on Scroll
window.addEventListener('scroll', () => {
    let current = '';
    
    const sections = document.querySelectorAll('section');
    sections.forEach(section => {
        const sectionTop = section.offsetTop;
        if (window.pageYOffset >= sectionTop - 200) {
            current = section.getAttribute('id');
        }
    });
    
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href').slice(1) === current) {
            link.classList.add('active');
        }
    });
});

// Scroll Animation for Cards
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -100px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, observerOptions);

document.querySelectorAll('.benefit-card, .feature-item, .chart-container, .stat-card').forEach(el => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(20px)';
    el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
    observer.observe(el);
});

// Mobile Responsive Menu Close
document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', () => {
        const navMenu = document.querySelector('.nav-menu');
        if (navMenu && window.innerWidth <= 768) {
            navMenu.style.display = 'none';
        }
    });
});

// Chart Animations
const animateChart = () => {
    const chartCanvases = document.querySelectorAll('canvas');
    
    chartCanvases.forEach(canvas => {
        const rect = canvas.getBoundingClientRect();
        if (rect.top < window.innerHeight && rect.bottom > 0) {
            canvas.parentElement.style.animation = 'fadeIn 0.6s ease';
        }
    });
};

window.addEventListener('scroll', animateChart);

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    console.log('Dashboard loaded successfully!');
    
    // Add loading animation
    const dashboard = document.querySelector('.dashboard');
    if (dashboard) {
        dashboard.style.animation = 'fadeIn 0.8s ease';
    }
});
