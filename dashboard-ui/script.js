// Custom Chart.js Defaults for Dark Mode Aesthetics
Chart.defaults.color = '#94a3b8';
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(15, 23, 42, 0.9)';
Chart.defaults.plugins.tooltip.titleFont = { family: "'Outfit', sans-serif", size: 14, weight: 'bold' };
Chart.defaults.plugins.tooltip.padding = 12;
Chart.defaults.plugins.tooltip.cornerRadius = 8;
Chart.defaults.plugins.tooltip.borderColor = 'rgba(255, 255, 255, 0.1)';
Chart.defaults.plugins.tooltip.borderWidth = 1;

async function fetchDashboardData() {
    try {
        const response = await fetch('dashboard_data.json');
        if (!response.ok) throw new Error('Network response was not ok');
        return await response.json();
    } catch (error) {
        console.error('Failed to load dashboard data:', error);
        // Fallback data if JSON fails to load (e.g. CORS issues from local files)
        return {
            pareto: {
                rmse: [0.82, 0.85, 0.88, 0.95, 1.05],
                fairness: [0.45, 0.40, 0.35, 0.28, 0.20]
            },
            exposure: {
                base: Array.from({length: 100}, (_, i) => 1000 * Math.pow(0.93, i)),
                dp: Array.from({length: 100}, (_, i) => 500 * Math.pow(0.95, i))
            }
        };
    }
}

function renderParetoChart(data) {
    const ctx = document.getElementById('paretoChart').getContext('2d');
    
    // Prepare scatter data
    const scatterData = data.rmse.map((x, i) => ({
        x: x,
        y: data.fairness[i],
        label: `Sol ${i+1}`
    }));

    new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Pareto Optimal Solutions',
                data: scatterData,
                backgroundColor: '#38bdf8',
                borderColor: '#0ea5e9',
                pointRadius: 8,
                pointHoverRadius: 12,
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: '#38bdf8',
                pointHoverBorderWidth: 3,
                showLine: true, // Connect the dots to show the curve
                fill: false,
                tension: 0.4 // Smooth curve
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: { display: true, text: 'RMSE (Accuracy Loss - Lower is Better)', font: {size: 13, weight: 'bold'} },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' }
                },
                y: {
                    title: { display: true, text: 'Gini Index (Fairness Loss - Lower is Better)', font: {size: 13, weight: 'bold'} },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' }
                }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: (context) => {
                            const point = context.raw;
                            return `${point.label}: RMSE ${point.x.toFixed(3)}, Gini ${point.y.toFixed(3)}`;
                        }
                    }
                }
            },
            animation: {
                duration: 2000,
                easing: 'easeOutQuart'
            }
        }
    });
}

function renderExposureChart(data) {
    const ctx = document.getElementById('exposureChart').getContext('2d');
    const labels = Array.from({length: data.base.length}, (_, i) => i + 1);

    // Create glowing gradients
    const gradBase = ctx.createLinearGradient(0, 0, 0, 400);
    gradBase.addColorStop(0, 'rgba(239, 68, 68, 0.5)');
    gradBase.addColorStop(1, 'rgba(239, 68, 68, 0.0)');

    const gradDP = ctx.createLinearGradient(0, 0, 0, 400);
    gradDP.addColorStop(0, 'rgba(34, 197, 94, 0.5)');
    gradDP.addColorStop(1, 'rgba(34, 197, 94, 0.0)');

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Base Model (Biased)',
                    data: data.base,
                    borderColor: '#ef4444',
                    backgroundColor: gradBase,
                    borderWidth: 2,
                    fill: true,
                    pointRadius: 0,
                    pointHitRadius: 10,
                    tension: 0.4
                },
                {
                    label: 'DP-Fair Model',
                    data: data.dp,
                    borderColor: '#22c55e',
                    backgroundColor: gradDP,
                    borderWidth: 2,
                    fill: true,
                    pointRadius: 0,
                    pointHitRadius: 10,
                    tension: 0.4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            scales: {
                x: {
                    title: { display: true, text: 'Item Rank (Popular to Unpopular)' },
                    grid: { display: false }
                },
                y: {
                    type: 'logarithmic',
                    title: { display: true, text: 'Exposure Count (Log Scale)' },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' }
                }
            },
            animation: {
                duration: 2500,
                easing: 'easeOutQuart'
            }
        }
    });
}

// Initialize Dashboard
document.addEventListener('DOMContentLoaded', async () => {
    const data = await fetchDashboardData();
    renderParetoChart(data.pareto);
    renderExposureChart(data.exposure);
});
