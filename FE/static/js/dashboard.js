let startTime = Date.now();
let statusChart;
const maxDataPoints = 20;

// Khởi tạo biểu đồ
function initChart() {
    const ctx = document.getElementById('statusChart').getContext('2d');
    statusChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Trạng thái mắt',
                data: [],
                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 2,
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 3,
                    ticks: {
                        stepSize: 1,
                        callback: function(value) {
                            const states = ['', 'Ngủ', 'Ngủ gật', 'Tỉnh táo'];
                            return states[value] || '';
                        }
                    }
                },
                x: {
                    display: false
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            },
            animation: {
                duration: 500
            }
        }
    });
}

// Cập nhật uptime
function updateUptime() {
    let now = Date.now();
    let diff = now - startTime;

    let hours = Math.floor(diff / (1000 * 60 * 60));
    let minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
    let seconds = Math.floor((diff % (1000 * 60)) / 1000);

    document.getElementById("uptime").textContent =
        String(hours).padStart(2, '0') + ':' +
        String(minutes).padStart(2, '0') + ':' +
        String(seconds).padStart(2, '0');
}

// Cập nhật trạng thái mắt
async function updateEyeStatus() {
    try {
        const response = await fetch("/eye_status");
        const data = await response.json();
        
        // Cập nhật text trạng thái
        const statusElement = document.getElementById("eye-status");
        statusElement.textContent = data.status;
        
        // Đổi màu badge theo trạng thái
        statusElement.className = 'status-badge';
        if (data.status === "Ngủ") {
            statusElement.classList.add('alert');
        } else if (data.status === "Ngủ gật") {
            statusElement.classList.add('warning');
        } else {
            statusElement.classList.add('safe');
        }
        
        // Chuyển đổi trạng thái thành số cho biểu đồ
        let statusValue;
        if (data.status === "Ngủ") statusValue = 1;
        else if (data.status === "Ngủ gật") statusValue = 2;
        else statusValue = 3;
        
        // Cập nhật biểu đồ
        const now = new Date().toLocaleTimeString();
        statusChart.data.labels.push(now);
        statusChart.data.datasets[0].data.push(statusValue);
        
        // Giới hạn số điểm dữ liệu
        if (statusChart.data.labels.length > maxDataPoints) {
            statusChart.data.labels.shift();
            statusChart.data.datasets[0].data.shift();
        }
        
        // Đổi màu biểu đồ theo trạng thái
        if (data.status === "Ngủ") {
            statusChart.data.datasets[0].borderColor = 'rgba(220, 53, 69, 1)';
            statusChart.data.datasets[0].backgroundColor = 'rgba(220, 53, 69, 0.2)';
        } else if (data.status === "Ngủ gật") {
            statusChart.data.datasets[0].borderColor = 'rgba(255, 193, 7, 1)';
            statusChart.data.datasets[0].backgroundColor = 'rgba(255, 193, 7, 0.2)';
        } else {
            statusChart.data.datasets[0].borderColor = 'rgba(40, 167, 69, 1)';
            statusChart.data.datasets[0].backgroundColor = 'rgba(40, 167, 69, 0.2)';
        }
        
        statusChart.update();
    } catch (error) {
        console.error("Error fetching eye status:", error);
    }
}

// Khởi động
window.onload = function() {
    initChart();
    setInterval(updateUptime, 1000);
    setInterval(updateEyeStatus, 1000); // Cập nhật mỗi giây
    updateEyeStatus(); // Gọi ngay lần đầu
}