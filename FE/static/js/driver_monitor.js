// Update data every 200ms
setInterval(updateData, 200);

async function updateData() {
    try {
        const response = await fetch("/api/state");
        const data = await response.json();

        // Update eyes
        updateEye("left", data.eyes.left);
        updateEye("right", data.eyes.right);

        // Update closure duration
        const closureDuration = data.eyes.closure_duration;
        const closureThreshold = data.eyes.closure_threshold;
        document.getElementById("closureDuration").textContent =
            closureDuration.toFixed(1) + "s";

        const closureProgress = Math.min(
            100,
            (closureDuration / closureThreshold) * 100
        );
        const closureBar = document.getElementById("closureProgress");
        closureBar.style.width = closureProgress + "%";
        closureBar.className =
            closureProgress > 75 ? "progress-fill danger" : "progress-fill";

        const closureMetric = document.getElementById("closureMetric");
        closureMetric.className = data.eyes.alert
            ? "metric danger"
            : "metric";

        // Update yawn
        const yawnCount = data.yawn.count;
        const yawnThreshold = data.yawn.threshold;
        document.getElementById(
            "yawnCount"
        ).textContent = `${yawnCount} / ${yawnThreshold}`;

        const yawnProgress = Math.min(
            100,
            (yawnCount / yawnThreshold) * 100
        );
        const yawnBar = document.getElementById("yawnProgress");
        yawnBar.style.width = yawnProgress + "%";
        yawnBar.className =
            yawnProgress >= 100 ? "progress-fill danger" : "progress-fill";

        const yawnMetric = document.getElementById("yawnMetric");
        yawnMetric.className = data.yawn.alert
            ? "metric danger"
            : "metric";

        // Update system status
        const isDrowsy = data.system.is_drowsy;
        const statusIndicator =
            document.getElementById("statusIndicator");
        statusIndicator.className = isDrowsy
            ? "status-indicator alert"
            : "status-indicator";

        const alertBanner = document.getElementById("alertBanner");
        alertBanner.className = isDrowsy
            ? "alert-banner active"
            : "alert-banner";

        // Music status
        const musicStatus = document.getElementById("musicStatus");
        musicStatus.style.display = data.system.music_playing
            ? "block"
            : "none";

        // Statistics
        document.getElementById("eyeAlerts").textContent =
            data.system.total_eye_alerts;
        document.getElementById("yawnAlerts").textContent =
            data.system.total_yawn_alerts;
        document.getElementById("microsleeps").textContent =
            data.system.total_microsleeps;
    } catch (error) {
        console.error("Error fetching data:", error);
    }
}

function updateEye(side, eyeData) {
    const valueEl = document.getElementById(side + "EyeValue");
    const probEl = document.getElementById(side + "EyeProb");
    const metricEl = document.getElementById(side + "EyeMetric");

    valueEl.textContent = eyeData.label;
    probEl.textContent = `Confidence: ${(eyeData.prob * 100).toFixed(
        0
    )}%`;

    if (eyeData.label === "Closed") {
        metricEl.className = "metric warning";
    } else {
        metricEl.className = "metric";
    }
}

async function resetSystem() {
    try {
        await fetch("/api/reset", { method: "POST" });
        alert("System reset successfully!");
    } catch (error) {
        console.error("Error resetting system:", error);
    }
}

function handleVideoError() {
    console.log("Video feed error, attempting to reload...");
    const videoFeed = document.querySelector(".video-feed");
    setTimeout(() => {
        videoFeed.src = "/video_feed?t=" + new Date().getTime();
    }, 2000);
}

document.addEventListener("DOMContentLoaded", function () {
    const videoFeed = document.querySelector(".video-feed");
    videoFeed.onerror = handleVideoError;
    videoFeed.onload = function () {
        console.log("Video feed loaded successfully");
    };
});

async function checkCameraStatus() {
    try {
        const response = await fetch("/api/camera_status");
        const data = await response.json();

        const cameraMetric = document.getElementById("cameraMetric");
        const cameraStatus = document.getElementById("cameraStatus");
        const cameraInfo = document.getElementById("cameraInfo");

        if (data.camera_working) {
            cameraStatus.textContent = "✅ Working";
            cameraMetric.className = "metric";
        } else {
            cameraStatus.textContent = "❌ Test Mode";
            cameraMetric.className = "metric warning";
        }

        cameraInfo.textContent = `Frames: ${data.frame_count} | Errors: ${data.error_count}`;
    } catch (error) {
        console.error("Error checking camera status:", error);
    }
}

// Gọi hàm này mỗi 2 giây
setInterval(checkCameraStatus, 2000);