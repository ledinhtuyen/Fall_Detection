var videoElement = document.getElementsByClassName('input_video')[0];
var canvasElement = document.getElementsByClassName('output_canvas')[0];
var canvasCtx = canvasElement.getContext('2d');
// Optimization: Turn off animated spinner after its hiding animation is done.
var spinner = document.querySelector('.loading');
spinner.ontransitionend = function () {
    spinner.style.display = 'none';
};

var fps;
var startTime = Date.now();
var frame = 0;
var isFall = false;

var alertAudio = new Audio('/static/assets/alert.wav');
alertAudio.addEventListener('ended', function() {
    this.currentTime = 0;
    this.play();
}, false);

function tick() {
  var time = Date.now();
  frame++;
  if (time - startTime > 1000) {
      fps = (frame / ((time - startTime) / 1000)).toFixed(1);
      startTime = time;
      frame = 0;
	}
  window.requestAnimationFrame(tick);
}
tick();

function onResults(results) {
    if (!results.poseLandmarks) {
        return;
    }
    document.body.classList.add('loaded');
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);
    canvasCtx.font = "30px Arial bold";
    canvasCtx.fillStyle = "white";
    canvasCtx.fillText('FPS: ' + fps, 10, 40);
    canvasCtx.fillText('Fall: ' + (isFall ? 'Fall' : 'Up'), 10, 80);
    if (results.poseLandmarks) {
        const pos = [];
        for (let i = 0; i < 33; i++) {
            const x = results.poseLandmarks[i].x;
            const y = results.poseLandmarks[i].y;
            const z = results.poseLandmarks[i].z;
            pos.push(x);
            pos.push(y);
            pos.push(z);
        }
        fetch('/api/fall_detection', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({'keypoints': pos})
        })
        .then(response => response.text())
        .then(response => {
            if(response == 'Fall') {
                isFall = true;
                alertAudio.play();
            }
            else {
                isFall = false;
                alertAudio.pause();
            }
        }
        );
        drawConnectors(canvasCtx, results.poseLandmarks, POSE_CONNECTIONS, { visibilityMin: 0.65, color: 'white' });
        drawLandmarks(canvasCtx, Object.values(POSE_LANDMARKS_LEFT).map(function (index) { return results.poseLandmarks[index]; }), { visibilityMin: 0.65, color: 'white', fillColor: 'rgb(255,138,0)' });
        drawLandmarks(canvasCtx, Object.values(POSE_LANDMARKS_RIGHT).map(function (index) { return results.poseLandmarks[index]; }), { visibilityMin: 0.65, color: 'white', fillColor: 'rgb(0,217,231)' });
        drawLandmarks(canvasCtx, Object.values(POSE_LANDMARKS_NEUTRAL).map(function (index) { return results.poseLandmarks[index]; }), { visibilityMin: 0.65, color: 'white', fillColor: 'white' });
    }
    canvasCtx.restore();
}

const pose = new Pose({locateFile: (file) => {
    return `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`;
}});
pose.setOptions({
    modelComplexity: 1,
    smoothLandmarks: true,
    enableSegmentation: false,
    smoothSegmentation: true,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
});
pose.onResults(onResults);

const camera = new Camera(videoElement, {
    onFrame: async () => {
      await pose.send({image: videoElement});
    }
  });
camera.start();
