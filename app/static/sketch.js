const video3 = document.getElementsByClassName('input_video3')[0];
const out3 = document.getElementsByClassName('output3')[0];
const controlsElement3 = document.getElementsByClassName('control3')[0];
const canvasCtx3 = out3.getContext('2d');
const fpsControl = new FPS();

const spinner = document.querySelector('.loading');
spinner.ontransitionend = () => {
  spinner.style.display = 'none';
};

function onResultsHands(results) {
  document.body.classList.add('loaded');
  fpsControl.tick();

  canvasCtx3.save();
  canvasCtx3.clearRect(0, 0, out3.width, out3.height);
  canvasCtx3.drawImage(
    results.image, 0, 0, out3.width, out3.height);
  if (results.multiHandLandmarks && results.multiHandedness) {
    for (let index = 0; index < results.multiHandLandmarks.length; index++) {
      const classification = results.multiHandedness[index];
      const isRightHand = classification.label === 'Right';
      const landmarks = results.multiHandLandmarks[index];
      drawConnectors(
        canvasCtx3, landmarks, HAND_CONNECTIONS,
        { color: isRightHand ? '#00FF00' : '#FF0000' }),
        drawLandmarks(canvasCtx3, landmarks, {
          color: isRightHand ? '#00FF00' : '#FF0000',
          fillColor: isRightHand ? '#FF0000' : '#00FF00',
          radius: (x) => {
            return lerp(x.from.z, -0.15, .1, 10, 1);
          }
        });
    }
  }
  canvasCtx3.restore();
}

const hands = new Hands({
  locateFile: (file) => {
    return `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.1/${file}`;
  }
});
hands.onResults(onResultsHands);

const camera = new Camera(video3, {
  onFrame: async () => {
    await hands.send({ image: video3 });
  },
  width: 480,
  height: 480
});
camera.start();

new ControlPanel(controlsElement3, {
  selfieMode: true,
  maxNumHands: 2,
  minDetectionConfidence: 0.7,
  minTrackingConfidence: 0.5
})
  .add([
    new StaticText({ title: 'MediaPipe Hands' }),
    fpsControl,
    new Toggle({ title: 'Selfie Mode', field: 'selfieMode' }),
    new Slider(
      { title: 'Max Number of Hands', field: 'maxNumHands', range: [1, 4], step: 1 }),
    new Slider({
      title: 'Min Detection Confidence',
      field: 'minDetectionConfidence',
      range: [0, 1],
      step: 0.01
    }),
    new Slider({
      title: 'Min Tracking Confidence',
      field: 'minTrackingConfidence',
      range: [0, 1],
      step: 0.01
    }),
  ])
  .on(options => {
    video3.classList.toggle('selfie', options.selfieMode);
    hands.setOptions(options);
  });


const clfPromise = tf.loadLayersModel('model_ann.h5');

let previousPrediction = null;
let predictedString = '';
let landmarksHistory = [];

hands.onResults((results) => {
  const image = results.image;

  if (results.multiHandLandmarks && results.multiHandedness) {
    for (let idx = 0; idx < results.multiHandLandmarks.length; idx++) {
      const classification = results.multiHandedness[idx];
      const handedness = classification.label;

      if (handedness === 'Right') {
        const handLandmarks = results.multiHandLandmarks[idx];

        // Clean hand landmarks data
        const cleanedLandmark = dataClean(handLandmarks);

        if (cleanedLandmark) {
          // Check hand stability based on landmark movement
          landmarksHistory.push(cleanedLandmark);
          if (landmarksHistory.length > 10) {
            landmarksHistory.shift();
            if (checkHandStability(landmarksHistory)) {
              clfPromise.then(clf => {
                // Make predictions using the trained model
                const yPred = clf.predict(cleanedLandmark);
                // Update the predicted sequence if a new prediction is made
                if (yPred !== previousPrediction) {
                  previousPrediction = yPred;
                  if (yPred === 'del') {
                    predictedString = '';
                    previousPrediction = null;
                  } else if (yPred === 'space') {
                    predictedString += ' ';
                  } else {
                    predictedString += yPred;
                  }
                }

                // Display the predicted sequence
                document.getElementById('prediction').innerText = predictedString;
              });
            }
          }
        }
      }
    }
  }
});



function dataClean(landmark) {
  let data = landmark[0];

  try {
    data = data.toString();

    data = data.trim().split('\n');

    const garbage = ['landmark {', '  visibility: 0.0', '  presence: 0.0', '}'];

    let withoutGarbage = [];

    for (let i = 0; i < data.length; i++) {
      if (!garbage.includes(data[i])) {
        withoutGarbage.push(data[i]);
      }
    }

    let clean = [];

    for (let i = 0; i < withoutGarbage.length; i++) {
      let trimmed = withoutGarbage[i].trim();
      clean.push(parseFloat(trimmed.substring(2)));
    }

    return [clean];

  } catch (error) {
    return new Array(63).fill(0);
  }
}

// Function to calculate the Euclidean distance between two points
function calculateDistance(point1, point2) {
  let sum = 0;
  for (let i = 0; i < point1.length; i++) {
    sum += Math.pow((point1[i] - point2[i]), 2);
  }
  return Math.sqrt(sum);
}

// Function to check hand stability based on landmark movement
function checkHandStability(landmarksHistory) {
  // Check if the Euclidean distances between consecutive landmark positions are below a threshold
  const threshold = 10.0;
  for (let i = 0; i < landmarksHistory[0].length - 1; i++) {
    const distance = calculateDistance(
      landmarksHistory[0][i], landmarksHistory[0][i + 1]);
    if (distance > threshold) {
      return false;
    }
  }
  return true;
}
