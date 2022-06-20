import "./Model.css";
import Webcam from "react-webcam";
import { useRef, useState } from "react";
import {
  Holistic,
  HAND_CONNECTIONS,
  POSE_CONNECTIONS,
} from "@mediapipe/holistic";
import { Camera } from "@mediapipe/camera_utils";
import encoder from "./encoder.json";
import { drawConnectors, drawLandmarks } from "@mediapipe/drawing_utils";
import * as tf from "@tensorflow/tfjs";
import { Progress } from "react-sweet-progress";
import "react-sweet-progress/lib/style.css";

const Model = () => {
  const [display, setDisplay] = useState("none");
  const [mountError, setMountError] = useState(false);
  const [predictionError,setPredictionError]=useState(false)
  const [top3, setTop3] = useState(null);
  const canvas = useRef(null);
  const model = useRef(null);
  const resultarr = useRef([]);
  const webcam = useRef(null);

  const afterMount = async () => {
    let loadModel = new Promise(async (resolve, reject) => {
      const holistic = new Holistic({
        locateFile: (file) => {
          return `/static/${file}`;
        },
      });
      holistic.setOptions({
        modelComplexity: 1,
        smoothLandmarks: true,
        minDetectionConfidence: 0.1,
        minTrackingConfidence: 0.1,
      });
      holistic.onResults(onResults);
      model.current = await tf.loadLayersModel(
        "/static/model.json"
      );
      resolve(
        new Camera(webcam.current.video, {
          onFrame: async () => {
            await holistic.send({ image: webcam.current.video });
          },
        })
      );
    });
    await loadModel.then((camera) => camera.start());
  };

  const extractKeyPoints = (results) => {
    let la;
    if (results.poseLandmarks) {
      const poseleftPoints = [13, 15];
      la = poseleftPoints.map((index) => {
        if (results.poseLandmarks[index].visibility > 0.2) {
          const { x, y, z } = results.poseLandmarks[index];
          return [x, y, z];
        } else {
          return [0, 0, 0];
        }
      });
    } else {
      la = tf.zeros([2, 3]).arraySync();
    }
    let lh;
    if (results.leftHandLandmarks) {
      lh = results.leftHandLandmarks.map((landmark) => {
        const { x, y, z } = landmark;
        return [x, y, z];
      });
    } else {
      lh = tf.zeros([21, 3]).arraySync();
    }
    let ra;
    if (results.poseLandmarks) {
      const poseRightPoints = [14, 16];
      ra = poseRightPoints.map((index) => {
        if (results.poseLandmarks[index].visibility > 0.2) {
          const { x, y, z } = results.poseLandmarks[index];
          return [x, y, z];
        } else {
          return [0, 0, 0];
        }
      });
    } else {
      ra = tf.zeros([2, 3]).arraySync();
    }
    let rh;
    if (results.rightHandLandmarks) {
      rh = results.rightHandLandmarks.map((landmark) => {
        const { x, y, z } = landmark;
        return [x, y, z];
      });
    } else {
      rh = tf.zeros([21, 3]).arraySync();
    }

    return [...la, ...lh, ...ra, ...rh];
  };

  const onResults = (results) => {
    setDisplay("grid");
    canvas.current.width = webcam.current.video.videoWidth;
    canvas.current.height = webcam.current.video.videoHeight;
    const canvasElement = canvas.current;
    const canvasCtx = canvasElement.getContext("2d");
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    // Only overwrite existing pixels.
    canvasCtx.globalCompositeOperation = "source-in";
    canvasCtx.fillStyle = "#00FF00";
    canvasCtx.fillRect(0, 0, canvasElement.width, canvasElement.height);

    // Only overwrite missing pixels.
    canvasCtx.globalCompositeOperation = "destination-atop";

    canvasCtx.drawImage(
      results.image,
      0,
      0,
      canvasElement.width,
      canvasElement.height
    );

    canvasCtx.globalCompositeOperation = "source-over";
    drawConnectors(canvasCtx, results.poseLandmarks, POSE_CONNECTIONS, {
      color: "#00FF00",
      lineWidth: 4,
    });
    drawLandmarks(canvasCtx, results.poseLandmarks, {
      color: "#FF0000",
      lineWidth: 2,
    });
    drawConnectors(canvasCtx, results.leftHandLandmarks, HAND_CONNECTIONS, {
      color: "#CC0000",
      lineWidth: 5,
    });
    drawLandmarks(canvasCtx, results.leftHandLandmarks, {
      color: "#00FF00",
      lineWidth: 2,
    });
    drawConnectors(canvasCtx, results.rightHandLandmarks, HAND_CONNECTIONS, {
      color: "#00CC00",
      lineWidth: 5,
    });
    drawLandmarks(canvasCtx, results.rightHandLandmarks, {
      color: "#FF0000",
      lineWidth: 2,
    });
    canvasCtx.restore();
    resultarr.current = [...resultarr.current, extractKeyPoints(results)];
    if (resultarr.current.length === 30) {
      let { values, indices } = tf.topk(
        model.current.predict(
          tf.reshape(
            tf.tensor3d(resultarr.current, [30, 46, 3]),
            [1, 30, 46, 3]
          )
        ),
        3
      );
      values = values.arraySync();
      indices = indices.arraySync();
      setTop3(
        Object.fromEntries(indices[0].map((e, i) => [encoder[e], values[0][i]]))
      );
      resultarr.current = resultarr.current.slice(1);
    }
    if(results.leftHandLandmarks===undefined && results.rightHandLandmarks===undefined){
      setPredictionError(true)
    }else{
      setPredictionError(false)
    }
  };

  return (
    <>
      {display === "none" ? (
        <div className="loader">
          {mountError ? null : <div className="spinner"></div>}
          {mountError ? (
            <p>check your device's camera</p>
          ) : (
            <p>Loading Model and Files that might take a while</p>
          )}
        </div>
      ) : null}
      <div className="ModelContainer" style={{ display }}>
        <div className="model">
          <Webcam
            ref={webcam}
            className="camera"
            audio={false}
            videoConstraints={{
              facingMode: "user",
              frameRate: 30,
            }}
            onUserMedia={afterMount}
            onUserMediaError={() => {
              setMountError(true);
            }}
          />
          <canvas ref={canvas} className="camera canv"></canvas>
        </div>
        {top3 === null || predictionError ? (
          <div className="spinner2Block">
            <div className="spinner2"></div>
            {predictionError?<p>{'Hands not detected'}</p>:null}
          </div>
        ) : (
          <div className="prediction">
            <div className="current">
              <h1>{`${Object.entries(top3)[0][0]}`}</h1>
            </div>
            <div className="top3Prediction">
              {Object.entries(top3).map((e, i) => (
                <div key={e[0]}>
                  <p
                    key={e[0]}
                    style={{ textAlign: "right", paddingRight: "20px" }}
                  >
                    {`${e[0]}`}
                  </p>
                  <Progress key={i} percent={Math.round(e[1] * 100)} />
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </>
  );
};

export default Model;
