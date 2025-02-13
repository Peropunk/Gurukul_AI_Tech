
import React, { useState, useRef, useEffect } from "react";
import * as faceapi from "face-api.js";
import * as tf from "@tensorflow/tfjs";
import * as handpose from "@tensorflow-models/handpose";

const CameraFeed = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const handModelRef = useRef(null);
  const [studentsPresent, setStudentsPresent] = useState(new Set());
  const [handRaiseCount, setHandRaiseCount] = useState({});
  const [modelsLoaded, setModelsLoaded] = useState(false);
  const [faceMatcher, setFaceMatcher] = useState(null);

  // ✅ Load Face API Models
  const loadFaceModels = async () => {
    try {
      console.log("⏳ Loading face models...");
      await Promise.all([
        faceapi.nets.ssdMobilenetv1.loadFromUri("/models"),
        faceapi.nets.faceLandmark68Net.loadFromUri("/models"),
        faceapi.nets.faceRecognitionNet.loadFromUri("/models"),
      ]);
      console.log("✅ Face models loaded!");
    } catch (error) {
      console.error("❌ Error loading face models:", error);
    }
  };

  // ✅ Load Labeled Face Descriptors
  const loadLabeledImages = async () => {
    try {
      console.log("⏳ Loading labeled images...");
      const labels = ["Ayush", "priyanshu"];
      const labeledDescriptors = [];

      for (let label of labels) {
        const img = await faceapi.fetchImage(`/students/${label.toLowerCase()}.jpg`);
        const detections = await faceapi
          .detectSingleFace(img)
          .withFaceLandmarks()
          .withFaceDescriptor();

        if (detections) {
          labeledDescriptors.push(new faceapi.LabeledFaceDescriptors(label, [detections.descriptor]));
        }
      }

      setFaceMatcher(new faceapi.FaceMatcher(labeledDescriptors));
      console.log("✅ Face Descriptors Loaded!");
    } catch (error) {
      console.error("❌ Error loading labeled images:", error);
    }
  };

  // ✅ Load Handpose Model
  const loadHandModel = async () => {
    try {
      console.log("⏳ Loading hand detection model...");
      handModelRef.current = await handpose.load();
      console.log("✅ Hand Model Loaded!");
    } catch (error) {
      console.error("❌ Error loading hand model:", error);
    }
  };

  // ✅ Start Camera
  useEffect(() => {
    const startVideo = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) videoRef.current.srcObject = stream;
      } catch (err) {
        console.error("❌ Error accessing webcam:", err);
      }
    };

    startVideo();
    loadFaceModels().then(loadLabeledImages).then(() => setModelsLoaded(true));
    loadHandModel();
  }, []);

  // ✅ Detect Faces
  const detectFaces = async () => {
    if (!modelsLoaded || !videoRef.current || !faceMatcher) return;

    const detections = await faceapi.detectAllFaces(
      videoRef.current,
      new faceapi.SsdMobilenetv1Options()
    ).withFaceLandmarks().withFaceDescriptors();

    if (!detections.length) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    detections.forEach((detection) => {
      const bestMatch = faceMatcher.findBestMatch(detection.descriptor);
      ctx.fillStyle = "#00FF00";
      ctx.font = "16px Arial";
      ctx.fillText(bestMatch.toString(), detection.detection.box.x, detection.detection.box.y - 10);
      faceapi.draw.drawDetections(canvas, [detection]);

      if (bestMatch.label !== "unknown") {
        setStudentsPresent((prev) => {
          const updatedSet = new Set(prev);
          updatedSet.add(bestMatch.label);
          return new Set(updatedSet); // Ensures reactivity
        });
      }
    });

    requestAnimationFrame(detectFaces);
  };
  console.log("Hand Model Ref:", handModelRef.current);

  // ✅ Detect Hand Raises
  const detectHands = async () => {
    if (!handModelRef.current || !videoRef.current) {
      console.log("❌ Hand model or video not ready");
      return;
    }
  
    console.log("🟡 Running hand detection...");
    
    try {
      const predictions = await handModelRef.current.estimateHands(videoRef.current);
      console.log("🔍 Hand detection predictions:", predictions);
      
      if (predictions.length > 0) {
        console.log("✋ Hand detected!", predictions);
      } else {
        console.log("🛑 No hands detected.");
      }
    } catch (error) {
      console.error("🚨 Error detecting hands:", error);
    }
  };
  
  

  // ✅ Run Face & Hand Detection Continuously
  useEffect(() => {
    if (modelsLoaded && faceMatcher) {
      requestAnimationFrame(detectFaces);
      requestAnimationFrame(detectHands);
    }
  }, [modelsLoaded, faceMatcher]);

  return (
    <div style={{ textAlign: "center", position: "relative" }}>
      <h2>📸 Face & Hand Tracking</h2>

      {!modelsLoaded ? <p>⏳ Loading models, please wait...</p> : null}

      <div style={{ position: "relative", display: "inline-block" }}>
        <video ref={videoRef} autoPlay style={{ width: "100%", borderRadius: "10px" }} />
        <canvas
          ref={canvasRef}
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            width: "100%",
            borderRadius: "10px",
          }}
        />
      </div>

      <h3>✅ Attendance List:</h3>
      <ul>
        {[...studentsPresent].map((name, index) => (
          <li key={index}>{name}</li>
        ))}
      </ul>

      <h3>🙋‍♂️ Hand Raise Count:</h3>
      <ul>
        {Object.entries(handRaiseCount).map(([name, count]) => (
          <li key={name}>
            {name}: {count} times
          </li>
        ))}
      </ul>
    </div>
  );
};

export default CameraFeed;
