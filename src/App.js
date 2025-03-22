import React, { useState, useRef, useEffect } from "react";
import * as faceapi from "face-api.js";
import * as handpose from "@tensorflow-models/handpose";
import * as tmImage from "@teachablemachine/image";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";

const CameraFeed = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const handModelRef = useRef(null);
  const bookModelRef = useRef(null);
  const [studentsPresent, setStudentsPresent] = useState(new Set());
  const [handRaiseCount, setHandRaiseCount] = useState({});
  const [modelsLoaded, setModelsLoaded] = useState(false);
  const [faceMatcher, setFaceMatcher] = useState(null);
  const [detectedBook, setDetectedBook] = useState(null);

  const bookLabels = {
    "notebook": "Notebook",
    "dictionary": "Dictionary",
    "novel": "Novel",
    "textbook": "Textbook"
  };

  // Initialize TensorFlow.js backend
  useEffect(() => {
    tf.setBackend("webgl").then(() => {
      console.log("TensorFlow.js backend initialized: WebGL");
    });
  }, []);

  // Load Face API Models
  const loadFaceModels = async () => {
    await Promise.all([
      faceapi.nets.ssdMobilenetv1.loadFromUri("/models"),
      faceapi.nets.faceLandmark68Net.loadFromUri("/models"),
      faceapi.nets.faceRecognitionNet.loadFromUri("/models"),
    ]);
  };

  // Load Labeled Face Descriptors
  const loadLabeledImages = async () => {
    const labels = ["Ayush", "Priyanshu", "Shivam", "Dinesh"];
    const labeledDescriptors = [];
    
    for (let label of labels) {
      const img = await faceapi.fetchImage(`/students/${label.toLowerCase()}.jpg`);
      const detections = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor();
      if (detections) {
        labeledDescriptors.push(new faceapi.LabeledFaceDescriptors(label, [detections.descriptor]));
      }
    }
    setFaceMatcher(new faceapi.FaceMatcher(labeledDescriptors));
  };

  // Load Handpose Model
  const loadHandModel = async () => {
    try {
      console.log("⏳ Loading Handpose Model...");
      handModelRef.current = await handpose.load();
      console.log("✅ Handpose Model Loaded!");
    } catch (error) {
      console.error("❌ Error loading Handpose Model:", error);
    }
  };

  // Load Book Classification Model
  const loadBookModel = async () => {
    try {
      console.log("⏳ Loading Teachable Machine Book Model...");
      const modelURL = "/models/model.json";
      const metadataURL = "/models/metadata.json";
      const model = await tmImage.load(modelURL, metadataURL);
      bookModelRef.current = model;
      console.log("✅ Teachable Machine Book Model Loaded!");
    } catch (error) {
      console.error("❌ Error loading book model:", error);
    }
  };

  // Start Video
  useEffect(() => {
    const startVideo = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.onloadedmetadata = () => {
            videoRef.current.play();
            console.log("Video is playing");
          };
        }
      } catch (error) {
        console.error("🚨 Error accessing camera:", error);
      }
    };

    startVideo();
    loadFaceModels().then(loadLabeledImages).then(() => setModelsLoaded(true));
    loadHandModel();
    loadBookModel();
  }, []);

  // Detect Faces
  const detectFaces = async () => {
    if (!modelsLoaded || !videoRef.current || !faceMatcher) return;
    const detections = await faceapi.detectAllFaces(videoRef.current, new faceapi.SsdMobilenetv1Options()).withFaceLandmarks().withFaceDescriptors();
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
        setStudentsPresent((prev) => new Set([...prev, bestMatch.label]));
      }
    });
  };

  // Detect Hands
  const detectHands = async () => {
    if (!handModelRef.current || !videoRef.current) return;

    try {
      const predictions = await handModelRef.current.estimateHands(videoRef.current);
      if (predictions.length > 0) {
        const raisedHands = predictions.filter(prediction => {
          const landmarks = prediction.landmarks;
          const wrist = landmarks[0];
          const middleFingerTip = landmarks[12];
          return middleFingerTip[1] < wrist[1]; // Check if middle finger tip is above the wrist
        });

        setHandRaiseCount((prev) => ({
          ...prev,
          raisedHands: raisedHands.length,
        }));
      }
    } catch (error) {
      console.error("🚨 Error detecting hands:", error);
    }
  };

  // Detect Books
  const detectBook = async () => {
    if (!bookModelRef.current || !videoRef.current) return;

    try {
      const prediction = await bookModelRef.current.predict(videoRef.current);
      console.log("📚 Book Predictions:", prediction);

      if (prediction.length > 0) {
        const highestPrediction = prediction.reduce((prev, curr) =>
          prev.probability > curr.probability ? prev : curr
        );
        setDetectedBook(bookLabels[highestPrediction.className] || "Unknown Book");
      }
    } catch (error) {
      console.error("🚨 Error classifying book:", error);
    }
  };

  // Run Face, Hand & Book Detection Continuously
  useEffect(() => {
    if (modelsLoaded && faceMatcher) {
      const interval = setInterval(() => {
        detectFaces();
        detectHands();
        detectBook();
      }, 100);
      return () => clearInterval(interval);
    }
  }, [modelsLoaded, faceMatcher]);

  return (<div style={{ backgroundColor: "#232F47", color:"white",textAlign: "center" }}>
      <h1 style={{ backgroundColor: "#2569ED", color: "white", padding: "15px", borderRadius: "10px", textAlign: "center" }}>
  SAAMARTH
</h1>


      <h2>📸 Face, Hand & Book Recognition</h2>
      {!modelsLoaded ? <p>⏳ Loading models, please wait...</p> : null}
      <div style={{ position: "relative", display: "inline-block" }}>
        <video ref={videoRef} autoPlay style={{ width: "100%", borderRadius: "10px" }} />
        <canvas ref={canvasRef} style={{ position: "absolute", top: 0, left: 0, width: "100%", borderRadius: "10px" }} />
      </div>
      <div style={{ display: "flex", width: "100%", justifyContent: "space-around", gap: "20px", padding: "15px", borderRadius: "10px" }}>
  <div>
    <h3>✅ Attendance List:</h3>
    <ul>
      {[...studentsPresent].map((name, index) => (
        <li key={index}>{name}</li>
      ))}
    </ul>
  </div>
  
  <div>
    <h3>✋ Raised Hands:</h3>
    <p>{handRaiseCount.raisedHands || 0}</p>
  </div>

  <div>
    <h3>📚 Detected Book:</h3>
    <p>{detectedBook || "No book detected"}</p>
  </div>
</div>
</div>
  );
};

export default CameraFeed;