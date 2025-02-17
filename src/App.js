
import React, { useState, useRef, useEffect } from "react";
import * as faceapi from "face-api.js";
import * as handpose from "@tensorflow-models/handpose";
import * as mobilenet from "@tensorflow-models/mobilenet";
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
    const labels = ["Ayush", "Priyanshu"];
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
    handModelRef.current = await handpose.load();
  };

  // Load Book Classification Model
  const loadBookModel = async () => {
    try {
      console.log("⏳ Loading Teachable Machine Book Model...");
  
      const modelURL = "/models/model.json";
      const metadataURL = "/models/metadata.json";
  
      // Load the model
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
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) videoRef.current.srcObject = stream;
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
        detectBook(); // Call the book detection function
      }, 1000);
      return () => clearInterval(interval);
    }
  }, [modelsLoaded, faceMatcher]);
  

  return (
    <div style={{ textAlign: "center", position: "relative" }}>
      <h2>📸 Face, Hand & Book Recognition</h2>
      {!modelsLoaded ? <p>⏳ Loading models, please wait...</p> : null}
      <div style={{ position: "relative", display: "inline-block" }}>
        <video ref={videoRef} autoPlay style={{ width: "100%", borderRadius: "10px" }} />
        <canvas ref={canvasRef} style={{ position: "absolute", top: 0, left: 0, width: "100%", borderRadius: "10px" }} />
      </div>
      <h3>✅ Attendance List:</h3>
      <ul>
        {[...studentsPresent].map((name, index) => (
          <li key={index}>{name}</li>
        ))}
      </ul>
      <h3>📚 Detected Book:</h3>
      <p>{detectedBook || "No book detected"}</p>
    </div>
  );
};

export default CameraFeed;
