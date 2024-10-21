import React, { useState } from 'react';
import Dropzone from 'react-dropzone';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleDrop = (acceptedFiles) => {
    setSelectedFile(acceptedFiles[0]); // Save the selected file
    setResult(null); // Reset previous result
    setError(null); // Reset previous error
  };

  const handlePredict = async () => {
    if (!selectedFile) return; // Ensure a file is selected
    setLoading(true);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
      setError(null);
    } catch (error) {
      console.error("Error during fetch:", error);
      setError("Failed to fetch prediction.");
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <h1>🌱 Plant Species Predictor 🌿</h1>
      <h2>Upload an image of your plant!</h2>

      <Dropzone onDrop={handleDrop} className="dropzone">
        {({ getRootProps, getInputProps }) => (
          <div {...getRootProps()} className="dropzone">
            <input {...getInputProps()} />
            <p>Drag 'n' drop your image here, or click to select it!</p>
          </div>
        )}
      </Dropzone>

      {selectedFile && (
        <img
          src={URL.createObjectURL(selectedFile)}
          alt="Preview"
          className="preview"
          style={{ marginTop: '20px', width: '300px', height: 'auto', borderRadius: '8px' }}
        />
      )}

      <button className="button" onClick={handlePredict} disabled={loading || !selectedFile}>
        {loading ? 'Predicting...' : 'Predict'}
      </button>

      {result && (
        <div className="result">
          <p>Predicted Class: {result.predicted_class} ☘️</p>
          <p>Confidence: {Math.round(result.confidence * 100)}%</p>
        </div>
      )}
      {error && <p style={{ color: 'red' }}>{error}</p>}
    </div>
  );
}

export default App;
