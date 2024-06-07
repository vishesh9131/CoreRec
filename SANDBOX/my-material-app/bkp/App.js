import React, { useState, useEffect } from 'react';
import { Container, Button, Select, MenuItem, Slider, Typography } from '@mui/material';
import axios from 'axios';

function App() {
  const [model, setModel] = useState('');
  const [label, setLabel] = useState('');
  const [topK, setTopK] = useState(2);
  const [threshold, setThreshold] = useState(0.5);
  const [recommendations, setRecommendations] = useState([]);
  const [labels, setLabels] = useState([]);
  const [models, setModels] = useState([]);

  useEffect(() => {
    // Fetch labels from the backend
    const fetchLabels = async () => {
      try {
        const response = await axios.get('http://localhost:5000/labels');
        console.log('Fetched labels:', response.data.labels); // Debug log
        setLabels(response.data.labels);
      } catch (error) {
        console.error('Error fetching labels:', error);
        if (error.response) {
          console.error('Error response data:', error.response.data);
        }
      }
    };
    fetchLabels();

    // Fetch model options from the backend
    const fetchModels = async () => {
      try {
        const response = await axios.get('http://localhost:5000/models');
        console.log('Fetched models:', response.data.models); // Debug log
        setModels(response.data.models);
      } catch (error) {
        console.error('Error fetching models:', error);
        if (error.response) {
          console.error('Error response data:', error.response.data);
        }
      }
    };
    fetchModels();
  }, []);

  const handleRunModel = async () => {
    try {
      const response = await axios.post('http://localhost:5000/predict', {
        model, label, topK, threshold
      });
      console.log('Received recommendations:', response.data.recommendations); // Debug log
      setRecommendations(response.data.recommendations);
    } catch (error) {
      console.error('Error running model:', error);
      if (error.response) {
        console.error('Error response data:', error.response.data);
      }
    }
  };

  return (
    <Container>
      <Typography variant="h4">Test_A</Typography>
      <Select value={model} onChange={(e) => setModel(e.target.value)} displayEmpty>
        <MenuItem value="" disabled>Select model</MenuItem>
        {models.map((mdl, index) => (
          <MenuItem key={index} value={mdl.value}>{mdl.label}</MenuItem>
        ))}
      </Select>
      <Select value={label} onChange={(e) => setLabel(e.target.value)} displayEmpty>
        <MenuItem value="" disabled>Select node label</MenuItem>
        {labels.map((lbl, index) => (
          <MenuItem key={index} value={lbl}>{lbl}</MenuItem>
        ))}
      </Select>
      <Typography>Top K</Typography>
            <Slider value={topK} onChange={(e, val) => setTopK(val)} min={1} max={10} />
            <Typography>Threshold</Typography>
            <Slider value={threshold} onChange={(e, val) => setThreshold(val)} min={0} max={1} step={0.1} />
            <Button variant="contained" color="primary" onClick={handleRunModel}>Run Model</Button>
            <Typography>Recommended nodes:</Typography>
            {recommendations.map((rec, index) => (
              <Typography key={index}>{rec}</Typography>
            ))}
          </Container>
        );
      }
      
      export default App;