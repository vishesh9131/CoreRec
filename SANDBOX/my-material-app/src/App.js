import React, { useState, useEffect, useMemo } from 'react';
import { Container, Button, Select, MenuItem, Slider, Typography, Box, AppBar, Toolbar, CssBaseline, Switch, FormControlLabel, Paper, Grid, Avatar, Card, CardContent, Divider } from '@mui/material';
import { createTheme, ThemeProvider, styled } from '@mui/material/styles';
import useMediaQuery from '@mui/material/useMediaQuery';
import axios from 'axios';
import DraggableWrapper from './DraggableWrapper'; // Import custom Draggable wrapper
import ResizableWrapper from './ResizableWrapper'; // Import custom Resizable wrapper
import 'react-resizable/css/styles.css';
import './App.css'; // Import CSS for transitions

// Import Roboto font if using npm
import '@fontsource/roboto/400.css';
import '@fontsource/roboto/700.css';

const CustomSwitch = styled(Switch)(({ theme }) => ({
  width: 42,
  height: 26,
  padding: 0,
  display: 'flex',
  '&:hover': {
    '& .MuiSwitch-thumb': {
      boxShadow: '0 0 8px rgba(0, 0, 0, 0.3)',
    },
    '& .MuiSwitch-track': {
      backgroundColor: theme.palette.mode === 'dark' ? '#6b6b6b' : '#d0d0d0',
    },
  },
  '& .MuiSwitch-switchBase': {
    padding: 1,
    '&.Mui-checked': {
      transform: 'translateX(16px)',
      color: '#fff',
      '& + .MuiSwitch-track': {
        opacity: 1,
        backgroundColor: theme.palette.mode === 'dark' ? '#8796A5' : '#aab4be',
      },
    },
  },
  '& .MuiSwitch-thumb': {
    width: 24,
    height: 24,
    boxShadow: 'none',
  },
  '& .MuiSwitch-track': {
    borderRadius: 13,
    opacity: 1,
    backgroundColor: theme.palette.mode === 'dark' ? '#8796A5' : '#aab4be',
    boxSizing: 'border-box',
  },
}));

const lightTheme = createTheme({
  palette: {
    mode: 'light',
    primary: { main: '#6750A4' },
    background: { default: '#f9f7fc', paper: '#FFFFFF' },
    text: { primary: '#1C1B1F' },
  },
  typography: {
    fontFamily: 'Roboto, sans-serif',
    fontWeightBold: 700,
    h4: { fontWeight: 700, fontSize: '1.5rem' },
    h6: { fontWeight: 700, fontSize: '1.25rem' },
    body1: { fontWeight: 400, fontSize: '1rem' },
  },
  shape: { borderRadius: 12 },
  components: {
    MuiButton: { styleOverrides: { root: { borderRadius: '12px', textTransform: 'none' } } },
    MuiPaper: { styleOverrides: { root: { padding: '16px', borderRadius: '12px', boxShadow: 'none' } } },
    MuiAppBar: { styleOverrides: { root: { borderRadius: '12px' } } },
  },
});

const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: { main: '#D0BCFF' },
    background: { default: '#121212', paper: '#1E1E1E' },
    text: { primary: '#E6E1E5' },
  },
  typography: {
    fontFamily: 'Roboto, sans-serif',
    fontWeightBold: 700,
    h4: { fontWeight: 700, fontSize: '1.5rem' },
    h6: { fontWeight: 700, fontSize: '1.25rem' },
    body1: { fontWeight: 400, fontSize: '1rem' },
  },
  shape: { borderRadius: 12 },
  components: {
    MuiButton: { styleOverrides: { root: { borderRadius: '12px', textTransform: 'none' } } },
    MuiPaper: { styleOverrides: { root: { padding: '16px', borderRadius: '12px' } } },
    MuiAppBar: { styleOverrides: { root: { borderRadius: '12px' } } },
  },
});

function App() {
  const [model, setModel] = useState('');
  const [label, setLabel] = useState('');
  const [topK, setTopK] = useState(2);
  const [threshold, setThreshold] = useState(0.5);
  const [recommendations, setRecommendations] = useState([]);
  const [labels, setLabels] = useState([]);
  const [models, setModels] = useState([]);
  const [darkMode, setDarkMode] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');

  const prefersDarkMode = useMediaQuery('(prefers-color-scheme: dark)');

  useEffect(() => {
    setDarkMode(prefersDarkMode);
  }, [prefersDarkMode]);

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
    if (!model || !label) {
      setErrorMessage('Please select both a model and a label before running the model.');
      return;
    }

    try {
      const response = await axios.post('http://localhost:5000/predict', {
        model, label, topK, threshold
      });
      console.log('Received recommendations:', response.data.recommendations); // Debug log
      setRecommendations(response.data.recommendations);
      setErrorMessage(''); // Clear error message on success
    } catch (error) {
      console.error('Error running model:', error);
      if (error.response) {
        console.error('Error response data:', error.response.data);
      }
      setErrorMessage('Error running model. Please try again.');
    }
  };

  const handleToggleDarkMode = () => {
    setDarkMode(!darkMode);
  };

  const theme = useMemo(() => (darkMode ? darkTheme : lightTheme), [darkMode]);

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ flexGrow: 1 }}>
        <AppBar position="static" color="background" sx={{ borderRadius: 2 }}>
          <Toolbar>
            <Avatar alt="App Icon" src="coreRec.svg" sx={{ mr: 2 }} />
            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
              CoreRec
            </Typography>
            <FormControlLabel
              control={<CustomSwitch checked={darkMode} onChange={handleToggleDarkMode} />}
              sx={{ mr: 2 }}
              label={darkMode ? "Light Mode" : "Dark Mode"}
            />
          </Toolbar>
        </AppBar>
        <Container sx={{ mt: 4 }}>
          {errorMessage && (
            <Card sx={{ mb: 4, backgroundColor: 'error.main', color: 'error.contrastText' }}>
              <CardContent>
                <Typography variant="body1">{errorMessage}</Typography>
              </CardContent>
            </Card>
          )}
          <Paper elevation={3} sx={{ mb: 4, borderRadius: 2 }}>
            <Typography variant="h4" gutterBottom>
              Test_A
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12}>
                <Select
                  value={model}
                  onChange={(e) => setModel(e.target.value)}
                  displayEmpty
                  fullWidth
                  variant="outlined"
                >
                  <MenuItem value="" disabled>Select model</MenuItem>
                  {models.map((mdl, index) => (
                    <MenuItem key={index} value={mdl.value}>{mdl.label}</MenuItem>
                  ))}
                </Select>
              </Grid>
              <Grid item xs={12}>
                <Select
                  value={label}
                  onChange={(e) => setLabel(e.target.value)}
                  displayEmpty
                  fullWidth
                  variant="outlined"
                >
                  <MenuItem value="" disabled>Select node label</MenuItem>
                  {labels.map((lbl, index) => (
                    <MenuItem key={index} value={lbl}>{lbl}</MenuItem>
                  ))}
                </Select>
              </Grid>
              <Grid item xs={12}>
                <Typography gutterBottom>Top K</Typography>
                <Slider
                  value={topK}
                  onChange={(e, val) => setTopK(val)}
                  min={1}
                  max={10}
                  marks
                  valueLabelDisplay="auto"
                  sx={{ mb: 2 }}
                />
              </Grid>
              <Grid item xs={12}>
                <Typography gutterBottom>Threshold</Typography>
                <Slider
                  value={threshold}
                  onChange={(e, val) => setThreshold(val)}
                  min={0}
                  max={1}
                  step={0.1}
                  marks
                  valueLabelDisplay="auto"
                  sx={{ mb: 2 }}
                />
              </Grid>
              <Grid item xs={12}>
                <Button
                  variant="contained"
                  color="primary"
                  onClick={handleRunModel}
                  fullWidth
                  sx={{ mb: 2, borderRadius: 2, boxShadow: 'none' }}
                >
                  Run Model
                </Button>
              </Grid>
            </Grid>
          </Paper>
          <DraggableWrapper>
            <ResizableWrapper width={298} height={276} minConstraints={[200, 100]} maxConstraints={[600, 400]}>
              <Paper elevation={3} sx={{ cursor: 'move', height: '100%', borderRadius: 2 }}>
              <Typography variant="h5" gutterBottom sx={{ fontWeight: 'bold', color: 'text.primary' }}>
                Recommended nodes:
              </Typography>
              <Divider sx={{ my: 2 }} />
              <Box sx={{ p: 2, height: 'calc(100% - 48px)', overflow: 'auto' }}>
                {recommendations.map((rec, index) => (
                  <Typography key={index} variant="h6" gutterBottom>
                    {rec}
                  </Typography>
                ))}
              </Box>
              </Paper>
            </ResizableWrapper>
          </DraggableWrapper>
        </Container>
      </Box>
    </ThemeProvider>
  );
}

export default App;