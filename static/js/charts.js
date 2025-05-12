// charts.js - Functions for creating charts and visualizations

/**
 * Creates a radar chart to visualize input parameters
 * @param {string} canvasId - The ID of the canvas element
 * @param {object} data - Object containing input parameter values
 */
function createInputParametersChart(canvasId, data) {
    const canvas = document.getElementById(canvasId);
    
    if (!canvas) {
        console.error(`Canvas element with ID "${canvasId}" not found.`);
        return;
    }
    
    // Normalize data for display (different scales)
    const normalizedData = {
        nitrogen: (data.nitrogen / 140) * 100,
        phosphorus: (data.phosphorus / 145) * 100,
        potassium: (data.potassium / 205) * 100,
        temperature: ((data.temperature - 8) / (44 - 8)) * 100,
        humidity: data.humidity,
        ph: ((data.ph - 3.5) / (10 - 3.5)) * 100,
        rainfall: (data.rainfall / 300) * 10  // Cap at 3000mm, display as percentage
    };
    
    // Create radar chart
    const ctx = canvas.getContext('2d');
    const chart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: [
                'Nitrogen', 
                'Phosphorus', 
                'Potassium', 
                'Temperature', 
                'Humidity', 
                'pH', 
                'Rainfall'
            ],
            datasets: [{
                label: 'Input Parameters',
                data: [
                    normalizedData.nitrogen,
                    normalizedData.phosphorus,
                    normalizedData.potassium,
                    normalizedData.temperature,
                    normalizedData.humidity,
                    normalizedData.ph,
                    normalizedData.rainfall
                ],
                backgroundColor: 'rgba(76, 175, 80, 0.2)',
                borderColor: 'rgba(76, 175, 80, 1)',
                pointBackgroundColor: 'rgba(76, 175, 80, 1)',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: 'rgba(76, 175, 80, 1)'
            }]
        },
        options: {
            scales: {
                r: {
                    angleLines: {
                        display: true
                    },
                    suggestedMin: 0,
                    suggestedMax: 100
                }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const originalValue = data[context.label.toLowerCase()];
                            let unit = '';
                            switch(context.label.toLowerCase()) {
                                case 'nitrogen':
                                case 'phosphorus':
                                case 'potassium':
                                    unit = ' kg/ha';
                                    break;
                                case 'temperature':
                                    unit = ' °C';
                                    break;
                                case 'humidity':
                                    unit = '%';
                                    break;
                                case 'ph':
                                    unit = ' pH';
                                    break;
                                case 'rainfall':
                                    unit = ' mm';
                                    break;
                            }
                            return `${context.label}: ${originalValue}${unit}`;
                        }
                    }
                }
            }
        }
    });
    
    return chart;
}

/**
 * Creates an empty chart as a placeholder before data is entered
 * @param {string} canvasId - The ID of the canvas element
 */
function createEmptyChart(canvasId) {
    const canvas = document.getElementById(canvasId);
    
    if (!canvas) {
        console.error(`Canvas element with ID "${canvasId}" not found.`);
        return;
    }
    
    const ctx = canvas.getContext('2d');
    const chart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: [
                'Nitrogen', 
                'Phosphorus', 
                'Potassium', 
                'Temperature', 
                'Humidity', 
                'pH', 
                'Rainfall'
            ],
            datasets: [{
                label: 'Enter Data',
                data: [0, 0, 0, 0, 0, 0, 0],
                backgroundColor: 'rgba(200, 200, 200, 0.2)',
                borderColor: 'rgba(200, 200, 200, 1)',
                pointBackgroundColor: 'rgba(200, 200, 200, 1)',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: 'rgba(200, 200, 200, 1)'
            }]
        },
        options: {
            scales: {
                r: {
                    angleLines: {
                        display: true
                    },
                    suggestedMin: 0,
                    suggestedMax: 100,
                    ticks: {
                        display: false
                    }
                }
            },
            plugins: {
                tooltip: {
                    enabled: false
                }
            }
        }
    });
    
    return chart;
}

/**
 * Creates a bar chart for comparing crop suitability scores
 * @param {string} canvasId - The ID of the canvas element
 * @param {object} scores - Object with crop names and scores
 */
function createCropComparisonChart(canvasId, scores) {
    const canvas = document.getElementById(canvasId);
    
    if (!canvas) {
        console.error(`Canvas element with ID "${canvasId}" not found.`);
        return;
    }
    
    // Sort scores
    const sortedEntries = Object.entries(scores).sort((a, b) => b[1] - a[1]);
    const cropNames = sortedEntries.map(entry => entry[0]);
    const cropScores = sortedEntries.map(entry => entry[1]);
    
    // Create colors array with primary color for top crop
    const colors = cropScores.map((score, index) => 
        index === 0 ? 'rgba(76, 175, 80, 0.8)' : 'rgba(76, 175, 80, 0.4)');
    
    const ctx = canvas.getContext('2d');
    const chart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: cropNames,
            datasets: [{
                label: 'Suitability Score',
                data: cropScores,
                backgroundColor: colors,
                borderColor: colors.map(color => color.replace('0.4', '1').replace('0.8', '1')),
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `Suitability: ${context.raw}%`;
                        }
                    }
                }
            }
        }
    });
    
    return chart;
}

/**
 * Creates a bar chart for comparing model predictions
 * @param {string} canvasId - The ID of the canvas element
 * @param {object} modelPredictions - Object with model names and predictions
 * @param {string} finalPrediction - The final prediction based on majority voting
 */
function createModelComparisonChart(canvasId, modelPredictions, finalPrediction) {
    const canvas = document.getElementById(canvasId);
    
    if (!canvas) {
        console.error(`Canvas element with ID "${canvasId}" not found.`);
        return;
    }
    
    // Handle empty or invalid modelPredictions
    if (!modelPredictions || Object.keys(modelPredictions).length === 0) {
        console.warn(`No model predictions provided for chart "${canvasId}".`);
        const ctx = canvas.getContext('2d');
        ctx.font = '14px Arial';
        ctx.fillStyle = '#777';
        ctx.textAlign = 'center';
        ctx.fillText('No model predictions available', canvas.width / 2, canvas.height / 2);
        return;
    }
    
    // Transform predictions into a format for visualization
    // Count occurrences of each prediction
    const predictionCounts = {};
    const modelAssignments = {}; // Track which models predict each crop
    Object.entries(modelPredictions).forEach(([model, prediction]) => {
        predictionCounts[prediction] = (predictionCounts[prediction] || 0) + 1;
        if (!modelAssignments[prediction]) {
            modelAssignments[prediction] = [];
        }
        modelAssignments[prediction].push(model);
    });
    
    const uniquePredictions = Object.keys(predictionCounts);
    const counts = Object.values(predictionCounts);
    
    // Create colors array with primary color for final prediction
    const colors = uniquePredictions.map(prediction => 
        prediction === finalPrediction ? 'rgba(76, 175, 80, 0.8)' : 'rgba(153, 153, 153, 0.4)');
    
    const ctx = canvas.getContext('2d');
    const chart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: uniquePredictions,
            datasets: [{
                label: 'Number of Models',
                data: counts,
                backgroundColor: colors,
                borderColor: colors.map(color => color.replace('0.4', '1').replace('0.8', '1')),
                borderWidth: 1
            }]
        },
        options: {
            indexAxis: 'y',
            plugins: {
                title: {
                    display: true,
                    text: 'Model Predictions Distribution'
                },
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const prediction = context.label;
                            const count = context.raw;
                            const models = modelAssignments[prediction] || [];
                            const modelList = models.join(', ');
                            const countText = count === 1 ? `${count} model` : `${count} models`;
                            return `${countText}: ${modelList}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    beginAtZero: true,
                    max: Object.keys(modelPredictions).length,
                    title: {
                        display: true,
                        text: 'Number of Models'
                    },
                    ticks: {
                        stepSize: 1
                    }
                }
            }
        }
    });
    
    return chart;
}

/**
 * Creates a gauge chart to visualize a value within a range
 * @param {string} canvasId - The ID of the canvas element
 * @param {number} value - The value to display
 * @param {number} min - Minimum value of the range
 * @param {number} max - Maximum value of the range
 * @param {string} label - Label for the gauge
 */
function createGaugeChart(canvasId, value, min, max, label) {
    const canvas = document.getElementById(canvasId);
    
    if (!canvas) {
        console.error(`Canvas element with ID "${canvasId}" not found.`);
        return;
    }
    
    // Calculate percentage and colors
    const percentage = ((value - min) / (max - min)) * 100;
    let color = '#f44336'; // Red by default
    
    if (percentage > 80) {
        color = '#4caf50'; // Green for high values
    } else if (percentage > 50) {
        color = '#ff9800'; // Orange for medium values
    }
    
    const ctx = canvas.getContext('2d');
    const chart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            datasets: [{
                data: [percentage, 100 - percentage],
                backgroundColor: [color, '#eceff1'],
                borderWidth: 0
            }]
        },
        options: {
            circumference: 180,
            rotation: 270,
            cutout: '70%',
            plugins: {
                tooltip: {
                    enabled: false
                },
                legend: {
                    display: false
                }
            },
            layout: {
                padding: {
                    bottom: 10
                }
            }
        }
    });
    
    // Add text in center
    const fontSizeValue = canvas.width / 10;
    const fontSizeLabel = canvas.width / 14;
    
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    
    // Value text
    ctx.font = `bold ${fontSizeValue}px Arial`;
    ctx.fillStyle = '#333';
    ctx.fillText(value, canvas.width / 2, canvas.height / 2);
    
    // Label text
    ctx.font = `${fontSizeLabel}px Arial`;
    ctx.fillStyle = '#777';
    ctx.fillText(label, canvas.width / 2, canvas.height / 2 + fontSizeValue);
    
    return chart;
}