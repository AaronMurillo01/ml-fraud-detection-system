/**
 * Advanced Analytics Suite for FraudGuard AI
 * Implements predictive forecasting, trend analysis, and advanced visualizations
 */

// Advanced Analytics Engine
class AdvancedAnalyticsEngine {
  constructor() {
    this.charts = {};
    this.data = {};
    this.models = {};
    this.realTimeInterval = null;
    this.forecastModel = null;
  }

  // Initialize all analytics components
  async initialize() {
    console.log("🚀 Initializing Advanced Analytics Suite...");

    await this.loadHistoricalData();
    await this.initializeForecastingModel();
    this.createAllCharts();
    this.startRealTimeUpdates();
    this.generateAIInsights();

    console.log("✅ Advanced Analytics Suite initialized successfully!");
  }

  // Load and generate synthetic historical data
  async loadHistoricalData() {
    console.log("📊 Loading historical fraud data...");

    const now = new Date();
    this.data = {
      historical: [],
      forecast: [],
      anomalies: [],
      trends: [],
      geographic: [],
      modelPerformance: [],
      featureImportance: [],
    };

    // Generate 90 days of historical data with realistic patterns
    for (let i = 89; i >= 0; i--) {
      const date = new Date(now.getTime() - i * 24 * 60 * 60 * 1000);
      const dayOfWeek = date.getDay();
      const hour = date.getHours();

      // Base fraud rate with realistic patterns
      let baseRate = 0.023; // 2.3% base fraud rate

      // Weekend effect (higher fraud on weekends)
      if (dayOfWeek === 0 || dayOfWeek === 6) {
        baseRate *= 1.15;
      }

      // Time-of-day effect (higher fraud at night)
      if (hour >= 22 || hour <= 6) {
        baseRate *= 1.3;
      }

      // Seasonal trends
      const seasonality = Math.sin((i / 365) * 2 * Math.PI) * 0.005;

      // Random noise
      const noise = (Math.random() - 0.5) * 0.01;

      // Occasional fraud spikes
      const spike = Math.random() < 0.05 ? Math.random() * 0.02 : 0;

      const fraudRate = Math.max(0, baseRate + seasonality + noise + spike);
      const transactions = Math.floor(
        800 +
          Math.random() * 400 +
          (dayOfWeek === 0 || dayOfWeek === 6 ? -200 : 0)
      );
      const fraudCount = Math.floor(fraudRate * transactions);

      this.data.historical.push({
        date: date,
        fraudRate: fraudRate,
        transactions: transactions,
        fraudCount: fraudCount,
        accuracy: 0.94 + Math.random() * 0.06,
        precision: 0.91 + Math.random() * 0.08,
        recall: 0.88 + Math.random() * 0.1,
        f1Score: 0.89 + Math.random() * 0.08,
      });
    }

    // Generate anomalies
    const anomalyIndices = [5, 12, 23, 34, 45, 67, 78, 85];
    anomalyIndices.forEach((index) => {
      if (index < this.data.historical.length) {
        const point = this.data.historical[index];
        this.data.anomalies.push({
          date: point.date,
          value: point.fraudRate,
          severity: point.fraudRate > 0.04 ? "high" : "medium",
          description: `Anomalous fraud pattern detected: ${(
            point.fraudRate * 100
          ).toFixed(2)}% fraud rate`,
          confidence: 0.85 + Math.random() * 0.1,
        });
      }
    });

    // Generate feature importance data
    this.data.featureImportance = [
      { feature: "Transaction Amount", importance: 0.23, change: 0.02 },
      {
        feature: "Time Since Last Transaction",
        importance: 0.19,
        change: -0.01,
      },
      { feature: "Merchant Category", importance: 0.16, change: 0.03 },
      { feature: "Geographic Location", importance: 0.14, change: 0.01 },
      { feature: "Payment Method", importance: 0.12, change: -0.02 },
      { feature: "User Behavior Score", importance: 0.1, change: 0.04 },
      { feature: "IP Address Risk", importance: 0.06, change: 0.01 },
    ];

    // Generate model performance comparison
    this.data.modelPerformance = [
      {
        model: "XGBoost Ensemble",
        accuracy: 0.968,
        precision: 0.943,
        recall: 0.921,
        f1: 0.932,
      },
      {
        model: "Random Forest",
        accuracy: 0.952,
        precision: 0.928,
        recall: 0.905,
        f1: 0.916,
      },
      {
        model: "Neural Network",
        accuracy: 0.945,
        precision: 0.919,
        recall: 0.898,
        f1: 0.908,
      },
      {
        model: "Logistic Regression",
        accuracy: 0.923,
        precision: 0.891,
        recall: 0.876,
        f1: 0.883,
      },
      {
        model: "SVM",
        accuracy: 0.918,
        precision: 0.885,
        recall: 0.869,
        f1: 0.877,
      },
    ];

    console.log("✅ Historical data loaded successfully");
  }

  // Initialize forecasting model using simple linear regression
  async initializeForecastingModel() {
    console.log("🔮 Initializing forecasting model...");

    // Prepare training data
    const trainingData = this.data.historical.map((point, index) => [
      index,
      point.fraudRate,
    ]);

    // Simple trend analysis
    const n = trainingData.length;
    const sumX = trainingData.reduce((sum, point) => sum + point[0], 0);
    const sumY = trainingData.reduce((sum, point) => sum + point[1], 0);
    const sumXY = trainingData.reduce(
      (sum, point) => sum + point[0] * point[1],
      0
    );
    const sumXX = trainingData.reduce(
      (sum, point) => sum + point[0] * point[0],
      0
    );

    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;

    this.forecastModel = { slope, intercept };

    // Generate forecast for next 14 days
    const lastIndex = this.data.historical.length - 1;
    for (let i = 1; i <= 14; i++) {
      const date = new Date(
        this.data.historical[lastIndex].date.getTime() + i * 24 * 60 * 60 * 1000
      );
      const predicted =
        this.forecastModel.slope * (lastIndex + i) +
        this.forecastModel.intercept;
      const uncertainty = 0.005 + i * 0.001; // Increasing uncertainty over time

      this.data.forecast.push({
        date: date,
        predicted: Math.max(0, predicted),
        upperBound: Math.max(0, predicted + uncertainty),
        lowerBound: Math.max(0, predicted - uncertainty),
        confidence: Math.max(0.5, 0.95 - i * 0.03),
      });
    }

    console.log("✅ Forecasting model initialized");
  }

  // Create all visualization charts
  createAllCharts() {
    console.log("📈 Creating visualization charts...");

    this.createForecastChart();
    this.createTrendChart();
    this.createAnomalyChart();
    this.createModelComparisonChart();
    this.createFeatureImportanceChart();
    this.createTimeSeriesChart();
    this.createGeographicChart();

    console.log("✅ All charts created successfully");
  }

  // Create predictive forecasting chart
  createForecastChart() {
    const ctx = document.getElementById("forecastChart").getContext("2d");

    const historicalData = this.data.historical.slice(-30).map((point) => ({
      x: point.date,
      y: point.fraudRate * 100,
    }));

    const forecastData = this.data.forecast.map((point) => ({
      x: point.date,
      y: point.predicted * 100,
    }));

    const upperBoundData = this.data.forecast.map((point) => ({
      x: point.date,
      y: point.upperBound * 100,
    }));

    const lowerBoundData = this.data.forecast.map((point) => ({
      x: point.date,
      y: point.lowerBound * 100,
    }));

    this.charts.forecast = new Chart(ctx, {
      type: "line",
      data: {
        datasets: [
          {
            label: "Historical Fraud Rate",
            data: historicalData,
            borderColor: "#6366f1",
            backgroundColor: "rgba(99, 102, 241, 0.1)",
            borderWidth: 2,
            fill: false,
            tension: 0.4,
          },
          {
            label: "Predicted Fraud Rate",
            data: forecastData,
            borderColor: "#8b5cf6",
            backgroundColor: "rgba(139, 92, 246, 0.1)",
            borderWidth: 2,
            borderDash: [5, 5],
            fill: false,
            tension: 0.4,
          },
          {
            label: "Confidence Interval",
            data: upperBoundData.concat(lowerBoundData.reverse()),
            borderColor: "rgba(139, 92, 246, 0.3)",
            backgroundColor: "rgba(139, 92, 246, 0.1)",
            borderWidth: 1,
            fill: true,
            pointRadius: 0,
            tension: 0.4,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        layout: {
          padding: {
            top: 10,
            right: 15,
            bottom: 10,
            left: 15,
          },
        },
        plugins: {
          title: {
            display: true,
            text: "Fraud Rate Prediction with Confidence Intervals",
            font: { size: 14, weight: "bold" },
            padding: {
              bottom: 15,
            },
          },
          legend: {
            position: "top",
            align: "center",
            labels: {
              boxWidth: 12,
              padding: 12,
              font: {
                size: 11,
              },
            },
          },
        },
        scales: {
          x: {
            type: "time",
            time: {
              unit: "day",
              displayFormats: {
                day: "MMM dd",
              },
            },
            title: {
              display: true,
              text: "Date",
              font: {
                size: 11,
              },
            },
            ticks: {
              font: {
                size: 10,
              },
              maxRotation: 0,
              minRotation: 0,
              maxTicksLimit: 6,
            },
            grid: {
              color: "rgba(0, 0, 0, 0.1)",
            },
          },
          y: {
            title: {
              display: true,
              text: "Fraud Rate (%)",
              font: {
                size: 11,
              },
            },
            min: 0,
            ticks: {
              font: {
                size: 10,
              },
              callback: function (value) {
                return value.toFixed(1) + "%";
              },
            },
            grid: {
              color: "rgba(0, 0, 0, 0.1)",
            },
          },
        },
        interaction: {
          intersect: false,
          mode: "index",
        },
      },
    });
  }

  // Create trend analysis chart
  createTrendChart() {
    const ctx = document.getElementById("trendChart").getContext("2d");

    // Aggregate data by week
    const weeklyData = this.aggregateByWeek(this.data.historical);

    this.charts.trend = new Chart(ctx, {
      type: "line",
      data: {
        labels: weeklyData.map((point) => point.week),
        datasets: [
          {
            label: "Fraud Rate Trend",
            data: weeklyData.map((point) => point.avgFraudRate * 100),
            borderColor: "#10b981",
            backgroundColor: "rgba(16, 185, 129, 0.1)",
            borderWidth: 3,
            fill: true,
            tension: 0.4,
          },
          {
            label: "Transaction Volume",
            data: weeklyData.map((point) => point.totalTransactions / 100),
            borderColor: "#f59e0b",
            backgroundColor: "rgba(245, 158, 11, 0.1)",
            borderWidth: 2,
            fill: false,
            yAxisID: "y1",
            tension: 0.4,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        layout: {
          padding: {
            top: 10,
            right: 15,
            bottom: 10,
            left: 15,
          },
        },
        plugins: {
          title: {
            display: true,
            text: "Weekly Fraud Trends & Transaction Volume",
            font: { size: 14, weight: "bold" },
            padding: {
              bottom: 15,
            },
          },
          legend: {
            position: "top",
            align: "center",
            labels: {
              boxWidth: 12,
              padding: 12,
              font: {
                size: 11,
              },
            },
          },
        },
        scales: {
          x: {
            ticks: {
              font: {
                size: 10,
              },
              maxRotation: 0,
              minRotation: 0,
              maxTicksLimit: 8,
            },
            grid: {
              color: "rgba(0, 0, 0, 0.1)",
            },
          },
          y: {
            type: "linear",
            display: true,
            position: "left",
            title: {
              display: true,
              text: "Fraud Rate (%)",
              font: {
                size: 11,
              },
            },
            ticks: {
              font: {
                size: 10,
              },
              callback: function (value) {
                return value.toFixed(1) + "%";
              },
            },
            grid: {
              color: "rgba(0, 0, 0, 0.1)",
            },
          },
          y1: {
            type: "linear",
            display: true,
            position: "right",
            title: {
              display: true,
              text: "Transactions (hundreds)",
              font: {
                size: 11,
              },
            },
            ticks: {
              font: {
                size: 10,
              },
            },
            grid: {
              drawOnChartArea: false,
            },
          },
        },
      },
    });
  }

  // Aggregate historical data by week
  aggregateByWeek(data) {
    const weeks = {};

    data.forEach((point) => {
      const weekStart = new Date(point.date);
      weekStart.setDate(weekStart.getDate() - weekStart.getDay());
      const weekKey = weekStart.toISOString().split("T")[0];

      if (!weeks[weekKey]) {
        weeks[weekKey] = {
          week: weekStart,
          fraudRates: [],
          transactions: [],
        };
      }

      weeks[weekKey].fraudRates.push(point.fraudRate);
      weeks[weekKey].transactions.push(point.transactions);
    });

    return Object.values(weeks).map((week) => ({
      week: week.week,
      avgFraudRate:
        week.fraudRates.reduce((sum, rate) => sum + rate, 0) /
        week.fraudRates.length,
      totalTransactions: week.transactions.reduce(
        (sum, count) => sum + count,
        0
      ),
    }));
  }

  // Create anomaly detection chart
  createAnomalyChart() {
    const ctx = document.getElementById("anomalyChart").getContext("2d");

    const recentData = this.data.historical.slice(-30);
    const anomalies = this.data.anomalies.filter(
      (anomaly) => anomaly.date >= recentData[0].date
    );

    this.charts.anomaly = new Chart(ctx, {
      type: "scatter",
      data: {
        datasets: [
          {
            label: "Normal Transactions",
            data: recentData
              .filter(
                (point) =>
                  !anomalies.some(
                    (anomaly) => anomaly.date.getTime() === point.date.getTime()
                  )
              )
              .map((point) => ({
                x: point.date,
                y: point.fraudRate * 100,
              })),
            backgroundColor: "rgba(16, 185, 129, 0.6)",
            borderColor: "#10b981",
            borderWidth: 1,
            pointRadius: 4,
          },
          {
            label: "Anomalies",
            data: anomalies.map((anomaly) => ({
              x: anomaly.date,
              y: anomaly.value * 100,
            })),
            backgroundColor: "rgba(239, 68, 68, 0.8)",
            borderColor: "#ef4444",
            borderWidth: 2,
            pointRadius: 8,
            pointHoverRadius: 10,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          title: {
            display: true,
            text: "Anomaly Detection in Fraud Patterns",
            font: { size: 16, weight: "bold" },
          },
          tooltip: {
            callbacks: {
              afterLabel: function (context) {
                if (context.datasetIndex === 1) {
                  const anomaly = anomalies[context.dataIndex];
                  return `Severity: ${anomaly.severity}\nConfidence: ${(
                    anomaly.confidence * 100
                  ).toFixed(1)}%`;
                }
                return "";
              },
            },
          },
        },
        scales: {
          x: {
            type: "time",
            time: {
              unit: "day",
              displayFormats: {
                day: "MMM dd",
              },
            },
            title: {
              display: true,
              text: "Date",
            },
          },
          y: {
            title: {
              display: true,
              text: "Fraud Rate (%)",
            },
          },
        },
      },
    });
  }

  // Create model performance comparison chart
  createModelComparisonChart() {
    const ctx = document
      .getElementById("modelComparisonChart")
      .getContext("2d");

    this.charts.modelComparison = new Chart(ctx, {
      type: "radar",
      data: {
        labels: ["Accuracy", "Precision", "Recall", "F1-Score"],
        datasets: this.data.modelPerformance
          .slice(0, 3)
          .map((model, index) => ({
            label: model.model,
            data: [model.accuracy, model.precision, model.recall, model.f1],
            borderColor: ["#6366f1", "#8b5cf6", "#06b6d4"][index],
            backgroundColor: [
              `rgba(99, 102, 241, 0.2)`,
              `rgba(139, 92, 246, 0.2)`,
              `rgba(6, 182, 212, 0.2)`,
            ][index],
            borderWidth: 2,
            pointRadius: 4,
          })),
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          title: {
            display: true,
            text: "Model Performance Comparison",
            font: { size: 16, weight: "bold" },
          },
        },
        scales: {
          r: {
            beginAtZero: true,
            max: 1,
            ticks: {
              callback: function (value) {
                return (value * 100).toFixed(0) + "%";
              },
            },
          },
        },
      },
    });
  }

  // Create feature importance chart
  createFeatureImportanceChart() {
    const ctx = document
      .getElementById("featureImportanceChart")
      .getContext("2d");

    this.charts.featureImportance = new Chart(ctx, {
      type: "bar",
      data: {
        labels: this.data.featureImportance.map((item) => item.feature),
        datasets: [
          {
            label: "Feature Importance",
            data: this.data.featureImportance.map((item) => item.importance),
            backgroundColor: this.data.featureImportance.map((item, index) => {
              const colors = [
                "#6366f1",
                "#8b5cf6",
                "#06b6d4",
                "#10b981",
                "#f59e0b",
                "#ef4444",
                "#8b5cf6",
              ];
              return colors[index % colors.length];
            }),
            borderColor: this.data.featureImportance.map((item, index) => {
              const colors = [
                "#4f46e5",
                "#7c3aed",
                "#0891b2",
                "#059669",
                "#d97706",
                "#dc2626",
                "#7c3aed",
              ];
              return colors[index % colors.length];
            }),
            borderWidth: 1,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        layout: {
          padding: {
            top: 10,
            right: 15,
            bottom: 10,
            left: 15,
          },
        },
        plugins: {
          title: {
            display: true,
            text: "Feature Importance Analysis",
            font: { size: 14, weight: "bold" },
            padding: {
              bottom: 15,
            },
          },
          legend: {
            display: false,
          },
        },
        scales: {
          y: {
            beginAtZero: true,
            title: {
              display: true,
              text: "Importance Score",
              font: {
                size: 11,
              },
            },
            ticks: {
              font: {
                size: 10,
              },
              callback: function (value) {
                return (value * 100).toFixed(0) + "%";
              },
            },
            grid: {
              color: "rgba(0, 0, 0, 0.1)",
            },
          },
          x: {
            ticks: {
              font: {
                size: 9,
              },
              maxRotation: 0,
              minRotation: 0,
              callback: function (value, index) {
                const label = this.getLabelForValue(value);
                // Truncate long labels
                return label.length > 12
                  ? label.substring(0, 12) + "..."
                  : label;
              },
            },
            grid: {
              color: "rgba(0, 0, 0, 0.1)",
            },
          },
        },
      },
    });
  }

  // Create advanced time series chart using Plotly
  createTimeSeriesChart() {
    const container = document.getElementById("timeSeriesChart");

    const traces = [
      {
        x: this.data.historical.map((point) => point.date),
        y: this.data.historical.map((point) => point.fraudRate * 100),
        type: "scatter",
        mode: "lines",
        name: "Fraud Rate",
        line: { color: "#6366f1", width: 2 },
      },
      {
        x: this.data.historical.map((point) => point.date),
        y: this.data.historical.map((point) => point.accuracy * 100),
        type: "scatter",
        mode: "lines",
        name: "Model Accuracy",
        yaxis: "y2",
        line: { color: "#10b981", width: 2 },
      },
    ];

    const layout = {
      title: {
        text: "Advanced Time Series Analysis",
        font: { size: 18, family: "Inter" },
      },
      xaxis: {
        title: "Date",
        type: "date",
      },
      yaxis: {
        title: "Fraud Rate (%)",
        side: "left",
      },
      yaxis2: {
        title: "Model Accuracy (%)",
        side: "right",
        overlaying: "y",
      },
      hovermode: "x unified",
      showlegend: true,
      plot_bgcolor: "rgba(0,0,0,0)",
      paper_bgcolor: "rgba(0,0,0,0)",
    };

    const config = {
      responsive: true,
      displayModeBar: true,
      modeBarButtonsToRemove: ["pan2d", "lasso2d", "select2d"],
    };

    Plotly.newPlot(container, traces, layout, config);
  }

  // Create geographic fraud distribution chart
  createGeographicChart() {
    const container = document.getElementById("geoChart");

    // Generate synthetic geographic data
    const geoData = [
      { country: "Brazil", fraudRate: 3.1, transactions: 4500 },
      { country: "Australia", fraudRate: 2.3, transactions: 3500 },
      { country: "USA", fraudRate: 2.1, transactions: 15000 },
      { country: "France", fraudRate: 1.9, transactions: 5500 },
      { country: "UK", fraudRate: 1.8, transactions: 8000 },
      { country: "Canada", fraudRate: 1.7, transactions: 4000 },
      { country: "Germany", fraudRate: 1.5, transactions: 6000 },
      { country: "Japan", fraudRate: 1.2, transactions: 7000 },
    ];

    // Sort by fraud rate descending
    geoData.sort((a, b) => b.fraudRate - a.fraudRate);

    // Create color array based on fraud rate
    const colors = geoData.map((item) => {
      if (item.fraudRate >= 2.5) return "#ef4444"; // High risk - red
      if (item.fraudRate >= 2.0) return "#f59e0b"; // Medium risk - amber
      if (item.fraudRate >= 1.5) return "#3b82f6"; // Low-medium risk - blue
      return "#10b981"; // Low risk - green
    });

    const trace = {
      type: "bar",
      x: geoData.map((item) => item.country),
      y: geoData.map((item) => item.fraudRate),
      text: geoData.map(
        (item) =>
          `${
            item.fraudRate
          }%<br>${item.transactions.toLocaleString()} transactions`
      ),
      textposition: "none",
      hovertemplate:
        "<b>%{x}</b><br>" +
        "Fraud Rate: %{y}%<br>" +
        "Transactions: %{customdata:,}<br>" +
        "<extra></extra>",
      customdata: geoData.map((item) => item.transactions),
      marker: {
        color: colors,
        line: {
          color: "rgba(0,0,0,0.1)",
          width: 1,
        },
      },
    };

    const layout = {
      title: {
        text: "Global Fraud Distribution by Country",
        font: { size: 16, family: "Inter", weight: 600 },
      },
      xaxis: {
        title: "Country",
        tickangle: -45,
        tickfont: { size: 11 },
      },
      yaxis: {
        title: "Fraud Rate (%)",
        tickfont: { size: 11 },
        gridcolor: "rgba(0,0,0,0.05)",
      },
      plot_bgcolor: "rgba(0,0,0,0)",
      paper_bgcolor: "rgba(0,0,0,0)",
      margin: { l: 60, r: 30, t: 60, b: 100 },
      showlegend: false,
    };

    const config = {
      responsive: true,
      displayModeBar: false,
    };

    Plotly.newPlot(container, [trace], layout, config);
  }

  // Start real-time updates
  startRealTimeUpdates() {
    console.log("🔄 Starting real-time updates...");

    this.realTimeInterval = setInterval(() => {
      this.updateKPIs();
      this.updateInsights();
      this.simulateNewData();
    }, 30000); // Update every 30 seconds
  }

  // Update KPI values with real-time data
  updateKPIs() {
    const latestData = this.data.historical[this.data.historical.length - 1];

    // Update fraud rate
    const fraudRateElement = document.getElementById("fraudRate");
    if (fraudRateElement) {
      const newRate = (latestData.fraudRate * 100).toFixed(1) + "%";
      fraudRateElement.textContent = newRate;
    }

    // Update predicted fraud
    const predictedFraudElement = document.getElementById("predictedFraud");
    if (predictedFraudElement) {
      const predicted = Math.floor(120 + Math.random() * 80);
      predictedFraudElement.textContent = predicted.toString();
    }

    // Update model accuracy
    const modelAccuracyElement = document.getElementById("modelAccuracy");
    if (modelAccuracyElement) {
      const accuracy = (latestData.accuracy * 100).toFixed(1) + "%";
      modelAccuracyElement.textContent = accuracy;
    }

    // Update anomaly score
    const anomalyScoreElement = document.getElementById("anomalyScore");
    if (anomalyScoreElement) {
      const score = (5 + Math.random() * 5).toFixed(1);
      anomalyScoreElement.textContent = score;
    }
  }

  // Generate and update AI insights
  generateAIInsights() {
    const insights = [
      {
        icon: "fas fa-exclamation-triangle",
        title: "Fraud Spike Detected",
        description:
          "23% increase in high-risk transactions from IP range 192.168.1.0/24 in the last 2 hours.",
        severity: "high",
      },
      {
        icon: "fas fa-chart-line",
        title: "Trend Alert",
        description:
          "Weekend fraud patterns show 15% higher activity. Consider adjusting monitoring thresholds.",
        severity: "medium",
      },
      {
        icon: "fas fa-lightbulb",
        title: "Model Recommendation",
        description:
          'Feature importance analysis suggests adding "time_since_last_transaction" could improve accuracy by 2.1%.',
        severity: "info",
      },
      {
        icon: "fas fa-globe",
        title: "Geographic Pattern",
        description:
          "Unusual fraud concentration detected in Southeast Asia region. Recommend enhanced monitoring.",
        severity: "medium",
      },
      {
        icon: "fas fa-clock",
        title: "Temporal Anomaly",
        description:
          "Fraud activity spike detected during off-peak hours (2-4 AM). Possible automated attack.",
        severity: "high",
      },
    ];

    this.updateInsightsList(insights.slice(0, 3));
  }

  // Update AI insights list
  updateInsightsList(insights) {
    const insightsList = document.getElementById("aiInsights");
    if (!insightsList) return;

    insightsList.innerHTML = insights
      .map(
        (insight) => `
      <li>
        <div class="insight-icon">
          <i class="${insight.icon}"></i>
        </div>
        <div>
          <strong>${insight.title}:</strong> ${insight.description}
        </div>
      </li>
    `
      )
      .join("");
  }

  // Update insights with new data
  updateInsights() {
    const newInsights = [
      {
        icon: "fas fa-trending-up",
        title: "Performance Improvement",
        description: `Model accuracy increased by ${(
          Math.random() * 0.5
        ).toFixed(1)}% in the last hour.`,
        severity: "info",
      },
      {
        icon: "fas fa-shield-alt",
        title: "Security Alert",
        description: `Blocked ${Math.floor(
          50 + Math.random() * 100
        )} suspicious transactions in the last 30 minutes.`,
        severity: "medium",
      },
      {
        icon: "fas fa-chart-bar",
        title: "Volume Analysis",
        description: `Transaction volume ${
          Math.random() > 0.5 ? "increased" : "decreased"
        } by ${(Math.random() * 10).toFixed(1)}% compared to yesterday.`,
        severity: "info",
      },
    ];

    this.updateInsightsList(newInsights);
  }

  // Simulate new data points
  simulateNewData() {
    const now = new Date();
    const lastPoint = this.data.historical[this.data.historical.length - 1];

    // Add new data point
    const newPoint = {
      date: now,
      fraudRate: Math.max(
        0,
        lastPoint.fraudRate + (Math.random() - 0.5) * 0.005
      ),
      transactions: Math.floor(800 + Math.random() * 400),
      fraudCount: 0,
      accuracy: Math.min(
        1,
        Math.max(0.9, lastPoint.accuracy + (Math.random() - 0.5) * 0.02)
      ),
      precision: Math.min(
        1,
        Math.max(0.85, lastPoint.precision + (Math.random() - 0.5) * 0.02)
      ),
      recall: Math.min(
        1,
        Math.max(0.8, lastPoint.recall + (Math.random() - 0.5) * 0.02)
      ),
      f1Score: Math.min(
        1,
        Math.max(0.8, lastPoint.f1Score + (Math.random() - 0.5) * 0.02)
      ),
    };

    newPoint.fraudCount = Math.floor(
      newPoint.fraudRate * newPoint.transactions
    );

    // Keep only last 90 days
    if (this.data.historical.length >= 90) {
      this.data.historical.shift();
    }

    this.data.historical.push(newPoint);

    // Update charts with new data
    this.updateChartsWithNewData();
  }

  // Update charts with new data
  updateChartsWithNewData() {
    // Update trend chart
    if (this.charts.trend) {
      const weeklyData = this.aggregateByWeek(this.data.historical);
      this.charts.trend.data.labels = weeklyData.map((point) => point.week);
      this.charts.trend.data.datasets[0].data = weeklyData.map(
        (point) => point.avgFraudRate * 100
      );
      this.charts.trend.data.datasets[1].data = weeklyData.map(
        (point) => point.totalTransactions / 100
      );
      this.charts.trend.update("none");
    }

    // Update time series chart
    if (document.getElementById("timeSeriesChart")) {
      const traces = [
        {
          x: this.data.historical.map((point) => point.date),
          y: this.data.historical.map((point) => point.fraudRate * 100),
          type: "scatter",
          mode: "lines",
          name: "Fraud Rate",
          line: { color: "#6366f1", width: 2 },
        },
        {
          x: this.data.historical.map((point) => point.date),
          y: this.data.historical.map((point) => point.accuracy * 100),
          type: "scatter",
          mode: "lines",
          name: "Model Accuracy",
          yaxis: "y2",
          line: { color: "#10b981", width: 2 },
        },
      ];

      Plotly.redraw("timeSeriesChart");
    }
  }

  // Update charts theme
  updateChartsTheme() {
    const isDark = currentTheme === "dark";
    const textColor = isDark ? "#f1f5f9" : "#1f2937";
    const gridColor = isDark ? "#374151" : "#e5e7eb";

    Object.values(this.charts).forEach((chart) => {
      if (chart && chart.options) {
        chart.options.plugins.title.color = textColor;
        chart.options.scales.x.ticks.color = textColor;
        chart.options.scales.y.ticks.color = textColor;
        chart.options.scales.x.grid.color = gridColor;
        chart.options.scales.y.grid.color = gridColor;
        chart.update("none");
      }
    });
  }

  // Recreate chart in modal for better viewing
  recreateChartInModal(chartId) {
    const modalCanvas = document.querySelector(
      `#cardModal canvas[id="${chartId}"]`
    );
    if (!modalCanvas) return;

    // Create a new canvas with a unique ID for the modal
    const newCanvas = document.createElement("canvas");
    newCanvas.id = chartId + "_modal";

    // Replace the canvas
    modalCanvas.parentNode.replaceChild(newCanvas, modalCanvas);

    // Recreate the specific chart based on ID with enhanced modal styling
    switch (chartId) {
      case "forecastChart":
        this.createModalForecastChart(newCanvas.id);
        break;
      case "trendChart":
        this.createModalTrendChart(newCanvas.id);
        break;
      case "featureImportanceChart":
        this.createModalFeatureImportanceChart(newCanvas.id);
        break;
      default:
        // For other charts, just recreate the original
        this.createAllCharts();
        break;
    }
  }

  // Enhanced modal chart creation with larger fonts and better spacing
  createModalForecastChart(canvasId) {
    const ctx = document.getElementById(canvasId).getContext("2d");
    const historicalData = this.data.historical.slice(-30).map((point) => ({
      x: point.date,
      y: point.fraudRate * 100,
    }));
    const forecastData = this.data.forecast.map((point) => ({
      x: point.date,
      y: point.predicted * 100,
    }));

    new Chart(ctx, {
      type: "line",
      data: {
        datasets: [
          {
            label: "Historical Fraud Rate",
            data: historicalData,
            borderColor: "#6366f1",
            backgroundColor: "rgba(99, 102, 241, 0.1)",
            borderWidth: 4,
            fill: false,
            tension: 0.4,
            pointRadius: 6,
            pointHoverRadius: 10,
          },
          {
            label: "Predicted Fraud Rate",
            data: forecastData,
            borderColor: "#8b5cf6",
            backgroundColor: "rgba(139, 92, 246, 0.1)",
            borderWidth: 4,
            borderDash: [10, 10],
            fill: false,
            tension: 0.4,
            pointRadius: 6,
            pointHoverRadius: 10,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        layout: {
          padding: { top: 30, right: 40, bottom: 30, left: 40 },
        },
        plugins: {
          title: {
            display: true,
            text: "Fraud Rate Prediction with Confidence Intervals",
            font: { size: 22, weight: "bold" },
            padding: { bottom: 30 },
          },
          legend: {
            position: "top",
            labels: { boxWidth: 20, padding: 30, font: { size: 16 } },
          },
        },
        scales: {
          x: {
            type: "time",
            time: { unit: "day", displayFormats: { day: "MMM dd" } },
            title: { display: true, text: "Date", font: { size: 16 } },
            ticks: { font: { size: 14 }, maxRotation: 0, maxTicksLimit: 8 },
          },
          y: {
            title: {
              display: true,
              text: "Fraud Rate (%)",
              font: { size: 16 },
            },
            min: 0,
            ticks: {
              font: { size: 14 },
              callback: function (value) {
                return value.toFixed(1) + "%";
              },
            },
          },
        },
      },
    });
  }

  createModalTrendChart(canvasId) {
    const ctx = document.getElementById(canvasId).getContext("2d");
    const weeklyData = this.aggregateByWeek(this.data.historical);

    new Chart(ctx, {
      type: "line",
      data: {
        labels: weeklyData.map((point) => point.week),
        datasets: [
          {
            label: "Fraud Rate Trend",
            data: weeklyData.map((point) => point.avgFraudRate * 100),
            borderColor: "#10b981",
            backgroundColor: "rgba(16, 185, 129, 0.1)",
            borderWidth: 5,
            fill: true,
            tension: 0.4,
            pointRadius: 7,
          },
          {
            label: "Transaction Volume",
            data: weeklyData.map((point) => point.totalTransactions / 100),
            borderColor: "#f59e0b",
            backgroundColor: "rgba(245, 158, 11, 0.1)",
            borderWidth: 4,
            fill: false,
            yAxisID: "y1",
            tension: 0.4,
            pointRadius: 6,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        layout: { padding: { top: 30, right: 40, bottom: 30, left: 40 } },
        plugins: {
          title: {
            display: true,
            text: "Weekly Fraud Trends & Transaction Volume",
            font: { size: 22, weight: "bold" },
            padding: { bottom: 30 },
          },
          legend: {
            position: "top",
            labels: { boxWidth: 20, padding: 30, font: { size: 16 } },
          },
        },
        scales: {
          x: {
            ticks: { font: { size: 14 }, maxRotation: 0, maxTicksLimit: 10 },
          },
          y: {
            type: "linear",
            position: "left",
            title: {
              display: true,
              text: "Fraud Rate (%)",
              font: { size: 16 },
            },
            ticks: { font: { size: 14 } },
          },
          y1: {
            type: "linear",
            position: "right",
            title: {
              display: true,
              text: "Transactions (hundreds)",
              font: { size: 16 },
            },
            ticks: { font: { size: 14 } },
            grid: { drawOnChartArea: false },
          },
        },
      },
    });
  }

  createModalFeatureImportanceChart(canvasId) {
    const ctx = document.getElementById(canvasId).getContext("2d");

    new Chart(ctx, {
      type: "bar",
      data: {
        labels: this.data.featureImportance.map((item) => item.feature),
        datasets: [
          {
            label: "Feature Importance",
            data: this.data.featureImportance.map((item) => item.importance),
            backgroundColor: this.data.featureImportance.map((item, index) => {
              const colors = [
                "#6366f1",
                "#8b5cf6",
                "#06b6d4",
                "#10b981",
                "#f59e0b",
                "#ef4444",
                "#8b5cf6",
              ];
              return colors[index % colors.length];
            }),
            borderWidth: 2,
            borderRadius: 8,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        layout: { padding: { top: 30, right: 40, bottom: 30, left: 40 } },
        plugins: {
          title: {
            display: true,
            text: "Feature Importance Analysis",
            font: { size: 22, weight: "bold" },
            padding: { bottom: 30 },
          },
          legend: { display: false },
        },
        scales: {
          y: {
            beginAtZero: true,
            title: {
              display: true,
              text: "Importance Score",
              font: { size: 16 },
            },
            ticks: {
              font: { size: 14 },
              callback: function (value) {
                return (value * 100).toFixed(0) + "%";
              },
            },
          },
          x: {
            ticks: {
              font: { size: 12 },
              maxRotation: 45,
              callback: function (value, index) {
                const label = this.getLabelForValue(value);
                return label.length > 15
                  ? label.substring(0, 15) + "..."
                  : label;
              },
            },
          },
        },
      },
    });
  }
}

// Interactive Functions for UI
function refreshForecast() {
  console.log("🔄 Refreshing forecast...");
  const loading = document.getElementById("forecastLoading");
  if (loading) {
    loading.classList.remove("hidden");
    setTimeout(() => {
      loading.classList.add("hidden");
      // Regenerate forecast
      if (window.analyticsEngine) {
        window.analyticsEngine.initializeForecastingModel();
        window.analyticsEngine.createForecastChart();
      }
    }, 2000);
  }
}

function openForecastSettings() {
  alert(
    "Forecast settings panel would open here. This could include:\n\n• Forecast horizon (7-30 days)\n• Confidence intervals\n• Seasonal adjustments\n• Model selection"
  );
}

function exportTrendData() {
  console.log("📊 Exporting trend data...");
  alert(
    "Trend data export functionality would be implemented here.\n\nSupported formats:\n• CSV\n• Excel\n• JSON\n• PDF Report"
  );
}

function drillDownTrends() {
  console.log("🔍 Opening trend drill-down...");
  // Trigger the modal for trend analysis
  const trendCard = document
    .querySelector('[id*="trendChart"]')
    ?.closest(".analytics-card");
  if (trendCard) {
    openCardModal(trendCard);
  } else {
    showToast("Opening trend analysis...", "info");
  }
}

function configureAnomalyAlerts() {
  console.log("⚙️ Configuring anomaly alerts...");
  showToast("Anomaly alert configuration opened", "info");
  // Trigger the modal for anomaly detection
  const anomalyCard = document
    .querySelector('[id*="anomalyChart"]')
    ?.closest(".analytics-card");
  if (anomalyCard) {
    openCardModal(anomalyCard);
  }
}

function addModelComparison() {
  console.log("➕ Adding model comparison...");
  showToast("Model comparison view opened", "info");
  // Trigger the modal for model comparison
  const modelCard = document
    .querySelector('[id*="modelComparisonChart"]')
    ?.closest(".analytics-card");
  if (modelCard) {
    openCardModal(modelCard);
  }
}

function expandGeoMap() {
  console.log("🌍 Expanding geographic map...");
  showToast("Geographic analysis expanded", "success");
  // Trigger the modal for geographic chart
  const geoCard = document
    .querySelector("#geoChart")
    ?.closest(".analytics-card");
  if (geoCard) {
    openCardModal(geoCard);
  }
}

function runSHAPAnalysis() {
  console.log("🔬 Running SHAP analysis...");
  showToast("SHAP analysis initiated", "info");
  // Trigger the modal for feature importance
  const featureCard = document
    .querySelector('[id*="featureImportanceChart"]')
    ?.closest(".analytics-card");
  if (featureCard) {
    openCardModal(featureCard);
  }
}

function analyzeSeasonality() {
  console.log("📊 Analyzing seasonality...");
  showToast("Seasonality analysis opened", "info");
  // Trigger the modal for time series
  const timeSeriesCard = document
    .querySelector("#timeSeriesChart")
    ?.closest(".analytics-card");
  if (timeSeriesCard) {
    openCardModal(timeSeriesCard);
  }
}

function showCorrelationMatrix() {
  console.log("🔗 Showing correlation matrix...");
  showToast("Correlation matrix displayed", "info");
  // Trigger the modal for time series (which includes correlation data)
  const timeSeriesCard = document
    .querySelector("#timeSeriesChart")
    ?.closest(".analytics-card");
  if (timeSeriesCard) {
    openCardModal(timeSeriesCard);
  }
}

// Toast notification function
function showToast(message, type = "info") {
  // Check if toast container exists, if not create it
  let toastContainer = document.getElementById("toastContainer");
  if (!toastContainer) {
    toastContainer = document.createElement("div");
    toastContainer.id = "toastContainer";
    toastContainer.style.cssText = `
      position: fixed;
      top: 20px;
      right: 20px;
      z-index: 10000;
      display: flex;
      flex-direction: column;
      gap: 10px;
    `;
    document.body.appendChild(toastContainer);
  }

  // Create toast element
  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;

  // Set icon based on type
  let icon = "fa-info-circle";
  let bgColor = "#3b82f6";
  if (type === "success") {
    icon = "fa-check-circle";
    bgColor = "#10b981";
  } else if (type === "error") {
    icon = "fa-exclamation-circle";
    bgColor = "#ef4444";
  } else if (type === "warning") {
    icon = "fa-exclamation-triangle";
    bgColor = "#f59e0b";
  }

  toast.style.cssText = `
    background: ${bgColor};
    color: white;
    padding: 12px 20px;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    display: flex;
    align-items: center;
    gap: 10px;
    min-width: 250px;
    max-width: 400px;
    animation: slideIn 0.3s ease-out;
    font-family: Inter, sans-serif;
    font-size: 14px;
    font-weight: 500;
  `;

  toast.innerHTML = `
    <i class="fas ${icon}"></i>
    <span>${message}</span>
  `;

  toastContainer.appendChild(toast);

  // Auto remove after 3 seconds
  setTimeout(() => {
    toast.style.animation = "slideOut 0.3s ease-in";
    setTimeout(() => {
      toast.remove();
    }, 300);
  }, 3000);
}

// Initialize analytics when DOM is loaded
document.addEventListener("DOMContentLoaded", function () {
  window.analyticsEngine = new AdvancedAnalyticsEngine();
  window.analyticsEngine.initialize();

  // Add toast animations to document
  if (!document.getElementById("toastAnimations")) {
    const style = document.createElement("style");
    style.id = "toastAnimations";
    style.textContent = `
      @keyframes slideIn {
        from {
          transform: translateX(400px);
          opacity: 0;
        }
        to {
          transform: translateX(0);
          opacity: 1;
        }
      }
      @keyframes slideOut {
        from {
          transform: translateX(0);
          opacity: 1;
        }
        to {
          transform: translateX(400px);
          opacity: 0;
        }
      }
    `;
    document.head.appendChild(style);
  }
});

// Global functions for the main HTML
function initializeCharts() {
  // This will be called by the main HTML
  if (window.analyticsEngine) {
    window.analyticsEngine.createAllCharts();
  }
}

function loadAnalyticsData() {
  // This will be called by the main HTML
  if (window.analyticsEngine) {
    window.analyticsEngine.loadHistoricalData();
  }
}

function startRealTimeUpdates() {
  // This will be called by the main HTML
  if (window.analyticsEngine) {
    window.analyticsEngine.startRealTimeUpdates();
  }
}

function updateChartsTheme() {
  // This will be called by the main HTML
  if (window.analyticsEngine) {
    window.analyticsEngine.updateChartsTheme();
  }
}
