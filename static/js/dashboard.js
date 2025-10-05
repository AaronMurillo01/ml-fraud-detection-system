/**
 * FraudGuard AI Dashboard
 * Real-time fraud detection dashboard with WebSocket support
 */

// Global state
let ws = null;
let currentTheme = localStorage.getItem("theme") || "light";
let stats = {
  totalTransactions: 0,
  fraudDetected: 0,
  accuracy: 0,
  avgResponseTime: 0,
};

// Initialize dashboard
document.addEventListener("DOMContentLoaded", function () {
  initializeTheme();
  setupEventListeners();
  loadInitialData();
  connectWebSocket();
});

// Theme Management
function initializeTheme() {
  document.documentElement.setAttribute("data-theme", currentTheme);
  updateThemeIcon();
}

function toggleTheme() {
  currentTheme = currentTheme === "light" ? "dark" : "light";
  document.documentElement.setAttribute("data-theme", currentTheme);
  localStorage.setItem("theme", currentTheme);
  updateThemeIcon();
  showToast("Theme switched to " + currentTheme + " mode", "success");
}

function updateThemeIcon() {
  const themeIcon = document.querySelector("#themeToggle i");
  if (themeIcon) {
    themeIcon.className =
      currentTheme === "light" ? "fas fa-moon" : "fas fa-sun";
  }
}

// Event Listeners
function setupEventListeners() {
  const themeToggle = document.getElementById("themeToggle");
  if (themeToggle) {
    themeToggle.addEventListener("click", toggleTheme);
  }

  const mobileMenuToggle = document.getElementById("mobileMenuToggle");
  const navLinks = document.getElementById("navLinks");
  if (mobileMenuToggle && navLinks) {
    mobileMenuToggle.addEventListener("click", function () {
      navLinks.classList.toggle("active");
      const icon = mobileMenuToggle.querySelector("i");
      if (icon) {
        icon.className = navLinks.classList.contains("active")
          ? "fas fa-times"
          : "fas fa-bars";
      }
    });

    // Close menu when clicking outside
    document.addEventListener("click", function (event) {
      if (
        !navLinks.contains(event.target) &&
        !mobileMenuToggle.contains(event.target)
      ) {
        navLinks.classList.remove("active");
        const icon = mobileMenuToggle.querySelector("i");
        if (icon) {
          icon.className = "fas fa-bars";
        }
      }
    });
  }
}

// Load Initial Data
async function loadInitialData() {
  try {
    await Promise.all([loadStats(), loadRecentTransactions()]);
  } catch (error) {
    console.error("Error loading initial data:", error);
    showToast("Failed to load dashboard data", "error");
  }
}

// Load Statistics
async function loadStats() {
  try {
    const response = await fetch("/api/v1/history/stats/summary?days=30");
    if (response.ok) {
      const data = await response.json();
      updateStatsDisplay(data);
    }
  } catch (error) {
    console.error("Error loading stats:", error);
    // Use mock data for demo
    updateStatsDisplay({
      total_transactions: 12543,
      fraud_transactions: 127,
      fraud_rate: 0.0101,
      total_amount: 1254300.5,
      fraud_amount: 45230.75,
      amount_saved: 45230.75,
      avg_transaction_amount: 99.95,
    });
  }
}

// Update Stats Display
function updateStatsDisplay(data) {
  const statsGrid = document.getElementById("statsGrid");

  statsGrid.innerHTML = `
    <div class="stat-card" style="background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);">
      <div class="stat-label">Total Transactions</div>
      <div class="stat-value">${data.total_transactions.toLocaleString()}</div>
      <div class="stat-trend">
        <i class="fas fa-arrow-up"></i>
        <span>+12.5% from last month</span>
      </div>
    </div>
    
    <div class="stat-card" style="background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);">
      <div class="stat-label">Fraud Detected</div>
      <div class="stat-value">${data.fraud_transactions.toLocaleString()}</div>
      <div class="stat-trend">
        <i class="fas fa-arrow-down"></i>
        <span>-5.2% from last month</span>
      </div>
    </div>
    
    <div class="stat-card" style="background: linear-gradient(135deg, #10b981 0%, #059669 100%);">
      <div class="stat-label">Amount Saved</div>
      <div class="stat-value">$${(data.amount_saved / 1000).toFixed(1)}K</div>
      <div class="stat-trend">
        <i class="fas fa-arrow-up"></i>
        <span>+18.3% from last month</span>
      </div>
    </div>
    
    <div class="stat-card" style="background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%);">
      <div class="stat-label">Detection Rate</div>
      <div class="stat-value">${(data.fraud_rate * 100).toFixed(2)}%</div>
      <div class="stat-trend">
        <i class="fas fa-check"></i>
        <span>Within normal range</span>
      </div>
    </div>
  `;
}

// Load Recent Transactions
async function loadRecentTransactions() {
  try {
    const response = await fetch("/api/v1/history/?page=1&page_size=10");
    if (response.ok) {
      const data = await response.json();
      displayRecentTransactions(data.items);
    }
  } catch (error) {
    console.error("Error loading recent transactions:", error);
    // Use mock data for demo
    displayRecentTransactions([
      {
        transaction_id: "txn_001234",
        user_id: "user_5678",
        merchant_name: "Amazon",
        amount: 129.99,
        currency: "USD",
        timestamp: new Date().toISOString(),
        fraud_score: 0.15,
        risk_level: "low",
        status: "approved",
      },
      {
        transaction_id: "txn_001235",
        user_id: "user_9012",
        merchant_name: "Walmart",
        amount: 45.5,
        currency: "USD",
        timestamp: new Date(Date.now() - 300000).toISOString(),
        fraud_score: 0.82,
        risk_level: "high",
        status: "declined",
      },
    ]);
  }
}

// Display Recent Transactions
function displayRecentTransactions(transactions) {
  const container = document.getElementById("recentTransactions");

  if (!transactions || transactions.length === 0) {
    container.innerHTML =
      '<p style="color: var(--gray-500); text-align: center; padding: 2rem;">No recent transactions</p>';
    return;
  }

  container.innerHTML = `
    <div style="overflow-x: auto;">
      <table style="width: 100%; border-collapse: collapse;">
        <thead>
          <tr style="border-bottom: 2px solid var(--gray-200);">
            <th style="padding: 0.75rem; text-align: left; font-weight: 600; color: var(--gray-700);">Transaction</th>
            <th style="padding: 0.75rem; text-align: left; font-weight: 600; color: var(--gray-700);">Merchant</th>
            <th style="padding: 0.75rem; text-align: right; font-weight: 600; color: var(--gray-700);">Amount</th>
            <th style="padding: 0.75rem; text-align: center; font-weight: 600; color: var(--gray-700);">Risk</th>
            <th style="padding: 0.75rem; text-align: center; font-weight: 600; color: var(--gray-700);">Status</th>
          </tr>
        </thead>
        <tbody>
          ${transactions
            .map(
              (t) => `
            <tr style="border-bottom: 1px solid var(--gray-100);">
              <td style="padding: 0.75rem;">
                <div style="font-weight: 500; color: var(--gray-900);">${
                  t.transaction_id
                }</div>
                <div style="font-size: 0.75rem; color: var(--gray-500);">${new Date(
                  t.timestamp
                ).toLocaleString()}</div>
              </td>
              <td style="padding: 0.75rem; color: var(--gray-700);">${
                t.merchant_name
              }</td>
              <td style="padding: 0.75rem; text-align: right; font-weight: 600; color: var(--gray-900);">
                ${t.currency} ${t.amount.toFixed(2)}
              </td>
              <td style="padding: 0.75rem; text-align: center;">
                ${getRiskBadge(t.risk_level, t.fraud_score)}
              </td>
              <td style="padding: 0.75rem; text-align: center;">
                ${getStatusBadge(t.status)}
              </td>
            </tr>
          `
            )
            .join("")}
        </tbody>
      </table>
    </div>
  `;
}

// Get Risk Badge
function getRiskBadge(riskLevel, score) {
  const badges = {
    low: '<span class="badge badge-success"><i class="fas fa-check"></i> Low</span>',
    medium:
      '<span class="badge badge-warning"><i class="fas fa-exclamation"></i> Medium</span>',
    high: '<span class="badge badge-danger"><i class="fas fa-exclamation-triangle"></i> High</span>',
    critical:
      '<span class="badge badge-danger"><i class="fas fa-skull"></i> Critical</span>',
  };
  return badges[riskLevel] || badges.low;
}

// Get Status Badge
function getStatusBadge(status) {
  const badges = {
    approved: '<span class="badge badge-success">Approved</span>',
    declined: '<span class="badge badge-danger">Declined</span>',
    pending: '<span class="badge badge-warning">Pending</span>',
    under_review: '<span class="badge badge-info">Under Review</span>',
  };
  return badges[status] || badges.pending;
}

// WebSocket Connection
function connectWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws/fraud-updates?channels=transactions,alerts,metrics`;

  try {
    ws = new WebSocket(wsUrl);

    ws.onopen = function () {
      console.log("WebSocket connected");
      updateWSStatus(true);
    };

    ws.onmessage = function (event) {
      try {
        const message = JSON.parse(event.data);
        handleWebSocketMessage(message);
      } catch (error) {
        console.error("Error parsing WebSocket message:", error);
      }
    };

    ws.onerror = function (error) {
      console.error("WebSocket error:", error);
      updateWSStatus(false);
    };

    ws.onclose = function () {
      console.log("WebSocket disconnected");
      updateWSStatus(false);

      // Attempt to reconnect after 5 seconds
      setTimeout(connectWebSocket, 5000);
    };
  } catch (error) {
    console.error("Error connecting WebSocket:", error);
    updateWSStatus(false);
  }
}

// Handle WebSocket Messages
function handleWebSocketMessage(message) {
  console.log("WebSocket message:", message);

  switch (message.type) {
    case "transaction":
      handleTransactionUpdate(message.data);
      break;
    case "alert":
      handleFraudAlert(message.data);
      break;
    case "metric":
      handleMetricUpdate(message.data);
      break;
    case "system_status":
      handleSystemStatus(message.data);
      break;
    case "connection":
      console.log("Connection established:", message.message);
      break;
    default:
      console.log("Unknown message type:", message.type);
  }
}

// Handle Transaction Update
function handleTransactionUpdate(data) {
  // Refresh recent transactions
  loadRecentTransactions();

  // Show notification for high-risk transactions
  if (data.risk_level === "high" || data.risk_level === "critical") {
    showToast(
      `High-risk transaction detected: ${data.transaction_id}`,
      "warning"
    );
  }
}

// Handle Fraud Alert
function handleFraudAlert(data) {
  showToast(`Fraud Alert: ${data.reason}`, "error");

  // Optionally show a more prominent notification
  if (data.severity === "critical") {
    alert(
      `CRITICAL FRAUD ALERT\n\nTransaction: ${data.transaction_id}\nReason: ${data.reason}`
    );
  }
}

// Handle Metric Update
function handleMetricUpdate(data) {
  // Update stats if needed
  if (data.metric_type === "stats") {
    updateStatsDisplay(data);
  }
}

// Handle System Status
function handleSystemStatus(data) {
  console.log("System status:", data);
}

// Update WebSocket Status
function updateWSStatus(connected) {
  const statusEl = document.getElementById("wsStatus");
  if (statusEl) {
    if (connected) {
      statusEl.innerHTML =
        '<div class="pulse"></div><span>Real-time Updates Active</span>';
      statusEl.style.background = "var(--success)";
    } else {
      statusEl.innerHTML =
        '<i class="fas fa-exclamation-circle"></i><span>Disconnected</span>';
      statusEl.style.background = "var(--danger)";
    }
  }
}

// Toast Notifications
function showToast(message, type = "info") {
  const container = document.getElementById("toastContainer");
  if (!container) return;

  const toast = document.createElement("div");
  toast.className = `toast ${type}`;

  const iconMap = {
    success: "fa-check-circle",
    error: "fa-exclamation-circle",
    warning: "fa-exclamation-triangle",
    info: "fa-info-circle",
  };

  toast.innerHTML = `
    <i class="fas ${iconMap[type] || iconMap.info} toast-icon"></i>
    <span class="toast-message">${message}</span>
    <button class="toast-close" aria-label="Close">
      <i class="fas fa-times"></i>
    </button>
  `;

  container.appendChild(toast);

  // Close button functionality
  const closeBtn = toast.querySelector(".toast-close");
  closeBtn.addEventListener("click", () => {
    toast.style.animation = "slideIn 0.3s ease-out reverse";
    setTimeout(() => toast.remove(), 300);
  });

  // Auto-remove after 5 seconds
  setTimeout(() => {
    if (toast.parentElement) {
      toast.style.animation = "slideIn 0.3s ease-out reverse";
      setTimeout(() => toast.remove(), 300);
    }
  }, 5000);
}

// Legacy toast implementation (keeping for compatibility)
function showToastLegacy(message, type = "info") {
  const toast = document.createElement("div");
  toast.style.cssText = `
    position: fixed;
    bottom: 2rem;
    right: 2rem;
    padding: 1rem 1.5rem;
    background: ${
      type === "success"
        ? "var(--success)"
        : type === "error"
        ? "var(--danger)"
        : type === "warning"
        ? "var(--warning)"
        : "var(--info)"
    };
    color: white;
    border-radius: var(--border-radius);
    box-shadow: var(--shadow-lg);
    z-index: 10000;
    animation: slideIn 0.3s ease-out;
  `;
  toast.textContent = message;

  document.body.appendChild(toast);

  setTimeout(() => {
    toast.style.animation = "slideOut 0.3s ease-out";
    setTimeout(() => toast.remove(), 300);
  }, 3000);
}

// Add animations
const style = document.createElement("style");
style.textContent = `
  @keyframes slideIn {
    from {
      transform: translateX(100%);
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
      transform: translateX(100%);
      opacity: 0;
    }
  }
`;
document.head.appendChild(style);
