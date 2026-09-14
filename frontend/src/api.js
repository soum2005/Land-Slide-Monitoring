const BASE = import.meta.env.VITE_API_URL || "https://land-slide-monitoring.onrender.com";

export async function api(path, options = {}) {
  const token = localStorage.getItem("ner_token");
  const headers = { ...(options.headers || {}) };
  if (!(options.body instanceof FormData)) {
    headers["Content-Type"] = "application/json";
  }
  if (token) headers.Authorization = `Bearer ${token}`;
  const res = await fetch(`${BASE}${path}`, { ...options, headers });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Request failed");
  }
  const ct = res.headers.get("content-type") || "";
  if (ct.includes("text/csv")) return res.text();
  return res.json();
}

export const get = (p) => api(p);
export const post = (p, body) => api(p, { method: "POST", body: body instanceof FormData ? body : JSON.stringify(body) });
export const patch = (p, body) => api(p, { method: "PATCH", body: JSON.stringify(body) });

// High-level API methods
export const searchLocationAndPredict = (query, latitude, longitude) =>
  post("/api/predict/search-location", { query, latitude, longitude });

export const getGazetteer = (q = "") => get(`/api/predict/gazetteer?q=${encodeURIComponent(q)}`);

export const predictMultiHazard = (payload) => post("/api/predict/multi-hazard", payload);

export const inspectCoordinate = (lat, lon) => get(`/api/map/inspect-coordinate?lat=${lat}&lon=${lon}`);

export const getFloodZones = () => get("/api/map/flood-zones");

export const batchSyncIncidents = (reports) => post("/api/incidents/batch-sync", { reports });

export const analyzeRouteRisk = (origin, destination) =>
  post("/api/predict/route-analysis", { origin, destination });

export const analyzeIncidentImage = (formData) =>
  post("/api/incidents/analyze-image", formData);

export const generateAIReport = (query, location_data) =>
  post("/api/ai/report", { query, location_data });

export const simulateScenario = (payload) =>
  post("/api/ai/simulate", payload);

export const chatWithAIAssistant = (message, context_location, location_data) =>
  post("/api/ai/chat", { message, context_location, location_data });

export const compareLocations = (location_a, location_b) =>
  post("/api/predict/compare", { location_a, location_b });

export const getRegionalRiskIndices = () =>
  get("/api/ner/risk-indices");



// ─── Earthquake / Seismic ───────────────────────────────────────────────────
export const getEarthquakeProfile = (query, latitude, longitude) =>
  post("/api/earthquake/profile", { query, latitude, longitude });

// ─── NER enhanced indices ───────────────────────────────────────────────────
export const getNERRiskIndices = () => get("/api/ner/risk-indices");

// ─── Authority / Admin ──────────────────────────────────────────────────────
export const getMLMetrics = () => get("/api/ml/metrics");
export const getStateAnalytics = () => get("/api/dashboard/state-analytics");
