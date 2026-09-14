import {
  generateLocationPredictionFallback,
  generateRouteAnalysisFallback,
  searchGazetteer,
  getMockDashboardOverview,
  getMockMLMetrics,
  getMockLithologies,
  getMockMapStations,
  getMockHighRiskPolygons,
  getMockFloodZones,
  getMockSensors,
  getMockFacilities,
  getMockEvacuationRoutes,
} from "./services/fallbackEngine";

const CANDIDATE_BASE_URLS = [
  "https://land-slide-monitoring.onrender.com",
  
  import.meta.env.VITE_API_URL,
  "http://127.0.0.1:8000",
  "http://localhost:8000",
  ""
].filter((url) => url !== undefined && url !== null);

let activeBackendUrl = null;
let connectionStatus = "unknown";

export function getBackendStatus() {
  return connectionStatus;
}

export async function api(path, options = {}) {
  let cleanPath = path.startsWith("/") ? path : `/${path}`;
  if (!cleanPath.startsWith("/api/")) {
    cleanPath = `/api${cleanPath}`;
  }

  const token = localStorage.getItem("ner_token");
  const headers = { ...(options.headers || {}) };
  if (!(options.body instanceof FormData)) {
    headers["Content-Type"] = "application/json";
  }
  if (token) headers.Authorization = `Bearer ${token}`;

  const basesToTry = activeBackendUrl !== null
    ? [activeBackendUrl, ...CANDIDATE_BASE_URLS]
    : CANDIDATE_BASE_URLS;

  for (const base of basesToTry) {
    try {
      const fullUrl = `${base}${cleanPath}`;
      const res = await fetch(fullUrl, { ...options, headers, signal: AbortSignal.timeout(4000) });
      const contentType = res.headers.get("content-type") || "";

      if (res.ok) {
        activeBackendUrl = base;
        connectionStatus = "connected";
        if (contentType.includes("text/csv")) return await res.text();
        return await res.json();
      }
    } catch (e) {
      // Endpoint unreachable on this candidate base URL
    }
  }

  connectionStatus = "standalone";
  console.warn(`[API Proxy] Backend unreachable for ${cleanPath}. Utilizing Client Fallback Engine.`);

  // Client Fallbacks
  if (cleanPath.includes("/predict/search-location")) {
    const body = options.body ? (typeof options.body === 'string' ? JSON.parse(options.body) : options.body) : {};
    return generateLocationPredictionFallback(body.query, body.latitude, body.longitude);
  }
  if (cleanPath.includes("/predict/gazetteer")) {
    const queryStr = cleanPath.split("?q=")[1] || "";
    return searchGazetteer(decodeURIComponent(queryStr));
  }
  if (cleanPath.includes("/predict/route-analysis")) {
    const body = options.body ? (typeof options.body === 'string' ? JSON.parse(options.body) : options.body) : {};
    return generateRouteAnalysisFallback(body.origin, body.destination);
  }
  if (cleanPath.includes("/predict/multi-hazard") || cleanPath.includes("/predict-risk")) {
    return generateLocationPredictionFallback("Custom Assessment Grid");
  }
  if (cleanPath.includes("/ml/metrics")) {
    return getMockMLMetrics();
  }
  if (cleanPath.includes("/dashboard/summary")) {
    return getMockDashboardOverview();
  }
  if (cleanPath.includes("/map/stations")) {
    return getMockMapStations();
  }
  if (cleanPath.includes("/map/high-risk-polygons")) {
    return getMockHighRiskPolygons();
  }
  if (cleanPath.includes("/map/flood-zones")) {
    return getMockFloodZones();
  }
  if (cleanPath.includes("/sensors")) {
    return getMockSensors();
  }
  if (cleanPath.includes("/facilities") || cleanPath.includes("/emergency")) {
    return getMockFacilities();
  }
  if (cleanPath.includes("/evacuation-routes")) {
    return getMockEvacuationRoutes();
  }
  if (cleanPath.includes("/map/inspect-coordinate")) {
    const params = new URLSearchParams(cleanPath.split("?")[1] || "");
    const lat = parseFloat(params.get("lat")) || 27.04;
    const lon = parseFloat(params.get("lon")) || 88.26;
    return generateLocationPredictionFallback("", lat, lon);
  }

  return { status: "ok", mode: "client_fallback", timestamp: new Date().toISOString() };
}

export const get = (p) => api(p);
export const post = (p, body) => api(p, { method: "POST", body: body instanceof FormData ? body : JSON.stringify(body) });
export const patch = (p, body) => api(p, { method: "PATCH", body: JSON.stringify(body) });

// High-level API methods exported for components
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

export const getEarthquakeProfile = (query, latitude, longitude) =>
  post("/api/earthquake/profile", { query, latitude, longitude });

export const getNERRiskIndices = () => get("/api/ner/risk-indices");
export const getMLMetrics = () => get("/api/ml/metrics");
export const getStateAnalytics = () => get("/api/dashboard/state-analytics");
