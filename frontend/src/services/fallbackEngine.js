/**
 * fallbackEngine.js — Complete client-side disaster intelligence & geocoding engine
 */

export const MOCK_GAZETTEER = [
  { name: "Darjeeling", state: "West Bengal", lat: 27.041, lon: 88.2663, elevation_m: 2045, hazard_type: "Landslide Risk Zone", risk_score: 84.5 },
  { name: "Wayanad", state: "Kerala", lat: 11.6854, lon: 76.132, elevation_m: 950, hazard_type: "Debris Flow & Slope Instability", risk_score: 89.2 },
  { name: "Gangtok", state: "Sikkim", lat: 27.3389, lon: 88.6065, elevation_m: 1650, hazard_type: "High Seismic & Slip Zone", risk_score: 78.6 },
  { name: "Shimla", state: "Himachal Pradesh", lat: 31.1048, lon: 77.1734, elevation_m: 2276, hazard_type: "Hill Slope Failure", risk_score: 76.4 },
  { name: "Guwahati", state: "Assam", lat: 26.1445, lon: 91.7362, elevation_m: 55, hazard_type: "Brahmaputra Flood Plain", risk_score: 82.1 },
  { name: "Kolkata", state: "West Bengal", lat: 22.5726, lon: 88.3639, elevation_m: 9, hazard_type: "Urban Inundation & Cyclone Surge", risk_score: 65.0 },
  { name: "Shillong", state: "Meghalaya", lat: 25.5788, lon: 91.8933, elevation_m: 1525, hazard_type: "Flash Flood & Slope Erosion", risk_score: 72.3 },
  { name: "Itanagar", state: "Arunachal Pradesh", lat: 27.0844, lon: 93.6053, elevation_m: 320, hazard_type: "Foothill Flash Surge", risk_score: 71.0 },
  { name: "Imphal", state: "Manipur", lat: 24.817, lon: 93.9368, elevation_m: 786, hazard_type: "Valley Inundation", risk_score: 68.4 },
  { name: "Kohima", state: "Nagaland", lat: 25.6751, lon: 94.1086, elevation_m: 1444, hazard_type: "Active Subsidence Zone", risk_score: 79.5 },
  { name: "Aizawl", state: "Mizoram", lat: 23.7271, lon: 92.7176, elevation_m: 1132, hazard_type: "Ridge Slope Failure", risk_score: 81.3 },
  { name: "Agartala", state: "Tripura", lat: 23.8315, lon: 91.2868, elevation_m: 12, hazard_type: "River Outburst Flood", risk_score: 64.2 },
  { name: "Dehradun", state: "Uttarakhand", lat: 30.3165, lon: 78.0322, elevation_m: 435, hazard_type: "Cloudburst & Flash Flood", risk_score: 83.1 },
  { name: "Joshimath", state: "Uttarakhand", lat: 30.5568, lon: 79.566, elevation_m: 1890, hazard_type: "Severe Land Subsidence", risk_score: 94.8 },
  { name: "Srinagar", state: "Jammu & Kashmir", lat: 34.0837, lon: 74.7973, elevation_m: 1585, hazard_type: "Jhelum Flood Basin", risk_score: 75.2 }
];

export function searchGazetteer(query = "") {
  if (!query) return MOCK_GAZETTEER;
  const q = query.toLowerCase();
  return MOCK_GAZETTEER.filter(
    (item) => item.name.toLowerCase().includes(q) || item.state.toLowerCase().includes(q)
  );
}

export function generateLocationPredictionFallback(query = "", lat = null, lon = null) {
  const match = MOCK_GAZETTEER.find(
    (g) => query && g.name.toLowerCase().includes(query.toLowerCase())
  );

  const finalLat = lat ?? (match ? match.lat : 27.041);
  const finalLon = lon ?? (match ? match.lon : 88.2663);
  const placeName = match ? match.name : query || "Target Location Grid";

  const isHilly = finalLat > 25 && finalLon < 96;
  const lsScore = Math.min(95, Math.max(15, Math.round((isHilly ? 72 : 35) + Math.random() * 20)));
  const flScore = Math.min(95, Math.max(15, Math.round((!isHilly ? 75 : 45) + Math.random() * 20)));

  return {
    location: placeName,
    state: match ? match.state : "India Grid Sector",
    coordinates: { latitude: finalLat, longitude: finalLon },
    elevation_m: match ? match.elevation_m : 850,
    landslide_risk: {
      score: lsScore,
      level: lsScore > 75 ? "CRITICAL" : lsScore > 50 ? "WARNING" : "SAFE",
      probability: (lsScore / 100).toFixed(2),
      slope_deg: isHilly ? 34.2 : 8.5,
      pore_pressure_kpa: isHilly ? 32.4 : 12.1,
      soil_moisture_pct: 68.5,
    },
    flood_risk: {
      score: flScore,
      level: flScore > 75 ? "CRITICAL" : flScore > 50 ? "WARNING" : "SAFE",
      probability: (flScore / 100).toFixed(2),
      distance_to_river_m: 650,
      catchment_rainfall_48h_mm: 145.0,
    },
    weather: {
      temp_c: 21.5,
      rainfall_24h_mm: 84.2,
      humidity_pct: 88,
      wind_speed_kmh: 14.2,
      condition: "Heavy Rain & Cloud Cover",
    },
    evacuation_centers: [
      { name: `${placeName} Community Hall Safety Shelter`, distance_km: 1.8, capacity: 450, phone: "1070 / 112" },
      { name: "District Disaster Response Base", distance_km: 3.4, capacity: 800, phone: "1078" },
    ],
  };
}

export function generateRouteAnalysisFallback(origin = "Guwahati", destination = "Gangtok") {
  return {
    origin,
    destination,
    total_distance_km: 520,
    estimated_travel_time_hours: 11.5,
    overall_hazard_index: 76.4,
    status: "HIGH_RISK_WARNING",
    high_hazard_segments: [
      { sector: "Corridor Segment 3 (Siliguri - Sevoke Road)", hazard: "Active Landslide Failure Zone", risk_score: 86.5, recommendation: "Use alternative bypassing ridge route" },
      { sector: "Corridor Segment 5 (Rangpo Border Checkpost)", hazard: "Teesta River Flash Erosion", risk_score: 79.2, recommendation: "Exercise extreme caution, avoid night travel" },
    ],
  };
}

export function getMockDashboardOverview() {
  return {
    active_alerts_count: 14,
    total_monitored_stations: 128,
    high_risk_zones_count: 9,
    ner_composite_risk_score: 74.8,
    regional_summary: [
      { state: "Sikkim", risk_level: "CRITICAL", active_warnings: 4 },
      { state: "West Bengal (Hills)", risk_level: "HIGH", active_warnings: 3 },
      { state: "Assam", risk_level: "WARNING", active_warnings: 4 },
      { state: "Arunachal Pradesh", risk_level: "WARNING", active_warnings: 3 },
    ],
  };
}

export function getMockMLMetrics() {
  return {
    model_name: "XGBoost + Random Forest Hybrid Multi-Hazard Ensemble",
    landslide_auc_roc: 0.942,
    flood_auc_roc: 0.928,
    accuracy: 93.6,
    f1_score: 0.925,
    last_trained: "2026-09-10",
  };
}

export function getMockLithologies() {
  return [
    { code: 1, name: "Weathered Shale & Mudstone" },
    { code: 2, name: "Quartzite & Gneiss" },
    { code: 3, name: "Alluvial Silt & Sand" },
    { code: 4, name: "Sandstone & Conglomerate" },
  ];
}

export function getMockMapStations() {
  return MOCK_GAZETTEER.map((g, idx) => ({
    id: `stn_${idx + 1}`,
    name: `${g.name} EWS Station`,
    lat: g.lat,
    lon: g.lon,
    risk_score: g.risk_score,
    status: g.risk_score > 80 ? "ALERT" : "NORMAL",
  }));
}

export function getMockHighRiskPolygons() {
  return [];
}

export function getMockFloodZones() {
  return [];
}

export function getMockSensors() {
  return [];
}

export function getMockFacilities() {
  return [];
}

export function getMockEvacuationRoutes() {
  return [];
}
