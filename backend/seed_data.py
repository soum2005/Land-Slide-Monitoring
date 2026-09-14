"""Seed NER states, stations, sensors, history, facilities, and demo users."""

from __future__ import annotations

import random
from datetime import datetime, timedelta

from auth_utils import hash_password
from models import (
    Alert,
    EmergencyFacility,
    ForecastRecord,
    HistoricalLandslide,
    IncidentReport,
    MonitoringLocation,
    RainfallRecord,
    Sensor,
    SensorReading,
    SystemSetting,
    User,
)

rng = random.Random(2026)

STATIONS = [
    ("Gangtok NH-10 Corridor", "Sikkim", "East Sikkim", "NH-10", 27.3389, 88.6065, 1650, 38, "phyllite", 62000),
    ("Pakyong Ridge", "Sikkim", "Pakyoung", "NH-717A", 27.2313, 88.5971, 1400, 34, "gneiss", 18000),
    ("Mangan North Sikkim", "Sikkim", "Mangan", "NH-310", 27.4975, 88.534, 1240, 41, "weathered_shale", 9000),
    ("Namchi South Ridge", "Sikkim", "Namchi", "NH-10", 27.1667, 88.35, 1315, 36, "phyllite", 12000),
    ("Guwahati Foothills", "Assam", "Kamrup Metro", "NH-27", 26.1445, 91.7362, 55, 18, "alluvium", 1100000),
    ("Dima Hasao Haflong", "Assam", "Dima Hasao", "NH-27", 25.1641, 93.0154, 966, 42, "weathered_shale", 22000),
    ("Paglajhora Cutting", "Assam", "Kamrup", "NH-10", 26.72, 88.45, 220, 39, "weathered_shale", 8000),
    ("Lumding Ghat", "Assam", "Hojai", "NH-27", 25.75, 93.17, 148, 28, "sandstone", 15000),
    ("Shillong Peak Road", "Meghalaya", "East Khasi Hills", "NH-6", 25.5788, 91.8933, 1496, 33, "sandstone", 143000),
    ("Cherrapunji Escarpment", "Meghalaya", "East Khasi Hills", "NH-206", 25.3, 91.7, 1484, 46, "limestone", 14000),
    ("Jowai Plateau", "Meghalaya", "West Jaintia Hills", "NH-6", 25.45, 92.2, 1380, 31, "sandstone", 28000),
    ("Tura West Garo", "Meghalaya", "West Garo Hills", "NH-217", 25.514, 90.202, 349, 27, "laterite", 74000),
    ("Champhai Border Road", "Mizoram", "Champhai", "NH-2", 23.4564, 93.328, 1678, 44, "sandstone", 32000),
    ("Aizawl Hill City", "Mizoram", "Aizawl", "NH-6", 23.7271, 92.7176, 1132, 40, "sandstone", 293000),
    ("Lunglei Ridge", "Mizoram", "Lunglei", "NH-6", 22.88, 92.73, 722, 37, "weathered_shale", 57000),
    ("Kohima NH-29", "Nagaland", "Kohima", "NH-29", 25.6751, 94.1086, 1444, 39, "sandstone", 99000),
    ("Dimapur Foothill", "Nagaland", "Dimapur", "NH-29", 25.909, 93.727, 154, 16, "alluvium", 122000),
    ("Mokokchung Hills", "Nagaland", "Mokokchung", "NH-2", 26.322, 94.513, 1325, 35, "gneiss", 35000),
    ("Itanagar Capital Complex", "Arunachal Pradesh", "Papum Pare", "NH-415", 27.0844, 93.6053, 320, 29, "weathered_shale", 59000),
    ("Malinithan Slope", "Arunachal Pradesh", "Lower Siang", "NH-13", 27.666, 94.7, 210, 36, "weathered_shale", 6000),
    ("Tawang Corridor", "Arunachal Pradesh", "Tawang", "NH-13", 27.586, 91.859, 3048, 33, "gneiss", 11000),
    ("Roing Lower Dibang", "Arunachal Pradesh", "Lower Dibang Valley", "NH-13", 28.14, 95.84, 390, 38, "weathered_shale", 8000),
    ("Imphal Valley Edge", "Manipur", "Imphal West", "NH-2", 24.817, 93.9368, 786, 22, "alluvium", 268000),
    ("Tupul Railway Alignment", "Manipur", "Noney", "NH-37", 24.86, 93.64, 890, 47, "weathered_shale", 9000),
    ("Senapati Highland", "Manipur", "Senapati", "NH-2", 25.27, 94.02, 1784, 41, "sandstone", 19000),
    ("Ukhrul Ridge", "Manipur", "Ukhrul", "NH-202", 25.12, 94.36, 1662, 43, "sandstone", 14000),
    ("Agartala Urban Fringe", "Tripura", "West Tripura", "NH-8", 23.8315, 91.2868, 16, 12, "alluvium", 400000),
    ("Dhalai Ambassa", "Tripura", "Dhalai", "NH-8", 23.92, 91.85, 90, 24, "laterite", 18000),
    ("Unakoti Hills", "Tripura", "Unakoti", "NH-8", 24.32, 92.0, 140, 28, "sandstone", 12000),
    ("Silchar Barak Approach", "Assam", "Cachar", "NH-37", 24.8333, 92.7789, 22, 14, "alluvium", 172000),
    ("Bomdila Inner Line", "Arunachal Pradesh", "West Kameng", "NH-13", 27.265, 92.412, 2415, 34, "gneiss", 7000),
    ("Phek District Road", "Nagaland", "Phek", "NH-29", 25.67, 94.47, 1524, 37, "sandstone", 16000),
]

HISTORY = [
    ("Tupul Railway Cut Slope Collapse", "Manipur", "Noney", 2022, "2022-06-30", 24.86, 93.64, 61, "NF Railway cut-and-cover tunnel and workers' camp destroyed", "extreme monsoon rainfall", "weathered Disang shale", "Catastrophic", "Among the deadliest recent NER construction-related landslides."),
    ("Dima Hasao Monsoon Corridor", "Assam", "Dima Hasao", 2022, "2022-05-20", 25.164, 93.015, 8, "NH-27 and BG railway blocked for weeks", "cloudburst / prolonged rain", "highly weathered shale-sandstone", "High", "Recurring corridor isolation of Barak Valley."),
    ("Paglajhora Recurring Slide", "West Bengal/Sikkim approach", "Darjeeling-Kamrup corridor", 2011, "2011-07-15", 26.85, 88.28, 0, "NH-10 repeatedly severed", "monsoon saturation", "colluvium over phyllite", "High", "Chronic bottleneck on the Sikkim lifeline."),
    ("Malinithan Temple Slope Failure", "Arunachal Pradesh", "Lower Siang", 2017, "2017-08-11", 27.666, 94.7, 3, "Approach road and shrine precinct damaged", "intense rainfall", "Siwalik sandstone-shale", "Moderate", "Heritage and highway exposure."),
    ("Subansiri Hydropower Rim Slides", "Arunachal Pradesh", "Lower Subansiri", 2023, "2023-07-02", 27.55, 94.15, 2, "Access roads and cofferdam approaches", "drawdown + rainfall", "Himalayan sedimentary sequence", "High", "Reservoir-rim instability."),
    ("Haflong Bazaar Slide", "Assam", "Dima Hasao", 2018, "2018-06-14", 25.17, 93.02, 5, "Market and district HQ access cut", "24h rain > 180mm", "weathered shale", "High", ""),
    ("Cherrapunji Escarpment Rockfall", "Meghalaya", "East Khasi Hills", 2019, "2019-09-08", 25.28, 91.72, 4, "Tourism road buried", "extreme orographic rain", "limestone-sandstone", "High", "World-class rainfall totals."),
    ("Aizawl Laipuitlang Collapse", "Mizoram", "Aizawl", 2013, "2013-05-11", 23.74, 92.72, 17, "Residential buildings collapsed", "rain + unplanned cut slopes", "sandstone", "Catastrophic", "Urban hillside construction risk."),
    ("Kohima NH-29 Cutting", "Nagaland", "Kohima", 2020, "2020-07-22", 25.66, 94.11, 2, "Capital access delayed 36h", "monsoon", "Disang formation", "Moderate", ""),
    ("Imphal-Jiribam Highway Slide", "Manipur", "Noney", 2021, "2021-08-03", 24.78, 93.55, 6, "NH-37 blocked", "prolonged rain", "shale", "High", ""),
    ("Tawang Sela Corridor", "Arunachal Pradesh", "Tawang", 2021, "2021-09-19", 27.5, 92.1, 7, "Army logistics delayed", "late monsoon + freeze-thaw", "gneiss-schist", "High", ""),
    ("Gangtok Chandmari Historic", "Sikkim", "East Sikkim", 1997, "1997-06-08", 27.34, 88.62, 34, "Urban ward destruction", "cloudburst", "phyllite-schist", "Catastrophic", "Benchmark urban landslide."),
    ("North Sikkim Mangan 2023", "Sikkim", "Mangan", 2023, "2023-06-20", 27.5, 88.53, 12, "Teesta basin roads and hydropower", "cloudburst after glacial lake stress", "Himalayan metasediments", "Catastrophic", "Post-GLOF landscape."),
    ("Shillong Laitumkhrah Cut", "Meghalaya", "East Khasi Hills", 2016, "2016-07-29", 25.58, 91.89, 1, "Urban lane collapse", "pipe leakage + rain", "sandstone", "Moderate", ""),
    ("Champhai Border Trade Road", "Mizoram", "Champhai", 2020, "2020-08-16", 23.45, 93.33, 3, "Indo-Myanmar trade stalled", "monsoon", "sandstone", "High", ""),
    ("Senapati District HQ Road", "Manipur", "Senapati", 2019, "2019-07-07", 25.27, 94.03, 4, "District connectivity lost 48h", "rain", "sandstone", "High", ""),
    ("Ukhrul Shirui Approach", "Manipur", "Ukhrul", 2021, "2021-06-25", 25.13, 94.36, 2, "Tourism road", "rain", "sandstone", "Moderate", ""),
    ("Mokokchung Village Slide", "Nagaland", "Mokokchung", 2018, "2018-08-02", 26.33, 94.52, 5, "Habitations relocated", "rain", "gneiss", "High", ""),
    ("Phek Agricultural Terrace", "Nagaland", "Phek", 2017, "2017-07-18", 25.67, 94.48, 1, "Paddy terraces lost", "rain", "sandstone", "Moderate", ""),
    ("Itanagar Naharlagun Cutting", "Arunachal Pradesh", "Papum Pare", "2015", "2015-06-21", 27.1, 93.62, 3, "Capital road", "rain + excavation", "weathered shale", "Moderate", ""),
    ("Roing-Anini Road", "Arunachal Pradesh", "Lower Dibang Valley", 2022, "2022-07-11", 28.2, 95.9, 6, "Frontier road blocked", "extreme rain", "shale", "High", ""),
    ("Bomdila Market Slope", "Arunachal Pradesh", "West Kameng", 2014, "2014-08-09", 27.26, 92.41, 2, "Bazaar retaining wall", "rain", "gneiss", "Moderate", ""),
    ("Lunglei College Hill", "Mizoram", "Lunglei", 2016, "2016-06-30", 22.88, 92.74, 0, "Campus access", "rain", "shale", "Low", ""),
    ("Dhalai Ambassa Ghat", "Tripura", "Dhalai", 2018, "2018-09-12", 23.92, 91.85, 2, "NH-8 shoulder failure", "cyclonic rain", "laterite", "Moderate", ""),
    ("Unakoti Pilgrim Path", "Tripura", "Unakoti", 2015, "2015-07-04", 24.32, 92.01, 0, "Heritage trail", "rain", "sandstone", "Low", ""),
    ("Agartala Airport Approach Fill", "Tripura", "West Tripura", 2020, "2020-06-18", 23.89, 91.24, 0, "Embankment slump", "rain", "alluvium", "Low", ""),
    ("Silchar Barak Embankment", "Assam", "Cachar", 2022, "2022-06-05", 24.82, 92.8, 3, "Flood + bank failure", "rain + river undercutting", "alluvium", "High", ""),
    ("Jowai Pynthor Slide", "Meghalaya", "West Jaintia Hills", 2019, "2019-07-21", 25.45, 92.2, 4, "Coal belt road", "rain + mining", "sandstone", "High", ""),
    ("Tura Tura-Dalu Road", "Meghalaya", "West Garo Hills", 2021, "2021-08-14", 25.5, 90.22, 2, "NH-217", "rain", "laterite", "Moderate", ""),
    ("Dimapur Foothill Debris", "Nagaland", "Dimapur", 2016, "2016-09-01", 25.91, 93.73, 0, "Drain choking", "rain", "alluvium", "Low", ""),
    ("Pakyong Airport Cut Slope", "Sikkim", "Pakyong", 2019, "2019-07-12", 27.23, 88.59, 0, "Airport perimeter", "rain + cut slope", "gneiss", "Moderate", ""),
    ("Namchi Ravangla Road", "Sikkim", "Namchi", 2020, "2020-08-08", 27.3, 88.36, 1, "Tourism circuit", "rain", "phyllite", "Moderate", ""),
    ("Imphal Kangchup Road", "Manipur", "Imphal West", 2018, "2018-07-19", 24.85, 93.85, 2, "Western bypass", "rain", "shale", "Moderate", ""),
    ("Lumding Railway Catchment", "Assam", "Hojai", 2010, "2010-06-22", 25.75, 93.17, 9, "BG line buried", "monsoon", "sandstone-shale", "High", "Historic NE railway disruption."),
    ("Guwahati Nilachal Hills", "Assam", "Kamrup Metro", 2014, "2014-09-06", 26.15, 91.74, 5, "Kamakhya approach", "rain + urban loading", "granite-gneiss residual", "High", ""),
    ("Mangan Toong Slide", "Sikkim", "Mangan", 2016, "2016-07-03", 27.52, 88.5, 3, "North district isolation", "rain", "schist", "High", ""),
    ("Churachandpur Highland", "Manipur", "Churachandpur", 2023, "2023-07-28", 24.33, 93.7, 4, "District roads", "rain", "sandstone", "High", ""),
    ("Mon District Frontier Road", "Nagaland", "Mon", 2021, "2021-07-09", 26.75, 95.02, 2, "Border road", "rain", "sandstone", "Moderate", ""),
    ("Khonsa Tirap", "Arunachal Pradesh", "Tirap", 2019, "2019-08-17", 27.02, 95.5, 3, "District HQ access", "rain", "shale", "High", ""),
    ("Siaha Tipa Road", "Mizoram", "Siaha", 2020, "2020-07-26", 22.49, 92.98, 1, "Southern Mizoram", "rain", "sandstone", "Moderate", ""),
    ("East Jaintia Limestone Bench", "Meghalaya", "East Jaintia Hills", 2018, "2018-06-11", 25.2, 92.4, 6, "Mining bench failure", "rain + mining", "limestone", "High", ""),
]


def seed(db):
    if not db.query(SystemSetting).first():
        db.add(SystemSetting())

    # Seed or update default admin and team users
    default_users = [
        ("epsita", "Epsita Maity", "epsitamaity629@gmail.com", "Admin", "Directorate of Disaster Management"),
        ("epsitamaity629@gmail.com", "Epsita Maity", "epsitamaity629@gmail.com", "Admin", "Directorate of Disaster Management"),
        ("soumya", "Soumya Saha", "soumyasaha205@gmail.com", "Admin", "NER Landslide Engineering Directorate"),
        ("soumyasaha205@gmail.com", "Soumya Saha", "soumyasaha205@gmail.com", "Admin", "NER Landslide Engineering Directorate"),
        ("sanjana", "Sanjana Jana", "sanjanajana464@gmail.com", "Field Officer", "Field Response Team"),
        ("monira", "Monira Protappur", "monira.protappur@gmail.com", "Citizen", "Public"),
    ]
    for uname, fname, email, role, org in default_users:
        u = db.query(User).filter(User.username == uname).first()
        if not u:
            u = db.query(User).filter(User.email == email).first()
        if not u:
            db.add(
                User(
                    username=uname,
                    full_name=fname,
                    email=email,
                    hashed_password=hash_password("password123"),
                    role=role,
                    organization=org,
                    is_active=True,
                )
            )
        else:
            # Enforce target role for core admin emails
            if email in ("epsitamaity629@gmail.com", "soumyasaha205@gmail.com") and u.role != "Admin":
                u.role = "Admin"
    db.commit()

    if db.query(MonitoringLocation).count() > 0:
        db.commit()
        return True


    locations = []
    for row in STATIONS:
        name, state, district, hwy, lat, lon, elev, slope, lith, pop = row
        loc = MonitoringLocation(
            name=name,
            state=state,
            district=district,
            highway=hwy,
            latitude=lat,
            longitude=lon,
            elevation_m=elev,
            slope_deg=slope,
            lithology=lith,
            risk_score=round(20 + slope * 0.7 + (40 if "shale" in lith or "phyllite" in lith else 10), 1),
            risk_level="Advisory",
            population_exposed=pop,
            notes=f"Critical monitoring on {hwy}",
        )
        db.add(loc)
        locations.append(loc)
    db.flush()

    sensor_types = [
        ("rain_gauge", "mm"),
        ("soil_moisture", "%"),
        ("tilt", "deg"),
        ("inclinometer", "mm"),
        ("ground_displacement", "mm"),
        ("temperature", "C"),
        ("humidity", "%"),
        ("pore_pressure", "kPa"),
    ]
    sensors = []
    sid = 1
    for loc in locations:
        chosen = sensor_types if loc.slope_deg > 30 else sensor_types[:6]
        for stype, unit in chosen:
            s = Sensor(
                location_id=loc.id,
                sensor_code=f"NER-{loc.state[:3].upper()}-{sid:03d}",
                sensor_type=stype,
                status=rng.choice(["online", "online", "online", "warning", "online"]),
                battery_pct=round(rng.uniform(54, 99), 1),
                last_calibrated="2026-04-18",
                latitude=loc.latitude + rng.uniform(-0.01, 0.01),
                longitude=loc.longitude + rng.uniform(-0.01, 0.01),
            )
            db.add(s)
            sensors.append((s, unit))
            sid += 1
    db.flush()

    now = datetime.utcnow()
    for s, unit in sensors:
        base = {"rain_gauge": 2.2, "soil_moisture": 46, "tilt": 0.7, "inclinometer": 3.1, "ground_displacement": 2.4, "temperature": 21, "humidity": 78, "pore_pressure": 18}.get(
            s.sensor_type, 1
        )
        for h in range(24):
            db.add(
                SensorReading(
                    sensor_id=s.id,
                    timestamp=now - timedelta(hours=23 - h),
                    value=round(base + rng.uniform(-0.4, 1.8) * (1 + 0.04 * h), 3),
                    unit=unit,
                )
            )

    for loc in locations:
        acc7 = 0
        acc24 = 0
        for h in range(168):
            hourly = max(0, rng.gauss(1.1 if loc.slope_deg > 30 else 0.6, 1.4))
            if loc.district in ("East Khasi Hills", "Champhai", "East Sikkim"):
                hourly *= 1.6
            acc7 += hourly
            if h >= 144:
                acc24 += hourly
            if h % 6 == 0 or h >= 144:
                intensity = "violent" if hourly > 12 else "heavy" if hourly > 6 else "moderate" if hourly > 2 else "light"
                db.add(
                    RainfallRecord(
                        location_id=loc.id,
                        timestamp=now - timedelta(hours=167 - h),
                        hourly_mm=round(hourly, 2),
                        cumulative_24h_mm=round(acc24 if h >= 144 else hourly * 8, 1),
                        cumulative_7d_mm=round(acc7, 1),
                        intensity=intensity,
                    )
                )
        rain24 = acc24
        loc.risk_score = round(min(96, loc.slope_deg * 0.9 + rain24 * 0.15 + rng.uniform(0, 12)), 1)
        loc.risk_level = "Emergency" if loc.risk_score >= 80 else "Warning" if loc.risk_score >= 60 else "Advisory" if loc.risk_score >= 40 else "Info"
        for hz, factor in ((6, 0.35), (12, 0.55), (24, 1.0), (72, 2.1)):
            db.add(
                ForecastRecord(
                    location_id=loc.id,
                    horizon_hours=hz,
                    rainfall_mm=round(rain24 * factor * rng.uniform(0.8, 1.2), 1),
                    landslide_probability=round(min(0.95, loc.risk_score / 120 * factor), 3),
                    risk_score=round(min(99, loc.risk_score * (0.7 + 0.2 * factor)), 1),
                )
            )

    for h in HISTORY:
        db.add(
            HistoricalLandslide(
                name=h[0],
                state=h[1],
                district=h[2],
                year=int(h[3]) if str(h[3]).isdigit() else 2015,
                event_date=h[4],
                latitude=h[5],
                longitude=h[6],
                casualties=h[7],
                infrastructure_damage=h[8],
                trigger_cause=h[9],
                geological_formation=h[10],
                severity=h[11],
                notes=h[12],
            )
        )

    facilities = [
        ("NDRF 1st Bn Guwahati", "ndrf_base", "Assam", "Kamrup Metro", 26.18, 91.78, 450, "0361-2840284", "Guwahati response hub"),
        ("NDRF 12th Bn Doimukh", "ndrf_base", "Arunachal Pradesh", "Papum Pare", 27.14, 93.75, 380, "0360-2277334", "Itanagar / Arunachal deployment"),
        ("STNM Hospital Gangtok", "hospital", "Sikkim", "East Sikkim", 27.33, 88.61, 500, "03592-202044", "State referral"),
        ("NEIGRIHMS Shillong", "hospital", "Meghalaya", "East Khasi Hills", 25.59, 91.94, 600, "0364-2538029", "Regional institute"),
        ("RIMS Imphal", "hospital", "Manipur", "Imphal West", 24.81, 93.91, 700, "0385-2411484", "State medical college"),
        ("Civil Hospital Aizawl", "hospital", "Mizoram", "Aizawl", 23.73, 92.72, 280, "0389-2322117", ""),
        ("Naga Hospital Kohima", "hospital", "Nagaland", "Kohima", 25.67, 94.11, 250, "0370-2242216", ""),
        ("GBP Hospital Agartala", "hospital", "Tripura", "West Tripura", 23.84, 91.28, 400, "0381-2325746", ""),
        ("GMCH Guwahati", "hospital", "Assam", "Kamrup Metro", 26.15, 91.77, 900, "0361-2529457", ""),
        ("TRIHMS Naharlagun", "hospital", "Arunachal Pradesh", "Papum Pare", 27.1, 93.7, 320, "0360-2245260", ""),
        ("Tathangchen Relief Camp", "shelter", "Sikkim", "East Sikkim", 27.35, 88.62, 800, "03592-202018", "Municipal relief"),
        ("Haflong Indoor Stadium Shelter", "shelter", "Assam", "Dima Hasao", 25.17, 93.02, 600, "03673-236224", ""),
        ("Champhai Community Hall", "shelter", "Mizoram", "Champhai", 23.46, 93.33, 400, "03831-234567", ""),
        ("Senapati DEOC Shelter", "shelter", "Manipur", "Senapati", 25.27, 94.02, 350, "03871-222214", ""),
        ("Kohima Local Ground Camp", "shelter", "Nagaland", "Kohima", 25.67, 94.1, 500, "0370-2290201", ""),
        ("Shillong Polo Ground Camp", "shelter", "Meghalaya", "East Khasi Hills", 25.57, 91.88, 1200, "0364-2224282", ""),
        ("Itanagar IG Park Shelter", "shelter", "Arunachal Pradesh", "Papum Pare", 27.09, 93.61, 700, "0360-2212515", ""),
        ("Agartala Astabal Shelter", "shelter", "Tripura", "West Tripura", 23.83, 91.27, 900, "0381-2416046", ""),
        ("Sikkim Police HQ", "police", "Sikkim", "East Sikkim", 27.33, 88.61, 0, "100", ""),
        ("Meghalaya Fire & Emergency", "fire", "Meghalaya", "East Khasi Hills", 25.58, 91.89, 0, "101", ""),
        ("NH-10 Paglajhora Block", "blocked_route", "Assam", "Kamrup", 26.72, 88.45, 0, "", "Seasonal cut; use NH-31/27 bypass"),
        ("NH-29 Kohima Cutting", "blocked_route", "Nagaland", "Kohima", 25.66, 94.12, 0, "", "Single-lane regulated"),
        ("NH-37 Tupul Sector", "blocked_route", "Manipur", "Noney", 24.86, 93.64, 0, "", "Use Imphal-Kangpokpi alternative"),
        ("NH-27 Jatinga Ghat", "blocked_route", "Assam", "Dima Hasao", 25.2, 93.0, 0, "", "Night closure during heavy rain"),
    ]
    for f in facilities:
        db.add(
            EmergencyFacility(
                name=f[0],
                facility_type=f[1],
                state=f[2],
                district=f[3],
                latitude=f[4],
                longitude=f[5],
                capacity=f[6],
                contact=f[7],
                notes=f[8],
                is_blocked=f[1] == "blocked_route",
            )
        )

    for loc in locations:
        if loc.risk_score >= 55:
            level = loc.risk_level
            db.add(
                Alert(
                    location_id=loc.id,
                    level=level,
                    title=f"{level}: {loc.name}",
                    message=f"Elevated landslide susceptibility along {loc.highway} in {loc.district}, {loc.state}. 24h rainfall and slope sensors indicate rising pore pressure.",
                    risk_score=loc.risk_score,
                    probability=round(min(0.97, loc.risk_score / 110), 3),
                    recommended_actions="Restrict non-essential traffic; pre-position NDRF/SDRF; open identified shelters; issue last-mile SMS in vernacular.",
                    channels="Web,SMS,Email,Push",
                    is_active=True,
                )
            )

    db.add(
        IncidentReport(
            reporter_name="Field Officer Kohima",
            role="Field Officer",
            state="Nagaland",
            district="Kohima",
            latitude=25.67,
            longitude=94.11,
            severity="High",
            description="Fresh tension cracks 40m above NH-29 cutting. Minor debris on carriageway.",
            status="Verified",
        )
    )
    db.add(
        IncidentReport(
            reporter_name="Resident, Haflong",
            role="Citizen",
            state="Assam",
            district="Dima Hasao",
            latitude=25.16,
            longitude=93.02,
            severity="Moderate",
            description="Spring water bursting from slope toe near bazaar lane after overnight rain.",
            status="Under Review",
        )
    )

    db.commit()
    return True
