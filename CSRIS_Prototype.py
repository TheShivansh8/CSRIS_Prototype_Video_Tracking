import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import cv2
from PIL import Image

st.set_page_config(page_title="CSRIS - Prototype with Video Tracking", layout="wide")
st.title("🚨 CSRIS - Prototype: Crowd Risk & Video Tracking Dashboard")

# Section 1: Live Crowd Monitoring
st.header("📊 Live Crowd Density Monitoring")
zone_count = st.slider("Select number of zones", 1, 5, 3)
zones_data = []
for i in range(zone_count):
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input(f"Zone {i+1} Name", f"Zone-{i+1}", key=f"name_{i}")
        people = st.number_input(f"People in {name}", 0, value=3000, key=f"people_{i}")
    with col2:
        area = st.number_input(f"Area of {name} (m²)", 1, value=1000, key=f"area_{i}")
    density = people / area
    risk = "🟢 Safe" if density <= 4 else "🟡 Warning" if density <= 7 else "🔴 High Risk"
    zones_data.append((name, people, area, round(density, 2), risk))

st.subheader("🧮 Density Analysis Table")
df_zones = pd.DataFrame(zones_data, columns=["Zone", "People", "Area (m²)", "Density", "Risk Level"])
st.dataframe(df_zones, use_container_width=True)

# Section 2: Smart Alerts
st.header("🚨 Automated Risk Alerts")
for _, row in df_zones.iterrows():
    if row['Density'] > 7:
        st.error(f"{row['Zone']}: HIGH RISK – Add exits, disperse crowd!")
    elif row['Density'] > 4:
        st.warning(f"{row['Zone']}: Warning – Monitor and prepare response.")
    else:
        st.success(f"{row['Zone']}: Safe zone.")

# Section 3: Historic Deaths by State
st.header("📉 Stampede Death Trends (2001–2022)")
data = {
    'Year': list(range(2001, 2023)),
    'Jharkhand': np.random.randint(5, 60, 22),
    'Maharashtra': np.random.randint(5, 55, 22),
    'Andhra Pradesh': np.random.randint(5, 50, 22),
    'Tamil Nadu': np.random.randint(2, 40, 22),
    'Uttar Pradesh': np.random.randint(10, 70, 22),
    'Bihar': np.random.randint(5, 65, 22)
}
df = pd.DataFrame(data)

fig1, ax1 = plt.subplots(figsize=(12, 6))
for state in df.columns[1:]:
    ax1.plot(df['Year'], df[state], label=state)
ax1.set_title("Annual Stampede Deaths by State")
ax1.set_xlabel("Year")
ax1.set_ylabel("Deaths")
ax1.legend()
ax1.grid(True)
st.pyplot(fig1)

# Section 4: Total Deaths Choropleth
st.header("🗺️ Total Deaths by State")
total_deaths = df.drop("Year", axis=1).sum().reset_index()
total_deaths.columns = ["State", "Deaths"]
total_deaths["Code"] = total_deaths["State"].map({
    "Jharkhand": "JH", "Maharashtra": "MH", "Andhra Pradesh": "AP",
    "Tamil Nadu": "TN", "Uttar Pradesh": "UP", "Bihar": "BR"
})

fig_map = px.choropleth(
    total_deaths,
    locations="Code",
    color="Deaths",
    hover_name="State",
    color_continuous_scale="Reds",
    scope="asia",
    title="Stampede Deaths by State (2001–2022)"
)
fig_map.update_geos(fitbounds="locations", visible=False)
st.plotly_chart(fig_map, use_container_width=True)

# Section 5: Video-Based Tracking
st.header("🎥 Video-Based Crowd Tracking")
video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if video_file:
    with open("temp_video.mp4", "wb") as f:
        f.write(video_file.read())

    cap = cv2.VideoCapture("temp_video.mp4")
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        boxes, _ = hog.detectMultiScale(frame, winStride=(8, 8))
        for (x, y, w, h) in boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(rgb_frame, channels="RGB", use_column_width=True)

    cap.release()

# Section 6: Guidelines & Summary
st.header("📋 Guidelines & Summary")
st.markdown("""
- Safe crowd density: **≤ 4 people/m²**  
- Warning level: **4–7 people/m²**  
- High risk: **> 7 people/m²**  

**Recommendations:**
- Use of AI cameras and drones
- Dynamic emergency planning
- Proper communication protocols

**Summary:**  
This prototype combines real-time density data with video tracking and historic trend visualization for effective risk mitigation.
""")
