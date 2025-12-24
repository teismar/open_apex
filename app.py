import streamlit as st
import pandas as pd
import numpy as np
import re
from rapidocr_onnxruntime import RapidOCR
from pdf2image import convert_from_bytes
import plotly.express as px
import plotly.graph_objects as go
import io

# --- CONFIGURATION ---
st.set_page_config(page_title="OpenApex", page_icon="🏎️", layout="wide")
ocr_engine = RapidOCR()

# --- UTILS ---
def is_time_format(text):
    if re.match(r'^\d{2}\.\d{2}\.\d{4}', text): return False
    return bool(re.match(r'^(\d{1,2}[:.])?\d{1,2}[:.,]\d{3}$', text))

def is_lap_index(text):
    return text.isdigit() and len(text) <= 3

def parse_time_str(t_str):
    try:
        t_str = str(t_str).replace(',', '.')
        if ':' in t_str:
            parts = t_str.split(':')
            return float(parts[0]) * 60 + float(parts[1])
        return float(t_str)
    except:
        return None

# --- PARSING ENGINE ---
def parse_pdf_bytes(file_bytes):
    try:
        images = convert_from_bytes(file_bytes, dpi=200)
    except Exception as e:
        st.error(f"Error converting PDF: {e}")
        return []

    parsed_pages = []

    for img in images:
        result, _ = ocr_engine(np.array(img))
        if not result: continue

        lines = {}
        row_tolerance = 15
        for (bbox, text, conf) in result:
            y_center = bbox[0][1] 
            x_center = (bbox[0][0] + bbox[1][0]) / 2 
            matched = False
            for line_y in lines.keys():
                if abs(line_y - y_center) < row_tolerance:
                    lines[line_y].append({"x": x_center, "text": text})
                    matched = True
                    break
            if not matched: lines[y_center] = [{"x": x_center, "text": text}]

        sorted_y = sorted(lines.keys())
        
        # A. RESULTS TABLE
        race_rows = []
        for y in sorted_y:
            words = sorted(lines[y], key=lambda w: w['x'])
            text_list = [w['text'] for w in words]
            
            time_idx = -1
            for i, txt in enumerate(text_list):
                clean_txt = txt.replace(',', '.') 
                if is_time_format(clean_txt):
                    time_idx = i
                    text_list[i] = clean_txt
                    break
            
            if time_idx > 0:
                lap_idx = -1
                if time_idx > 0 and text_list[time_idx-1].isdigit() and len(text_list[time_idx-1]) < 4:
                    lap_idx = time_idx - 1
                
                if lap_idx != -1:
                    raw_left = text_list[:lap_idx]
                    driver_name_parts = []
                    driver_nr = ""
                    for token in reversed(raw_left):
                        if token.isdigit() and len(token) < 4:
                            if not driver_nr: driver_nr = token 
                            else: pass 
                        else: driver_name_parts.insert(0, token)
                    
                    name_str = " ".join(driver_name_parts).strip()
                    gap_str = text_list[-1] if time_idx < len(text_list) - 1 else ""
                    
                    if len(name_str) > 1:
                        race_rows.append({
                            "Nr": driver_nr,
                            "Fahrer": name_str,
                            "Rnd": text_list[lap_idx],
                            "Bestzeit": text_list[time_idx],
                            "Abstand": gap_str
                        })

        # B. LAP MATRIX
        lap_matrix = []
        col_map = [] 
        header_y = -1
        known_drivers = set([r['Fahrer'] for r in race_rows])
        
        for y in sorted_y:
            words = sorted(lines[y], key=lambda w: w['x'])
            texts = [w['text'] for w in words]
            if sum(1 for t in texts if t in known_drivers or "Nr." in t) >= 2:
                header_y = y
                for w in words:
                    clean_t = w['text'].strip()
                    if clean_t in known_drivers:
                        col_map.append({"x": w['x'], "driver": clean_t})
                break
                
        if header_y != -1 and col_map:
            for y in sorted_y:
                if y <= header_y: continue
                words = sorted(lines[y], key=lambda w: w['x'])
                texts = [w['text'] for w in words]
                
                if not texts or not is_lap_index(texts[0]): continue
                row_data = {"Lap": int(texts[0])}
                
                for w in words[1:]:
                    clean_time = w['text'].replace(',', '.')
                    if not is_time_format(clean_time): continue
                    closest_col = min(col_map, key=lambda c: abs(c['x'] - w['x']))
                    if abs(closest_col['x'] - w['x']) < 120:
                        row_data[closest_col['driver']] = clean_time
                lap_matrix.append(row_data)

        parsed_pages.append({
            'results': pd.DataFrame(race_rows),
            'laps': pd.DataFrame(lap_matrix)
        })
        
    return parsed_pages

# --- ANALYTICS ---
def render_session_analytics(df_laps, df_results, session_name):
    st.subheader(f"📊 {session_name} Analytics")
    
    if df_laps is None or df_laps.empty:
        st.warning(f"No Lap Matrix found for {session_name}.")
        return

    plot_df = df_laps[df_laps['Lap'] > 0].copy()
    
    if plot_df.empty:
        st.warning(f"No valid laps found for {session_name} after removing Lap 0.")
        return

    drivers = [c for c in plot_df.columns if c != "Lap"]
    for d in drivers:
        plot_df[d] = plot_df[d].apply(parse_time_str)
    
    long_df = plot_df.melt(id_vars='Lap', value_vars=drivers, var_name='Driver', value_name='Time')

    c1, c2 = st.columns(2)
    with c1:
        fig_pace = px.line(long_df, x='Lap', y='Time', color='Driver', 
                           title=f"{session_name}: Pace Evolution (Excl. Lap 0)", markers=True)
        fig_pace.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_pace, use_container_width=True)
        
    with c2:
        sorted_drivers = long_df.groupby('Driver')['Time'].median().sort_values().index
        fig_box = px.box(long_df, x='Driver', y='Time', color='Driver',
                         category_orders={'Driver': sorted_drivers},
                         title=f"{session_name}: Consistency (Excl. Lap 0)")
        st.plotly_chart(fig_box, use_container_width=True)

def render_global_comparison(quali_data, race_data):
    st.header("🌍 Global Analysis: Quali vs Race")
    
    q_res = quali_data[0]['results'] if quali_data else pd.DataFrame()
    r_res = race_data[0]['results'] if race_data else pd.DataFrame()
    
    if q_res.empty or r_res.empty:
        st.info("Upload BOTH Qualifying and Race PDFs to see the global comparison.")
        return

    q_df = q_res[['Fahrer', 'Bestzeit']].copy()
    q_df['Bestzeit'] = q_df['Bestzeit'].apply(parse_time_str)
    q_df = q_df.rename(columns={'Bestzeit': 'Quali_Time'})
    
    r_df = r_res[['Fahrer', 'Bestzeit']].copy()
    r_df['Bestzeit'] = r_df['Bestzeit'].apply(parse_time_str)
    r_df = r_df.rename(columns={'Bestzeit': 'Race_Time'})
    
    merged = pd.merge(q_df, r_df, on='Fahrer', how='inner')
    
    if merged.empty:
        st.warning("Could not match drivers. Check name spelling.")
        return
        
    merged['Time_Delta'] = merged['Race_Time'] - merged['Quali_Time']
    
    col1, col2 = st.columns(2)
    with col1:
        fig_scatter = px.scatter(merged, x='Quali_Time', y='Race_Time', text='Fahrer',
                                 title="Qualifying Pace vs Race Pace")
        min_val = min(merged['Quali_Time'].min(), merged['Race_Time'].min()) * 0.99
        max_val = max(merged['Quali_Time'].max(), merged['Race_Time'].max()) * 1.01
        
        fig_scatter.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val,
                              line=dict(color="Gray", dash="dash"))
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col2:
        fig_delta = px.bar(merged.sort_values('Time_Delta'), x='Fahrer', y='Time_Delta', 
                           color='Time_Delta', title="Time Delta (Race - Quali)",
                           color_continuous_scale=px.colors.diverging.RdYlGn_r)
        st.plotly_chart(fig_delta, use_container_width=True)

# --- MAIN APP UI ---
st.title("🏎️ OpenApex: RC Timing Parser")

with st.sidebar:
    st.header("📂 Data Upload")
    quali_file = st.file_uploader("1. Upload Qualifying PDF", type="pdf")
    race_file = st.file_uploader("2. Upload Race PDF", type="pdf")
    process_btn = st.button("🚀 Analyze Weekend")

if process_btn:
    if not quali_file and not race_file:
        st.error("Please upload at least one PDF.")
    else:
        with st.spinner("Parsing documents..."):
            quali_pages = parse_pdf_bytes(quali_file.getvalue()) if quali_file else []
            race_pages = parse_pdf_bytes(race_file.getvalue()) if race_file else []
            
            tab_names = []
            if quali_file: tab_names.append("⏱️ Qualifying")
            if race_file: tab_names.append("🏁 Race")
            tab_names.append("🌍 Global Analysis")
            
            tabs = st.tabs(tab_names)
            current_tab = 0
            
            if quali_file:
                with tabs[current_tab]:
                    for i, page in enumerate(quali_pages):
                        st.subheader(f"Qualifying Page {i+1}")
                        c1, c2 = st.columns(2)
                        c1.dataframe(page['results'], use_container_width=True)
                        c2.dataframe(page['laps'], use_container_width=True)
                        if not page['laps'].empty:
                            render_session_analytics(page['laps'], page['results'], "Qualifying")
                current_tab += 1
            
            if race_file:
                with tabs[current_tab]:
                    for i, page in enumerate(race_pages):
                        st.subheader(f"Race Page {i+1}")
                        c1, c2 = st.columns(2)
                        c1.dataframe(page['results'], use_container_width=True)
                        c2.dataframe(page['laps'], use_container_width=True)
                        if not page['laps'].empty:
                            render_session_analytics(page['laps'], page['results'], "Race")
                current_tab += 1
                
            with tabs[current_tab]:
                render_global_comparison(quali_pages, race_pages)

elif not process_btn:
    st.info("👈 Upload your PDFs in the sidebar and click 'Analyze Weekend' to start.")