import streamlit as st
import pandas as pd
import json
import os
import requests
import re
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from io import StringIO
import google.generativeai as genai
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
import time
from streamlit_lottie import st_lottie
import plotly.figure_factory as ff
import PyPDF2
import io

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="SecureScan Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
        :root {
        --primary-color: #4F46E5;
        --secondary-color: #6366F1;
        --accent-color: #8B5CF6;
        --background-light: #F8FAFC;
        --background-dark: #1E293B;
        --text-dark: #1E293B;
        --text-light: #F1F5F9;
        --shadow-light: rgba(0, 0, 0, 0.1);
    }

    body {
        background-color: var(--background-light);
        color: var(--text-dark);
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        font-size: 2.8rem;
        color: var(--primary-color);
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 800;
    }

    .sub-header {
        font-size: 2rem;
        color: var(--secondary-color);
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }

    .section-header {
        font-size: 1.6rem;
        color: var(--accent-color);
        margin-top: 0.8rem;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }

    .card {
        background: linear-gradient(135deg, #EFF6FF, #DBEAFE);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border-left: 5px solid var(--primary-color);
        box-shadow: 0 4px 8px var(--shadow-light);
        transition: transform 0.2s ease-in-out;
    }
    .card:hover {
        transform: translateY(-3px);
    }

    .critical-card { border-left-color: #E11D48; }
    .high-card { border-left-color: #F97316; }
    .medium-card { border-left-color: #EAB308; }
    .low-card { border-left-color: #16A34A; }

    .metric-card {
        background: white;
        border-radius: 10px;
        box-shadow: 0 2px 6px var(--shadow-light);
        padding: 1.2rem;
        text-align: center;
        height: 100%;
    }

    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }

    .metric-label {
        font-size: 1rem;
        color: #64748B;
        margin-bottom: 0.5rem;
    }

    .status-badge {
        display: inline-block;
        padding: 0.4rem 0.7rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        box-shadow: 0 2px 4px var(--shadow-light);
    }
    .badge-critical { background-color: #FECACA; color: #B91C1C; }
    .badge-high { background-color: #FED7AA; color: #C2410C; }
    .badge-medium { background-color: #FEF08A; color: #B45309; }
    .badge-low { background-color: #D9F99D; color: #15803D; }

    .stApp > header {
        background-color: transparent !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
        padding: 12px 18px;
        background: linear-gradient(135deg, #E0E7FF, #C7D2FE);
        color: var(--text-dark);
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        background: var(--primary-color) !important;
        color: var(--text-light) !important;
        font-weight: 700;
    }

    .tooltip {
        position: relative;
        display: inline-block;
        cursor: pointer;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 220px;
        background: var(--background-dark);
        color: var(--text-light);
        text-align: center;
        border-radius: 8px;
        padding: 6px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -110px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }

    div[data-testid="stExpander"] {
        border: none;
        box-shadow: 0 2px 6px var(--shadow-light);
        border-radius: 10px;
        margin-bottom: 1rem;
    }

    div[data-testid="stExpanderToggleButton"] {
        border-radius: 10px 10px 0 0;
    }

    /* Right-side dashboard layout */
    .dashboard-container {
        display: flex;
        flex-direction: row;
    }
    .upload-section {
        flex: 3;
        padding-right: 20px;
    }
    .dashboard-section {
        flex: 7;
        border-left: 1px solid #CBD5E1;
        padding-left: 20px;
    }

    /* Media query for mobile responsiveness */
    @media (max-width: 768px) {
        .dashboard-container {
            flex-direction: column;
        }
        .upload-section, .dashboard-section {
            flex: 1;
            padding: 0;
            border: none;
        }
    }

</style>
""", unsafe_allow_html=True)

# Function to load Lottie animations
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return None

# Configure Gemini API
def configure_gemini():
    # Try to get API key from environment variable first
    api_key = os.getenv("GEMINI_API_KEY", "")
    
    # If not in env, check session state
    if not api_key:
        api_key = st.session_state.get("api_key", "")
    
    # If we have an API key, configure the Gemini API
    if api_key:
        try:
            genai.configure(api_key=api_key)
            return True
        except Exception as e:
            st.sidebar.error(f"Error configuring Gemini API: {str(e)}")
            return False
    return False

# Function to parse system information file
def parse_system_info(file_content):
    data = {}
    current_section = None
    
    for line in file_content.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # Check for section headers
        if line.endswith(':') and not line.startswith(' ') and not line.startswith('\t'):
            current_section = line[:-1]
            data[current_section] = []
        elif current_section and line:
            data[current_section].append(line)
    
    return data

# Function to analyze vulnerabilities using Gemini
def analyze_vulnerabilities(system_data):
    if not configure_gemini():
        st.error("Please configure the Gemini API key first")
        return None
    
    try:
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Prepare the prompt
        prompt = f"""
        Analyze the following system information for security vulnerabilities and potential risks:
        
        {json.dumps(system_data, indent=2)}
        
        Provide a JSON response with the following structure:
        {{
          "summary": "Brief overview of the system security posture",
          "risk_score": "A number between 1-10 representing overall risk level",
          "vulnerabilities": [
            {{
              "id": "unique identifier",
              "title": "Vulnerability title",
              "description": "Detailed description",
              "severity": "Critical/High/Medium/Low",
              "affected_component": "Component with vulnerability",
              "recommendation": "How to mitigate",
              "cvss_score": "A number between 0-10 representing the CVSS score if applicable"
            }}
          ],
          "component_analysis": {{
            "os": "OS security analysis",
            "network": "Network security analysis",
            "applications": "Applications security analysis",
            "users": "User account security analysis"
          }},
          "risk_by_category": {{
            "os_risk": "A number between 1-10",
            "network_risk": "A number between 1-10",
            "application_risk": "A number between 1-10",
            "user_risk": "A number between 1-10"
          }},
          "recommendation_summary": "Overall security recommendations"
        }}
        """
        
        response = model.generate_content(prompt)
        
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            return json.loads(json_str)
        else:
            st.error("Failed to extract valid JSON from the Gemini response")
            return None
    except Exception as e:
        st.error(f"Error analyzing vulnerabilities: {str(e)}")
        return None

# Function to render vulnerability dashboard
def render_dashboard(analysis_result):
    # Header with animation
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h1 class='main-header'>Security Vulnerability Dashboard</h1>", unsafe_allow_html=True)
        security_animation = load_lottie_url("https://assets7.lottiefiles.com/packages/lf20_xv1gn9.json")
        if security_animation:
            st_lottie(security_animation, speed=1, height=150, key="security_anim")
    
    # Top metrics row
    st.markdown("<h2 class='sub-header'>Security Overview</h2>", unsafe_allow_html=True)
    
    # Risk score and vulnerability counts
    risk_score = float(analysis_result["risk_score"])
    vulns = analysis_result["vulnerabilities"]
    vuln_df = pd.DataFrame(vulns)
    
    # Count vulnerabilities by severity
    severity_counts = {
        "Critical": len(vuln_df[vuln_df["severity"] == "Critical"]) if "severity" in vuln_df.columns else 0,
        "High": len(vuln_df[vuln_df["severity"] == "High"]) if "severity" in vuln_df.columns else 0,
        "Medium": len(vuln_df[vuln_df["severity"] == "Medium"]) if "severity" in vuln_df.columns else 0,
        "Low": len(vuln_df[vuln_df["severity"] == "Low"]) if "severity" in vuln_df.columns else 0
    }
    
    # Risk level determination
    risk_level = "Critical" if risk_score >= 7.5 else "High" if risk_score >= 5 else "Medium" if risk_score >= 3 else "Low"
    
    # Top metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Overall Risk</div>
            <div class="metric-value metric-{risk_level.lower()}">{risk_score}/10</div>
            <div class="status-badge badge-{risk_level.lower()}">{risk_level}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Critical Vulnerabilities</div>
            <div class="metric-value metric-critical">{severity_counts["Critical"]}</div>
            <div class="status-badge badge-critical">Critical</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">High Vulnerabilities</div>
            <div class="metric-value metric-high">{severity_counts["High"]}</div>
            <div class="status-badge badge-high">High</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Medium Vulnerabilities</div>
            <div class="metric-value metric-medium">{severity_counts["Medium"]}</div>
            <div class="status-badge badge-medium">Medium</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Low Vulnerabilities</div>
            <div class="metric-value metric-low">{severity_counts["Low"]}</div>
            <div class="status-badge badge-low">Low</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Risk summary and visualization
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3 class='section-header'>Security Summary</h3>", unsafe_allow_html=True)
        st.write(analysis_result["summary"])
        
        if "recommendation_summary" in analysis_result:
            st.markdown("<h3 class='section-header'>Recommendation Summary</h3>", unsafe_allow_html=True)
            st.write(analysis_result["recommendation_summary"])
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        # Create a radial gauge for risk score
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Risk Score", 'font': {'size': 24}},
            gauge = {
                'axis': {'range': [0, 10], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "#1E3A8A"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 3], 'color': '#4ADE80'},
                    {'range': [3, 5], 'color': '#FBBF24'},
                    {'range': [5, 7.5], 'color': '#F97316'},
                    {'range': [7.5, 10], 'color': '#EF4444'}
                ],
            }
        ))
        
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            font={'color': "#1E3A8A", 'family': "Arial"}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Risk by category
    if "risk_by_category" in analysis_result:
        st.markdown("<h2 class='sub-header'>Risk by Category</h2>", unsafe_allow_html=True)
        
        risk_categories = analysis_result["risk_by_category"]
        categories = ["OS Risk", "Network Risk", "Application Risk", "User Risk"]
        risk_values = [
            float(risk_categories.get("os_risk", 0)),
            float(risk_categories.get("network_risk", 0)),
            float(risk_categories.get("application_risk", 0)),
            float(risk_categories.get("user_risk", 0))
        ]
        
        # Color mapping
        colors = ['#4ADE80', '#FBBF24', '#F97316', '#EF4444']
        category_colors = [
            colors[0] if val < 3 else colors[1] if val < 5 else colors[2] if val < 7.5 else colors[3]
            for val in risk_values
        ]
        
        # Create horizontal bar chart
        fig = go.Figure()
        
        for i, (cat, val, color) in enumerate(zip(categories, risk_values, category_colors)):
            fig.add_trace(go.Bar(
                x=[val],
                y=[cat],
                orientation='h',
                marker=dict(color=color),
                text=[f"{val}/10"],
                textposition='auto',
                hoverinfo='text',
                hovertext=[f"{cat}: {val}/10"],
                name=cat
            ))
        
        fig.update_layout(
            title="Risk Levels by Category",
            xaxis=dict(title="Risk Score", range=[0, 10]),
            yaxis=dict(title="Category"),
            height=300,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            font={'color': "#1E3A8A", 'family': "Arial"}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Vulnerabilities section
    st.markdown("<h2 class='sub-header'>Vulnerabilities</h2>", unsafe_allow_html=True)
    
    # Display vulnerabilities by severity
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Create a pie chart for vulnerabilities by severity
        labels = list(severity_counts.keys())
        values = list(severity_counts.values())
        
        fig = px.pie(
            names=labels, 
            values=values,
            color=labels,
            color_discrete_map={
                'Critical': '#EF4444',
                'High': '#F97316',
                'Medium': '#FBBF24',
                'Low': '#4ADE80'
            },
            title="Vulnerabilities by Severity"
        )
        
        fig.update_traces(textinfo='percent+label', hoverinfo='label+value')
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            font={'color': "#1E3A8A", 'family': "Arial"}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Create severity tabs for the vulnerabilities
        severity_tabs = st.tabs(["Critical", "High", "Medium", "Low"])
        
        for i, severity in enumerate(["Critical", "High", "Medium", "Low"]):
            with severity_tabs[i]:
                filtered_vulns = vuln_df[vuln_df["severity"] == severity] if "severity" in vuln_df.columns else pd.DataFrame()
                
                if filtered_vulns.empty:
                    st.info(f"No {severity} vulnerabilities found.")
                else:
                    for _, vuln in filtered_vulns.iterrows():
                        cvss_score = vuln.get("cvss_score", "N/A")
                        
                        st.markdown(f"""
                        <div class='card {severity.lower()}-card'>
                            <h3>{vuln['title']}</h3>
                            <p><strong>Description:</strong> {vuln['description']}</p>
                            <p><strong>Affected Component:</strong> {vuln['affected_component']}</p>
                            <p><strong>CVSS Score:</strong> {cvss_score}</p>
                            <p><strong>Recommendation:</strong> {vuln['recommendation']}</p>
                        </div>
                        """, unsafe_allow_html=True)
    
    # Component analysis
    st.markdown("<h2 class='sub-header'>Component Analysis</h2>", unsafe_allow_html=True)
    
    component_analysis = analysis_result["component_analysis"]
    
    tabs = st.tabs(["OS Security", "Network Security", "Application Security", "User Account Security"])
    
    tab_contents = [
        component_analysis["os"],
        component_analysis["network"],
        component_analysis["applications"],
        component_analysis["users"]
    ]
    
    tab_icons = ["💻", "🌐", "📱", "👤"]
    
    for i, (tab, content, icon) in enumerate(zip(tabs, tab_contents, tab_icons)):
        with tab:
            st.markdown(f"<h3 class='section-header'>{icon} {tab.label} Analysis</h3>", unsafe_allow_html=True)
            st.markdown(f"<div class='card'>{content}</div>", unsafe_allow_html=True)

    # Export report button
    st.markdown("<br>", unsafe_allow_html=True)
    if st.download_button(
        label="📊 Export Security Report (PDF)",
        data="This is a placeholder for the PDF export functionality.",
        file_name="security_report.pdf",
        mime="application/pdf",
    ):
        st.success("Report downloaded successfully!")

# Function to create a sample system information file
def create_sample_system_info():
    sample_data = """OS Details:
Computer Name: DESKTOP-ABC123
Windows Version: Windows 10 Pro
Architecture: x64
Build Number: 19044.1826

.NET Framework Versions:
4.8.03761

AMSI Providers:
Windows Defender

Antivirus Information:
Windows Defender - Enabled
Real-time Protection: On
Virus definitions: Up to date (2023-06-01)

Firewall Rules:
Allow TCP traffic on port 3389 (RDP) - Enabled
Block all incoming traffic - Disabled
Allow outbound HTTP/HTTPS - Enabled

Auto-run Programs:
C:\\Program Files\\Startup App\\startup.exe
C:\\Users\\Admin\\AppData\\Roaming\\Microsoft\\Windows\\Start Menu\\Programs\\Startup\\autorun.bat

Local Users:
Administrator - Enabled
Guest - Disabled
User1 - Enabled

Installed Hotfixes:
KB5003637
KB5005565
KB5006670

Installed Applications:
Microsoft Office 365 - 16.0.14326.20404
Google Chrome - 103.0.5060.114
NotePad++ - 8.4.1

Network Information:
ARP Table:
192.168.1.1 - 00-11-22-33-44-55 - eth0
192.168.1.100 - 66-77-88-99-AA-BB - eth0

DNS Cache:
google.com - 142.250.180.78
microsoft.com - 23.45.67.89

Network Profiles:
Home Network - Private - Connected
Work Network - Domain - Disconnected

Network Shares:
C:\\Users\\Admin\\Documents - ReadWrite
D:\\Shared - ReadOnly

TCP & UDP Connections:
TCP - 192.168.1.5:53425 -> 35.186.224.25:443 - ESTABLISHED
UDP - 192.168.1.5:5353 -> 224.0.0.251:5353

RPC Endpoints:
UUID: 12345678-1234-1234-1234-123456789abc - Active

Open Ports:
TCP - 80 (HTTP)
TCP - 443 (HTTPS)
TCP - 3389 (RDP)
UDP - 53 (DNS)

Network Adapters:
Intel(R) Wireless-AC 9560 - 00-11-22-33-44-66 - Connected
Realtek PCIe GbE Family Controller - 00-11-22-33-44-77 - Disconnected

LLDP/CDP Information:
Switch1 - Port 24 - 192.168.1.254

VLAN Information:
VLAN 10 - Default
VLAN 20 - Management
"""
    return sample_data

# Sidebar
with st.sidebar:
    security_icon = load_lottie_url("https://assets9.lottiefiles.com/packages/lf20_bkmfz3iv.json")
    if security_icon:
        st_lottie(security_icon, speed=1, height=120, key="security_icon")
    
    st.title("SecureScan Dashboard")
    st.markdown("---")
    
    # Check for API key in environment
    api_key_env = os.getenv("GEMINI_API_KEY", "")
    if api_key_env:
        st.success("✅ Gemini API key loaded from environment")
        st.session_state.api_key = api_key_env
    else:
        # API key input for Gemini
        if "api_key" not in st.session_state:
            st.session_state.api_key = ""
        
        api_key = st.text_input("Enter Gemini API Key:", value=st.session_state.api_key, type="password")
        if api_key:
            st.session_state.api_key = api_key
            genai_configured = configure_gemini()
            if genai_configured:
                st.success("✅ Gemini API configured successfully")
                # Create .env file with the API key
                with open(".env", "w") as f:
                    f.write(f"GEMINI_API_KEY={api_key}")
                st.info("API key saved to .env file")
            else:
                st.error("❌ Failed to configure Gemini API")
    
    st.markdown("---")
    st.markdown("### 📊 Dashboard Features")
    st.markdown("""
    - 🔍 System information analysis
    - 🔒 Security vulnerability detection
    - 📈 Risk assessment
    - 🛡️ Remediation recommendations
    """)
    
    st.markdown("---")
    st.markdown("### ℹ️ How to Use")
    st.markdown("""
    1. Upload your system information file
    2. Click "Analyze Vulnerabilities"
    3. Review the security findings
    4. Export the report if needed
    """)

# Main application
def main():
    # Create a two-column layout with upload on left, dashboard on right
    upload_col, dashboard_col = st.columns([3, 7])
    
    with upload_col:
        st.markdown("<h2 class='sub-header'>Upload Data</h2>", unsafe_allow_html=True)
        
        # File uploader that accepts txt, json, csv, and now pdf
        uploaded_file = st.file_uploader("Upload system information file", type=["txt", "json", "csv", "pdf"])
        
        if uploaded_file is not None:
            st.success("✅ File uploaded successfully!")
            
            # Read file content based on file type
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'pdf':
                # Process PDF file
                file_content = extract_text_from_pdf(uploaded_file)
                if file_content is None:
                    st.error("Failed to extract text from PDF. Please check the file format.")
                    return
            else:
                # Process text-based files
                file_content = uploaded_file.getvalue().decode("utf-8")
            
            # Parse system information
            with st.spinner("Parsing system information..."):
                system_data = parse_system_info(file_content)
                
            # Show parsed data in expandable section
            with st.expander("View parsed system information"):
                st.json(system_data)
            
            if st.button("🔍 Analyze Vulnerabilities", key="analyze_button"):
                # Show progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Simulated analysis steps
                status_text.text("Step 1/5: Initializing analysis...")
                progress_bar.progress(10)
                time.sleep(0.5)
                
                status_text.text("Step 2/5: Processing system information...")
                progress_bar.progress(30)
                time.sleep(0.5)
                
                status_text.text("Step 3/5: Identifying vulnerabilities...")
                progress_bar.progress(50)
                time.sleep(0.5)
                
                status_text.text("Step 4/5: Analyzing security risks...")
                progress_bar.progress(70)
                time.sleep(0.5)
                
                status_text.text("Step 5/5: Generating recommendations...")
                progress_bar.progress(90)
                
                # Actual analysis
                analysis_result = analyze_vulnerabilities(system_data)
                
                progress_bar.progress(100)
                status_text.text("Analysis complete!")
                time.sleep(0.5)
                
                # Clear progress elements
                progress_bar.empty()
                status_text.empty()
                
                if analysis_result:
                    # Save analysis result to session state
                    st.session_state.analysis_result = analysis_result
                    st.session_state.system_data = system_data
                    
                    # Show success message
                    st.success("✅ Analysis completed successfully!")
                else:
                    st.error("❌ Failed to analyze vulnerabilities")
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🔄 Or use sample data")
        
        # Option to use sample data
        if st.button("Load Data", key="sample_button"):
            file_content = create_sample_system_info()
            st.success("✅ Sample data loaded successfully!")
            
            # Parse system information
            with st.spinner("Parsing sample system information..."):
                system_data = parse_system_info(file_content)
                
            # Show parsed data in expandable section
            with st.expander("View parsed sample system information"):
                st.json(system_data)
            
            # Analyze the sample data
            with st.spinner("Analyzing sample data..."):
                # Show progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Simulated analysis steps
                status_text.text("Step 1/5: Initializing analysis...")
                progress_bar.progress(10)
                time.sleep(0.5)
                
                status_text.text("Step 2/5: Processing system information...")
                progress_bar.progress(30)
                time.sleep(0.5)
                
                status_text.text("Step 3/5: Identifying vulnerabilities...")
                progress_bar.progress(50)
                time.sleep(0.5)
                
                status_text.text("Step 4/5: Analyzing security risks...")
                progress_bar.progress(70)
                time.sleep(0.5)
                
                status_text.text("Step 5/5: Generating recommendations...")
                progress_bar.progress(90)
                
                # Actual analysis
                analysis_result = analyze_vulnerabilities(system_data)
                
                progress_bar.progress(100)
                status_text.text("Analysis complete!")
                time.sleep(0.5)
                
                # Clear progress elements
                progress_bar.empty()
                status_text.empty()
                
                if analysis_result:
                    # Save analysis result to session state
                    st.session_state.analysis_result = analysis_result
                    st.session_state.system_data = system_data
                    
                    # Show success message
                    st.success("✅ Analysis completed successfully!")
                else:
                    st.error("❌ Failed to analyze vulnerabilities")
    
    with dashboard_col:
        # Display dashboard if analysis result exists
        if 'analysis_result' in st.session_state:
            render_dashboard(st.session_state.analysis_result)
        else:
            # Welcome message and instructions
            welcome_animation = load_lottie_url("https://assets5.lottiefiles.com/packages/lf20_qmfs6c3i.json")
            
            st.markdown("<h2 class='sub-header'>Welcome to SecureScan Dashboard</h2>", unsafe_allow_html=True)
            
            if welcome_animation:
                st_lottie(welcome_animation, speed=1, height=200, key="welcome_anim")
            
            st.markdown("""
            <div class="card">
                <h3 class="section-header">Getting Started</h3>
                <p>This dashboard helps you analyze your system's security posture and identify potential vulnerabilities.</p>
                <p>To get started:</p>
                <ol>
                    <li>Upload a system information file using the panel on the left</li>
                    <li>Click "Analyze Vulnerabilities" to perform a security assessment</li>
                    <li>Review the findings and recommendations</li>
                    <li>Export the report for your records</li>
                </ol>
                <p>You can also use the "Load Sample Data" button to see a demonstration of the dashboard.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Features section
            st.markdown("<h3 class='section-header'>Key Features</h3>", unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-label">🔍 Vulnerability Detection</div>
                    <p>Identify security weaknesses in your system configuration</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-label">📊 Risk Assessment</div>
                    <p>Visualize security risks with intuitive graphs and metrics</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-label">🛡️ Remediation Guidance</div>
                    <p>Get actionable recommendations to improve security</p>
                </div>
                """, unsafe_allow_html=True)

# Run the main application
if __name__ == "__main__":
    main()
