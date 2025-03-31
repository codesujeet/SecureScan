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
import requests

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
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #1E3A8A;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    .section-header {
        font-size: 1.4rem;
        color: #2563EB;
        margin-top: 0.8rem;
        margin-bottom: 0.3rem;
        font-weight: 600;
    }
    .card {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #3B82F6;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .critical-card {
        border-left: 4px solid #DC2626;
    }
    .high-card {
        border-left: 4px solid #EA580C;
    }
    .medium-card {
        border-left: 4px solid #D97706;
    }
    .low-card {
        border-left: 4px solid #65A30D;
    }
    .metric-card {
        background-color: white;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        padding: 1rem;
        text-align: center;
        height: 100%;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6B7280;
        margin-bottom: 0.5rem;
    }
    .metric-critical {
        color: #DC2626;
    }
    .metric-high {
        color: #EA580C;
    }
    .metric-medium {
        color: #D97706;
    }
    .metric-low {
        color: #65A30D;
    }
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 500;
        text-transform: uppercase;
        margin-right: 0.5rem;
    }
    .badge-critical {
        background-color: #FEE2E2;
        color: #DC2626;
    }
    .badge-high {
        background-color: #FFEDD5;
        color: #EA580C;
    }
    .badge-medium {
        background-color: #FEF3C7;
        color: #D97706;
    }
    .badge-low {
        background-color: #ECFCCB;
        color: #65A30D;
    }
    .stApp > header {
        background-color: transparent !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 4px 4px 0 0;
        padding: 10px 16px;
        background-color: #EFF6FF;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3B82F6 !important;
        color: white !important;
    }
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: pointer;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    div[data-testid="stExpander"] {
        border: none;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    div[data-testid="stExpanderToggleButton"] {
        border-radius: 8px 8px 0 0;
    }
</style>
""", unsafe_allow_html=True)

# Function to load Lottie animations
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

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
    # Main content
    tab1, tab2 = st.tabs(["📊 Dashboard", "📤 Upload Data"])
    
    with tab2:
        st.markdown("<h2 class='sub-header'>Upload System Information</h2>", unsafe_allow_html=True)
        
        upload_col1, upload_col2 = st.columns([1, 1])
        
        with upload_col1:
            # File uploader
            uploaded_file = st.file_uploader("Upload system information file", type=["txt", "json", "csv"])
            
            if uploaded_file is not None:
                st.success("✅ File uploaded successfully!")
                
                # Read file content
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
                        
                        # Switch to dashboard tab
                        # This is a workaround since direct tab switching is not fully supported
                        st.markdown("Dashboard ready! Please click on the 'Dashboard' tab to view results.")
                    else:
                        st.error("❌ Failed to analyze vulnerabilities")
        
        with upload_col2:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 🔄 Or use sample data")
            
            # Option to use sample data
            if st.button("Load Sample Data", key="sample_button"):
                file_content = create_sample_system_info()
                st.success("✅ Sample data loaded successfully!")
                
                # Parse system information
                with st.spinner("Parsing sample system information..."):
                    system_data = parse_system_info(file_content)
                    
                # Show parsed data in expandable section
                with st.expander("View parsed system information"):
                    st.json(system_data)
                
                # Analyze vulnerabilities
                with st.spinner("Analyzing vulnerabilities..."):
                    # Show progress
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Simulated analysis steps
                    status_text.text("Step 1/5: Initializing analysis...")
                    progress_bar.progress(10)
                    time.sleep(0.3)
                    
                    status_text.text("Step 2/5: Processing system information...")
                    progress_bar.progress(30)
                    time.sleep(0.3)
                    
                    status_text.text("Step 3/5: Identifying vulnerabilities...")
                    progress_bar.progress(50)
                    time.sleep(0.3)
                    
                    status_text.text("Step 4/5: Analyzing security risks...")
                    progress_bar.progress(70)
                    time.sleep(0.3)
                    
                    status_text.text("Step 5/5: Generating recommendations...")
                    progress_bar.progress(90)
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
                        
                        # Switch to dashboard tab
                        # This is a workaround since direct tab switching is not fully supported
                        st.markdown("Dashboard ready! Please click on the 'Dashboard' tab to view results.")
                    else:
                        st.error("❌ Failed to analyze vulnerabilities")
    
    with tab1:
        # Check if analysis result exists in session state
        if 'analysis_result' in st.session_state:
            # Render dashboard with analysis result
            render_dashboard(st.session_state.analysis_result)
        else:
            # Show welcome message
            welcome_animation = load_lottie_url("https://assets5.lottiefiles.com/packages/lf20_ysas4vcp.json")
            if welcome_animation:
                st_lottie(welcome_animation, speed=1, height=300, key="welcome_anim")
            
            st.markdown("<h1 class='main-header'>Welcome to SecureScan Dashboard</h1>", unsafe_allow_html=True)
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("""
            ### 🛡️ Comprehensive Security Analysis
            
            SecureScan Dashboard provides detailed analysis of your system's security posture:
            
            - Identify security vulnerabilities
            - Assess risk levels
            - Get tailored recommendations
            - Generate comprehensive reports
            
            To get started, click on the "Upload Data" tab and upload your system information file or use the sample data.
            """)
            st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()