"""
Coherence UI - Main Streamlit Application

This module provides a web-based interface for interacting with the Coherence API,
allowing users to create and manage semantic axes, analyze text, search, and perform what-if analysis.
"""
import streamlit as st
from typing import Optional, Dict, Any
import requests
from pathlib import Path
import sys

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Default API URL
DEFAULT_API_URL = "http://localhost:8080"

# Import UI components and utilities
from coherence.ui import components
from coherence.ui.components import get_api_url, AxesForm, AnalyzeForm, SearchForm, WhatIfForm

def check_backend_health() -> Optional[Dict[str, Any]]:
    """Check if the backend is healthy and return its status."""
    try:
        response = requests.get(f"{get_api_url()}/api/v1/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except requests.exceptions.RequestException:
        return None

# Set page config
st.set_page_config(
    page_title="Coherence UI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


def main():
    """Main application entry point."""
    # Initialize session state variables
    if "api_url" not in st.session_state:
        st.session_state.api_url = DEFAULT_API_URL
    
    # Sidebar navigation and settings
    with st.sidebar:
        st.title("🔍 Coherence UI")
        
        # API Settings
        with st.expander("⚙️ Settings"):
            st.session_state.api_url = st.text_input(
                "API Base URL",
                value=st.session_state.api_url,
                help="URL where the Coherence API is running"
            )
        
        # Navigation
        st.markdown("## Navigation")
        page = st.radio(
            "Go to",
            ["Home", "Axes", "Analyze", "Search", "What-If", "Ethical Evaluation", "Interaction"],
            label_visibility="collapsed"
        )
        
        # Display system status
        st.markdown("---")
        st.markdown("## System Status")
        health = check_backend_health()
        if health:
            st.success("✅ Backend is running")
            st.caption(f"Version: {health.get('version', 'unknown')}")
            st.caption(f"Encoder: {health.get('encoder_model', 'unknown')}")
            st.caption(f"Active Pack: {health.get('active_pack', {}).get('pack_id', 'None')}")
        else:
            st.error("❌ Backend is not reachable")
            st.caption(f"Tried to connect to: {get_api_url()}")
    
    # Page routing
    if page == "Home":
        show_home_page()
    elif page == "Axes":
        components.AxesForm()
    elif page == "Analyze":
        components.AnalyzeForm()
    elif page == "Search":
        components.SearchForm()
    elif page == "What-If":
        components.WhatIfForm()
    elif page == "Ethical Evaluation":
        show_ethical_evaluation()
    elif page == "Interaction":
        show_interaction()


def show_home_page():
    """Render the home page with usage instructions."""
    st.title("Welcome to Coherence UI")
    
    st.markdown("""
    ## 🚀 Getting Started
    Coherence is a semantic analysis engine that helps you explore and understand text through 
    the lens of custom semantic axes. Follow these steps to get started:
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 1. Create Axes")
        st.markdown("""
        Define what to analyze by creating semantic axes (e.g., "Agency" vs "Control") 
        in the **Axes** tab.
        """)
    
    with col2:
        st.markdown("### 2. Analyze Text")
        st.markdown("""
        Paste text to analyze in the **Analyze** tab to see how it resonates with 
        your defined axes.
        """)
    
    with col3:
        st.markdown("### 3. Explore & Experiment")
        st.markdown("""
        Use **Search** to find relevant content and **What-If** to test how changes 
        affect the analysis.
        """)
    
    st.markdown("---")
    
    st.markdown("## 📚 Documentation")
    st.markdown("""
    - [API Documentation](/docs/API.md)
    - [Models Reference](/docs/Models.md)
    - [GitHub Repository](https://github.com/yourusername/coherence)
    """)
    
    st.markdown("---")
    
    st.markdown("## 🛠 System Information")
    health = check_backend_health()
    if health:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Backend Status", "Running")
            st.metric("Active Pack", health.get("active_pack", {}).get("pack_id", "None"))
        with col2:
            st.metric("Encoder", health.get("encoder_model", "Unknown"))
            st.metric("Version", health.get("version", "Unknown"))
    else:
        st.error("Backend is not reachable. Please ensure the Coherence API is running.")
        st.code("uvicorn coherence.api.main:app --reload --host 0.0.0.0 --port 8080")


def show_ethical_evaluation():
    st.title("Ethical Evaluation")
    text = st.text_area("Enter text to evaluate ethically")
    if st.button("Evaluate"):
        try:
            response = requests.post(f"{get_api_url()}/v1/eval/text", json={"text": text})
            if response.status_code == 200:
                data = response.json()
                st.subheader("Decision Proof")
                st.json(data["proof"])
                st.subheader("Veto Spans")
                st.json(data["spans"])
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Failed to connect to API: {e}")

def show_interaction():
    st.title("Interaction")
    prompt = st.text_input("Enter your prompt")
    if st.button("Respond"):
        try:
            response = requests.post(f"{get_api_url()}/v1/interaction/respond", json={"prompt": prompt})
            if response.status_code == 200:
                data = response.json()
                st.subheader("Final Response")
                st.write(data["final"])
                st.subheader("Decision Proof")
                st.json(data["proof"])
                st.subheader("Alternatives")
                for alt in data["alternatives"]:
                    st.write(alt["text"])
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Failed to connect to API: {e}")

if __name__ == "__main__":
    main()
