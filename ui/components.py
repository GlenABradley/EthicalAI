"""
UI Components for Coherence

This module contains reusable UI components for the Coherence application.
"""
from typing import List, Dict, Any, Optional, Tuple
import streamlit as st
import requests
import json
from typing import Any, Dict, List, Optional, Tuple

# Default API URL
DEFAULT_API_URL = "http://localhost:8080"

def get_api_url() -> str:
    """Get the API URL from session state or default."""
    return st.session_state.get("api_url", DEFAULT_API_URL)


def show_error(message: str, details: str = ""):
    """Display an error message with optional details."""
    st.error(message)
    if details:
        with st.expander("Error Details"):
            st.code(details)


def api_request(method: str, endpoint: str, **kwargs) -> Tuple[bool, Any]:
    """Make an API request and handle errors."""
    url = f"{get_api_url()}{endpoint}"
    try:
        response = requests.request(method, url, **kwargs)
        if response.status_code >= 400:
            return False, f"API Error ({response.status_code}): {response.text}"
        return True, response.json()
    except requests.exceptions.RequestException as e:
        return False, f"Request failed: {str(e)}"


class AxesForm:
    """Form for managing semantic axes."""
    
    def __init__(self):
        self.render()
    
    def render(self):
        """Render the axes management interface."""
        st.title("🔄 Manage Semantic Axes")
        
        st.markdown("""
        Semantic axes define the dimensions of meaning you want to analyze in your texts.
        Create axes by providing example words or phrases that represent opposite ends of a spectrum.
        """)
        
        # Create new axis form
        with st.expander("➕ Create New Axis", expanded=True):
            self._render_create_form()
        
        # List existing axes
        st.markdown("## Your Axes")
        self._render_axes_list()
    
    def _render_create_form(self):
        """Render the form for creating a new axis."""
        with st.form("create_axis"):
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("Axis Name", help="E.g., 'Agency vs Control'")
                method = st.selectbox(
                    "Method",
                    ["diffmean", "cca", "lda"],
                    help="Algorithm to derive the axis from examples"
                )
            
            with col2:
                positives = st.text_area(
                    "Positive Examples (one per line)",
                    "freedom\nchoice\nindependence",
                    help="Words/phrases that represent one end of the axis"
                )
                
                negatives = st.text_area(
                    "Negative Examples (one per line)",
                    "control\nconstraint\ndependence",
                    help="Words/phrases that represent the opposite end of the axis"
                )
            
            if st.form_submit_button("Create Axis"):
                self._handle_create_axis(name, positives, negatives, method)
    
    def _handle_create_axis(self, name: str, positives: str, negatives: str, method: str):
        """Handle axis creation form submission."""
        if not name:
            st.error("Please provide a name for the axis.")
            return
            
        positive_list = [p.strip() for p in positives.split("\n") if p.strip()]
        negative_list = [n.strip() for n in negatives.split("\n") if n.strip()]
        
        if not positive_list or not negative_list:
            st.error("Please provide at least one positive and one negative example.")
            return
        
        data = {
            "axes": [{
                "name": name,
                "positives": positive_list,
                "negatives": negative_list
            }],
            "method": method
        }
        
        success, response = api_request(
            "POST",
            "/v1/axes/build",
            json=data,
            headers={"Content-Type": "application/json"}
        )
        
        if success:
            st.success(f"Successfully created axis: {name}")
            st.rerun()
        else:
            show_error("Failed to create axis", response)
    
    def _render_axes_list(self):
        """Render the list of existing axes."""
        success, response = api_request("GET", "/axes/list")
        
        if not success:
            show_error("Failed to load axes", response)
            return
        
        if not response:
            st.info("No axes found. Create your first axis above!")
            return
        
        for axis in response:
            with st.expander(f"📊 {axis.get('pack_id', 'Unknown')}"):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**Method**: {axis.get('method', 'N/A')}")
                    st.markdown(f"**Dimensions**: {len(axis.get('axes', []))}")
                    st.markdown(f"**Created**: {axis.get('created_at', 'N/A')}")
                
                with col2:
                    if st.button("Activate", key=f"activate_{axis['pack_id']}"):
                        self._activate_pack(axis['pack_id'])
                
                if st.checkbox("Show details", key=f"details_{axis['pack_id']}"):
                    st.json(axis)
    
    def _activate_pack(self, pack_id: str):
        """Activate a specific axis pack."""
        success, response = api_request("POST", f"/v1/axes/{pack_id}/activate")
        if success:
            st.success(f"Activated pack: {pack_id}")
            st.rerun()
        else:
            show_error(f"Failed to activate pack {pack_id}", response)


class AnalyzeForm:
    """Form for text analysis."""
    
    def __init__(self):
        self.render()
    
    def render(self):
        """Render the text analysis interface."""
        st.title("🔍 Analyze Text")
        
        st.markdown("""
        Analyze text to see how it resonates with your semantic axes.
        The text will be processed into tokens, spans, and frames with associated vectors.
        """)
        
        # Get active pack info
        success, response = api_request("GET", "/health/ready")
        active_pack = response.get("active_pack", {}).get("pack_id") if success else None
        
        if not active_pack:
            st.warning("No active axis pack. Please create and activate an axis pack first.")
            return
        
        # Text input
        text = st.text_area(
            "Enter text to analyze",
            "User consent is required for data processing operations under GDPR.",
            height=150
        )
        
        # Analysis parameters
        with st.expander("⚙️ Analysis Settings"):
            col1, col2 = st.columns(2)
            with col1:
                max_span_len = st.slider("Maximum span length", 1, 20, 5)
            with col2:
                diffusion_tau = st.slider("Diffusion tau", 0.0, 1.0, 0.0, 0.1)
        
        if st.button("Analyze", type="primary"):
            self._analyze_text(text, max_span_len, diffusion_tau)
    
    def _analyze_text(self, text: str, max_span_len: int, diffusion_tau: float):
        """Send text to the analysis endpoint and display results."""
        data = {
            "texts": [text],
            "params": {
                "max_span_len": max_span_len,
                "diffusion_tau": diffusion_tau
            }
        }
        
        with st.spinner("Analyzing text..."):
            success, response = api_request(
                "POST",
                "/pipeline/analyze",
                json=data,
                headers={"Content-Type": "application/json"}
            )
            
            if not success:
                show_error("Analysis failed", response)
                return
            
            self._display_analysis_results(response)
    
    def _display_analysis_results(self, results: Dict[str, Any]):
        """Display the analysis results in an organized way."""
        st.success("Analysis complete!")
        
        # Tokens
        with st.expander("🔤 Tokens", expanded=True):
            tokens = results.get("tokens", {})
            if tokens:
                st.dataframe(tokens)
            else:
                st.info("No token data available")
        
        # Spans
        with st.expander("📜 Spans", expanded=False):
            spans = results.get("spans", [])
            if spans:
                for i, span in enumerate(spans):
                    with st.container():
                        st.markdown(f"**Span {i+1}** (tokens {span['start']}-{span['end']-1}): {span.get('text', '')}")
                        if "vectors" in span:
                            st.json(span["vectors"])
            else:
                st.info("No spans found")
        
        # Frames
        with st.expander("🖼️ Frames", expanded=False):
            frames = results.get("frames", [])
            if frames:
                for i, frame in enumerate(frames):
                    with st.container():
                        st.markdown(f"**Frame {i+1}**: {frame.get('id', 'Unknown')}")
                        if "vectors" in frame:
                            st.json(frame["vectors"])
            else:
                st.info("No frames found")
        
        # Raw JSON
        with st.expander("📄 Raw Results", expanded=False):
            st.json(results)


class SearchForm:
    """Form for semantic search."""
    
    def __init__(self):
        self.render()
    
    def render(self):
        """Render the search interface."""
        st.title("🔎 Semantic Search")
        
        st.markdown("""
        Search through indexed documents using semantic similarity to your query.
        Results are ranked by resonance with the active axis pack.
        """)
        
        # Get active pack info
        success, response = api_request("GET", "/health/ready")
        active_pack = response.get("active_pack", {}).get("pack_id") if success else None
        
        if not active_pack:
            st.warning("No active axis pack. Please create and activate an axis pack first.")
            return
        
        # Search form
        with st.form("search_form"):
            query = st.text_input("Search query", "transparent consent")
            min_coherence = st.slider("Minimum coherence", 0.0, 1.0, 0.0, 0.1)
            top_k = st.slider("Number of results", 1, 50, 10)
            
            if st.form_submit_button("Search"):
                self._perform_search(query, min_coherence, top_k)
    
    def _perform_search(self, query: str, min_coherence: float, top_k: int):
        """Perform a search and display results."""
        data = {
            "axis_pack_id": "active",  # Use active pack
            "query": {
                "type": "nl",
                "text": query
            },
            "filters": {
                "minC": min_coherence
            },
            "top_k": top_k
        }
        
        with st.spinner("Searching..."):
            success, response = api_request(
                "POST",
                "/search",
                json=data,
                headers={"Content-Type": "application/json"}
            )
            
            if not success:
                show_error("Search failed", response)
                return
            
            self._display_search_results(response)
    
    def _display_search_results(self, results: Dict[str, Any]):
        """Display search results in an organized way."""
        hits = results.get("hits", [])
        
        if not hits:
            st.info("No results found")
            return
        
        st.success(f"Found {len(hits)} results")
        
        for i, hit in enumerate(hits, 1):
            with st.container():
                st.markdown(f"### Result {i}")
                
                # Basic info
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.metric("Score", f"{hit.get('score', 0):.3f}")
                with col2:
                    st.caption(f"Document: {hit.get('doc_id', 'Unknown')}")
                    st.caption(f"Span: {hit.get('span', {}).get('start', 0)}-{hit.get('span', {}).get('end', 0)}")
                
                # Show the matched text
                st.markdown("#### Matched Text")
                st.write(hit.get("span", {}).get("text", "No text available"))
                
                # Show vectors if available
                if "vectors" in hit:
                    with st.expander("Vectors"):
                        st.json(hit["vectors"])
                
                st.markdown("---")


class WhatIfForm:
    """Form for what-if analysis."""
    
    def __init__(self):
        self.render()
    
    def render(self):
        """Render the what-if analysis interface."""
        st.title("❓ What-If Analysis")
        
        st.markdown("""
        Test how changes to a document would affect its resonance and coherence.
        This is useful for understanding the impact of edits before making them.
        """)
        
        # Get active pack info
        success, response = api_request("GET", "/health/ready")
        active_pack = response.get("active_pack", {}).get("pack_id") if success else None
        
        if not active_pack:
            st.warning("No active axis pack. Please create and activate an axis pack first.")
            return
        
        # Document selection
        doc_id = st.text_input("Document ID", "doc1")
        
        # Original text
        original_text = st.text_area(
            "Original Text",
            "The company values transparency and user consent in data processing.",
            height=100
        )
        
        # Edits
        st.markdown("### Proposed Edits")
        
        # Simple edit interface
        edit_type = st.selectbox(
            "Edit Type",
            ["replace_text", "remove_text"],
            format_func=lambda x: x.replace("_", " ").title()
        )
        
        col1, col2 = st.columns(2)
        with col1:
            start = st.number_input("Start Position", 0, len(original_text), 0)
        with col2:
            end = st.number_input(
                "End Position",
                start, len(original_text),
                min(10, len(original_text))
            )
        
        new_text = ""
        if edit_type == "replace_text":
            new_text = st.text_input("Replacement Text", "Our organization prioritizes")
        
        # Submit button
        if st.button("Analyze Impact"):
            self._analyze_impact(doc_id, original_text, edit_type, start, end, new_text)
    
    def _analyze_impact(self, doc_id: str, original_text: str, edit_type: str, 
                       start: int, end: int, new_text: str = ""):
        """Analyze the impact of a proposed edit."""
        # First index the original document
        index_success, _ = api_request(
            "POST",
            "/index",
            json={
                "doc_id": doc_id,
                "text": original_text
            },
            headers={"Content-Type": "application/json"}
        )
        
        if not index_success:
            st.warning("Could not index the original document. Some features may not work.")
        
        # Prepare the what-if request
        edits = [{
            "type": edit_type,
            "start": start,
            "end": end
        }]
        
        if edit_type == "replace_text":
            edits[0]["value"] = new_text
        
        data = {
            "axis_pack_id": "active",
            "doc_id": doc_id,
            "edits": edits
        }
        
        with st.spinner("Analyzing impact..."):
            success, response = api_request(
                "POST",
                "/whatif",
                json=data,
                headers={"Content-Type": "application/json"}
            )
            
            if not success:
                show_error("Analysis failed", response)
                return
            
            self._display_impact_results(original_text, edit_type, start, end, new_text, response)
    
    def _display_impact_results(self, original_text: str, edit_type: str, 
                              start: int, end: int, new_text: str, 
                              results: Dict[str, Any]):
        """Display the results of a what-if analysis."""
        st.success("Analysis complete!")
        
        # Show the edit preview
        st.markdown("### Edit Preview")
        
        # Create a visual diff-like display
        if edit_type == "remove_text":
            preview = (
                original_text[:start] + 
                "~~" + original_text[start:end] + "~~" +
                original_text[end:]
            )
        else:  # replace_text
            preview = (
                original_text[:start] + 
                f"~~{original_text[start:end]}~~ " +
                f"**>> {new_text} <<** " +
                original_text[end:]
            )
        
        st.markdown(preview)
        
        # Show the deltas
        st.markdown("### Impact Analysis")
        
        deltas = results.get("deltas", [])
        if not deltas:
            st.info("No significant changes detected.")
            return
        
        for delta in deltas:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("ΔU (Utility)", f"{delta.get('dU', 0):.4f}")
            with col2:
                st.metric("ΔC (Coherence)", f"{delta.get('dC', 0):.4f}")
            
            # Show vector deltas if available
            if "du" in delta:
                with st.expander("Vector Deltas"):
                    st.json({"du": delta["du"]})
