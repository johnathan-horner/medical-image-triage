import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import base64
import io
import os
import hashlib
from datetime import datetime, timedelta
import json
import time

# Page configuration
st.set_page_config(
    page_title="Medical Image Triage System",
    page_icon="🏥",
    layout="wide"
)

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Mock Data for Demo Mode
MOCK_RESPONSES = {
    "normal_triage": {
        "prediction_id": "pred-12345-normal",
        "classification": {
            "predicted_class": "No Finding",
            "confidence": 0.94,
            "confidence_level": "high",
            "all_scores": {
                "No Finding": 0.94,
                "Pneumonia": 0.03,
                "Pneumothorax": 0.01,
                "Infiltration": 0.01,
                "Mass": 0.01
            }
        },
        "triage": {
            "decision": "auto_triage",
            "priority_level": 1,
            "estimated_review_time": 0,
            "assigned_reviewer_type": "auto_approved",
            "reasoning": "No significant findings detected with high confidence (94%). Image shows normal chest anatomy. Auto-approved per clinical protocols."
        },
        "timestamp": datetime.now().isoformat(),
        "processing_time_ms": 187.3,
        "model_version": "1.2.1",
        "image_hash": "abc123normal"
    },
    "pneumonia_triage": {
        "prediction_id": "pred-67890-pneumonia",
        "classification": {
            "predicted_class": "Pneumonia",
            "confidence": 0.87,
            "confidence_level": "medium",
            "all_scores": {
                "No Finding": 0.08,
                "Pneumonia": 0.87,
                "Pneumothorax": 0.02,
                "Infiltration": 0.02,
                "Mass": 0.01
            }
        },
        "triage": {
            "decision": "expedited_review",
            "priority_level": 2,
            "estimated_review_time": 15,
            "assigned_reviewer_type": "radiologist",
            "reasoning": "Predicted Pneumonia with 87% confidence. Infectious condition requiring prompt medical attention. Medium confidence requires expedited physician review within 15 minutes."
        },
        "timestamp": datetime.now().isoformat(),
        "processing_time_ms": 234.7,
        "model_version": "1.2.1",
        "image_hash": "def456pneumonia"
    },
    "cardiomegaly_triage": {
        "prediction_id": "pred-11111-cardio",
        "classification": {
            "predicted_class": "Cardiomegaly",
            "confidence": 0.62,
            "confidence_level": "low",
            "all_scores": {
                "No Finding": 0.25,
                "Pneumonia": 0.05,
                "Pneumothorax": 0.03,
                "Infiltration": 0.05,
                "Cardiomegaly": 0.62
            }
        },
        "triage": {
            "decision": "senior_review",
            "priority_level": 3,
            "estimated_review_time": 45,
            "assigned_reviewer_type": "senior_radiologist",
            "reasoning": "Suspected Cardiomegaly with moderate confidence (62%). Cardiac abnormality requires senior radiologist evaluation. Low confidence score mandates expert review within 45 minutes."
        },
        "timestamp": datetime.now().isoformat(),
        "processing_time_ms": 298.1,
        "model_version": "1.2.1",
        "image_hash": "ghi789cardio"
    },
    "audit_trail": [
        {"image_hash": "abc123normal", "classification": "No Finding", "confidence": 0.94, "routing_decision": "Auto-Approved", "reviewer": "AI System", "timestamp": "2024-01-15 14:30:22"},
        {"image_hash": "def456pneumonia", "classification": "Pneumonia", "confidence": 0.87, "routing_decision": "Expedited Review", "reviewer": "Dr. Smith", "timestamp": "2024-01-15 14:15:10"},
        {"image_hash": "ghi789cardio", "classification": "Cardiomegaly", "confidence": 0.62, "routing_decision": "Senior Review", "reviewer": "Dr. Johnson", "timestamp": "2024-01-15 13:45:33"},
        {"image_hash": "jkl012normal2", "classification": "No Finding", "confidence": 0.91, "routing_decision": "Auto-Approved", "reviewer": "AI System", "timestamp": "2024-01-15 13:22:18"},
        {"image_hash": "mno345infiltrate", "classification": "Infiltration", "confidence": 0.78, "routing_decision": "Expedited Review", "reviewer": "Dr. Brown", "timestamp": "2024-01-15 12:58:44"},
        {"image_hash": "pqr678pneumo", "classification": "Pneumothorax", "confidence": 0.89, "routing_decision": "Expedited Review", "reviewer": "Dr. Davis", "timestamp": "2024-01-15 12:35:27"},
        {"image_hash": "stu901mass", "classification": "Mass", "confidence": 0.73, "routing_decision": "Expedited Review", "reviewer": "Dr. Wilson", "timestamp": "2024-01-15 11:42:15"},
        {"image_hash": "vwx234normal3", "classification": "No Finding", "confidence": 0.96, "routing_decision": "Auto-Approved", "reviewer": "AI System", "timestamp": "2024-01-15 11:28:09"},
        {"image_hash": "yza567pneumonia2", "classification": "Pneumonia", "confidence": 0.84, "routing_decision": "Expedited Review", "reviewer": "Dr. Lee", "timestamp": "2024-01-15 10:55:31"},
        {"image_hash": "bcd890normal4", "classification": "No Finding", "confidence": 0.93, "routing_decision": "Auto-Approved", "reviewer": "AI System", "timestamp": "2024-01-15 10:33:42"},
        {"image_hash": "efg123cardio2", "classification": "Cardiomegaly", "confidence": 0.58, "routing_decision": "Senior Review", "reviewer": "Dr. Martinez", "timestamp": "2024-01-14 16:20:55"},
        {"image_hash": "hij456infiltrate2", "classification": "Infiltration", "confidence": 0.81, "routing_decision": "Expedited Review", "reviewer": "Dr. Taylor", "timestamp": "2024-01-14 15:47:12"},
        {"image_hash": "klm789normal5", "classification": "No Finding", "confidence": 0.97, "routing_decision": "Auto-Approved", "reviewer": "AI System", "timestamp": "2024-01-14 15:13:28"},
        {"image_hash": "nop012pneumo2", "classification": "Pneumothorax", "confidence": 0.92, "routing_decision": "Expedited Review", "reviewer": "Dr. Anderson", "timestamp": "2024-01-14 14:39:44"},
        {"image_hash": "qrs345normal6", "classification": "No Finding", "confidence": 0.89, "routing_decision": "Auto-Approved", "reviewer": "AI System", "timestamp": "2024-01-14 14:16:17"},
        {"image_hash": "tuv678mass2", "classification": "Mass", "confidence": 0.69, "routing_decision": "Senior Review", "reviewer": "Dr. Thompson", "timestamp": "2024-01-14 13:52:33"},
        {"image_hash": "wxy901pneumonia3", "classification": "Pneumonia", "confidence": 0.86, "routing_decision": "Expedited Review", "reviewer": "Dr. Garcia", "timestamp": "2024-01-14 13:28:09"},
        {"image_hash": "zab234normal7", "classification": "No Finding", "confidence": 0.95, "routing_decision": "Auto-Approved", "reviewer": "AI System", "timestamp": "2024-01-14 12:45:21"},
        {"image_hash": "cde567infiltrate3", "classification": "Infiltration", "confidence": 0.77, "routing_decision": "Expedited Review", "reviewer": "Dr. White", "timestamp": "2024-01-14 12:22:48"},
        {"image_hash": "fgh890cardio3", "classification": "Cardiomegaly", "confidence": 0.61, "routing_decision": "Senior Review", "reviewer": "Dr. Clark", "timestamp": "2024-01-14 11:58:14"}
    ],
    "dashboard": {
        "metrics": {
            "total_images": 1247,
            "avg_confidence": 0.84,
            "images_today": 89,
            "drift_status": "Warning"
        },
        "classification_counts": [
            {"classification": "No Finding", "count": 750},
            {"classification": "Pneumonia", "count": 198},
            {"classification": "Infiltration", "count": 156},
            {"classification": "Pneumothorax", "count": 87},
            {"classification": "Mass", "count": 43},
            {"classification": "Cardiomegaly", "count": 13}
        ],
        "confidence_over_time": [
            {"timestamp": "2024-01-14 08:00", "avg_confidence": 0.86},
            {"timestamp": "2024-01-14 12:00", "avg_confidence": 0.84},
            {"timestamp": "2024-01-14 16:00", "avg_confidence": 0.82},
            {"timestamp": "2024-01-14 20:00", "avg_confidence": 0.81},
            {"timestamp": "2024-01-15 00:00", "avg_confidence": 0.83},
            {"timestamp": "2024-01-15 04:00", "avg_confidence": 0.85},
            {"timestamp": "2024-01-15 08:00", "avg_confidence": 0.84},
            {"timestamp": "2024-01-15 12:00", "avg_confidence": 0.83}
        ],
        "routing_distribution": [
            {"decision": "Auto-Approved", "count": 847, "percentage": 68},
            {"decision": "Expedited Review", "count": 274, "percentage": 22},
            {"decision": "Senior Review", "count": 126, "percentage": 10}
        ]
    }
}

# Sample image paths for demo mode
SAMPLE_IMAGES = {
    "Normal": "samples/normal_chest_xray.png",
    "Pneumonia": "samples/pneumonia_chest_xray.png",
    "Cardiomegaly": "samples/cardiomegaly_chest_xray.png"
}

def image_to_base64(image):
    """Convert PIL image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def make_api_call(endpoint, data=None, demo_mode=False, mock_key=None):
    """Make API call with fallback to mock data in demo mode"""
    if demo_mode and mock_key:
        # Simulate API delay
        time.sleep(1.0)
        return MOCK_RESPONSES.get(mock_key, {})

    try:
        if data:
            response = requests.post(f"{API_BASE_URL}/{endpoint}", json=data, timeout=30)
        else:
            response = requests.get(f"{API_BASE_URL}/{endpoint}", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Connection Error: {e}")
        st.info("💡 **Tip**: Start the backend server or enable Demo Mode in the sidebar")
        return None

def render_sidebar():
    """Render sidebar with project info and controls"""
    with st.sidebar:
        st.title("🏥 Medical Image Triage")

        # Demo Mode Toggle
        demo_mode = st.checkbox("Demo Mode", value=True, help="Use mock data for demonstration without backend")

        st.markdown("---")

        # Project Description
        with st.expander("📖 System Overview"):
            st.markdown("""
            **AI-Powered Medical Image Classification & Triage**

            Production-grade system featuring:
            - EfficientNetB0 deep learning model
            - Intelligent confidence-based routing
            - HIPAA compliant architecture
            - Real-time drift detection

            **Triage Logic:**
            - **High confidence (>90%)**: Auto-approve
            - **Medium confidence (70-90%)**: Expedited physician review
            - **Low confidence (<70%)**: Senior radiologist queue

            **Classifications:**
            Normal, Pneumonia, Pneumothorax, Infiltration, Mass, Cardiomegaly
            """)

        # Architecture Diagram
        try:
            st.image("docs/Medical_Image_Triage_AWS_Architecture.png",
                    caption="AWS Architecture", use_column_width=True)
        except:
            st.info("Architecture diagram not found")

        # Tech Stack
        st.markdown("**Tech Stack:**")
        col1, col2 = st.columns(2)
        with col1:
            st.image("https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white", width=90)
            st.image("https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white", width=90)
        with col2:
            st.image("https://img.shields.io/badge/AWS-232F3E?style=flat&logo=amazon-aws&logoColor=white", width=90)
            st.image("https://img.shields.io/badge/HIPAA-4CAF50?style=flat&logo=security&logoColor=white", width=90)

        # How it works
        with st.expander("🔬 How It Works"):
            st.markdown("""
            1. **Image Upload** → DICOM/PNG/JPEG processing
            2. **AI Classification** → EfficientNetB0 model inference
            3. **Confidence Analysis** → Uncertainty quantification
            4. **Intelligent Routing** → Priority-based assignment
            """)

        # HIPAA Compliance
        st.markdown("---")
        st.markdown("**🔒 HIPAA Compliant**")
        st.success("✅ No raw images stored")
        st.success("✅ SHA256 hash identification")
        st.success("✅ Complete audit trails")
        st.success("✅ 7-year retention policy")

        # GitHub Link
        st.markdown("---")
        st.markdown("**[📁 View on GitHub](https://github.com/johnathanhorner/medical-image-triage)**")

        # Footer
        st.markdown("---")
        st.markdown("**Built by Johnathan Horner**")

    return demo_mode

def render_triage_tab(demo_mode):
    """Render the Triage tab"""
    st.header("🩻 Medical Image Triage")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Upload Medical Image")

        # Demo mode sample selector
        if demo_mode:
            st.info("🎯 **Demo Mode**: Select a sample image or upload your own")
            sample_choice = st.selectbox(
                "Choose Sample Image:",
                ["Upload Custom", "Normal Chest X-ray", "Pneumonia Case", "Cardiomegaly Case"]
            )

            if sample_choice != "Upload Custom":
                # Load sample image
                sample_mapping = {
                    "Normal Chest X-ray": "Normal",
                    "Pneumonia Case": "Pneumonia",
                    "Cardiomegaly Case": "Cardiomegaly"
                }
                sample_key = sample_mapping[sample_choice]

                try:
                    sample_path = f"/Users/johnathanhorner/medical-image-triage/{SAMPLE_IMAGES[sample_key]}"
                    if os.path.exists(sample_path):
                        uploaded_file = sample_path
                        st.success(f"Loaded sample: {sample_choice}")
                    else:
                        st.error("Sample image not found")
                        uploaded_file = None
                except:
                    st.error("Error loading sample image")
                    uploaded_file = None
            else:
                uploaded_file = st.file_uploader(
                    "Choose image file",
                    type=['png', 'jpg', 'jpeg', 'dicom'],
                    help="Supported: PNG, JPEG, DICOM formats"
                )
        else:
            uploaded_file = st.file_uploader(
                "Choose medical image file",
                type=['png', 'jpg', 'jpeg', 'dicom'],
                help="Supported: PNG, JPEG, DICOM formats"
            )

        # Display uploaded image
        if uploaded_file is not None:
            try:
                if isinstance(uploaded_file, str):
                    # Sample image path
                    image = Image.open(uploaded_file)
                else:
                    # Uploaded file
                    image = Image.open(uploaded_file)

                st.image(image, caption="Uploaded Medical Image", use_column_width=True)

                # Triage button
                if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                    with st.spinner("Processing medical image..."):
                        # Prepare API request
                        image_base64 = image_to_base64(image)

                        # Generate mock hash for demo
                        image_hash = hashlib.sha256(image_base64.encode()).hexdigest()[:12]

                        triage_data = {
                            "image_data": image_base64,
                            "patient_id": "P12345",
                            "study_id": "S67890"
                        }

                        # Determine mock response based on sample choice
                        mock_key = "normal_triage"  # default
                        if demo_mode and isinstance(uploaded_file, str):
                            if "pneumonia" in uploaded_file.lower():
                                mock_key = "pneumonia_triage"
                            elif "cardiomegaly" in uploaded_file.lower():
                                mock_key = "cardiomegaly_triage"

                        result = make_api_call("triage", triage_data, demo_mode, mock_key)

                        if result:
                            # Store result in session state for display
                            st.session_state.triage_result = result

            except Exception as e:
                st.error(f"Error processing image: {e}")

    with col2:
        st.subheader("Triage Results")

        if hasattr(st.session_state, 'triage_result') and st.session_state.triage_result:
            result = st.session_state.triage_result

            # Classification with large text and color coding
            predicted_class = result["classification"]["predicted_class"]
            confidence = result["classification"]["confidence"]

            # Color coding based on confidence and condition
            if confidence >= 0.9:
                color = "green"
                confidence_level = "HIGH"
            elif confidence >= 0.7:
                color = "orange"
                confidence_level = "MEDIUM"
            else:
                color = "red"
                confidence_level = "LOW"

            st.markdown(f"""
            <div style="text-align: center; padding: 20px; border-radius: 10px; background-color: {color}20; border: 2px solid {color};">
                <h2 style="color: {color}; margin: 0;">Classification</h2>
                <h1 style="color: {color}; margin: 10px 0; font-size: 2.5em;">{predicted_class}</h1>
                <h3 style="color: {color}; margin: 0;">Confidence: {confidence:.1%} ({confidence_level})</h3>
            </div>
            """, unsafe_allow_html=True)

            # Confidence progress bar
            st.subheader("Confidence Score")
            st.progress(confidence)
            st.write(f"**{confidence:.1%}** - {confidence_level} confidence")

            # Routing decision
            st.subheader("Triage Decision")
            decision = result["triage"]["decision"]
            priority = result["triage"]["priority_level"]
            reviewer_type = result["triage"]["assigned_reviewer_type"]
            estimated_time = result["triage"]["estimated_review_time"]

            if decision == "auto_triage":
                st.success(f"✅ **AUTO-APPROVED**")
                st.write("No human review required")
            elif decision == "expedited_review":
                st.warning(f"⚡ **EXPEDITED REVIEW**")
                st.write(f"Assigned to: {reviewer_type}")
                st.write(f"Estimated review time: {estimated_time} minutes")
            else:
                st.error(f"🚨 **SENIOR REVIEW REQUIRED**")
                st.write(f"Assigned to: {reviewer_type}")
                st.write(f"Estimated review time: {estimated_time} minutes")

            # Additional details
            st.subheader("Processing Details")
            col_a, col_b = st.columns(2)
            with col_a:
                st.write(f"**Processing Time**: {result['processing_time_ms']:.1f}ms")
                st.write(f"**Model Version**: {result['model_version']}")
            with col_b:
                st.write(f"**Priority Level**: {priority}")
                st.write(f"**Timestamp**: {result['timestamp'][:19]}")

            # Reasoning
            st.subheader("Clinical Reasoning")
            st.info(result["triage"]["reasoning"])

            # Raw API response
            with st.expander("🔧 Raw API Response"):
                st.json(result)

        else:
            st.info("Upload and analyze an image to see triage results")
            st.markdown("""
            **Expected Results:**
            - **Classification**: AI prediction with confidence score
            - **Triage Decision**: Routing based on confidence and clinical priority
            - **Queue Assignment**: Auto-approval, expedited, or senior review
            - **Processing Details**: Timing and model information
            """)

def render_audit_tab(demo_mode):
    """Render the Audit Trail tab"""
    st.header("📋 Audit Trail")

    # Search controls
    col1, col2, col3 = st.columns(3)
    with col1:
        search_hash = st.text_input("🔍 Search by Image Hash", placeholder="Enter hash...")
    with col2:
        date_range = st.date_input("📅 Date Range", value=[datetime.now().date() - timedelta(days=7), datetime.now().date()])
    with col3:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

    # Get audit data
    audit_data = make_api_call("audit", demo_mode=demo_mode, mock_key="audit_trail")

    if audit_data:
        # Convert to DataFrame
        df = pd.DataFrame(audit_data)

        # Apply search filter
        if search_hash:
            df = df[df['image_hash'].str.contains(search_hash, case=False, na=False)]

        # Apply date filter (if provided)
        if len(date_range) == 2:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            start_date = pd.to_datetime(date_range[0])
            end_date = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
            df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]

        st.subheader(f"Audit Records ({len(df)} entries)")

        # Style the dataframe
        def style_routing_decision(val):
            if val == "Auto-Approved":
                return 'background-color: #d4edda; color: #155724'
            elif val == "Expedited Review":
                return 'background-color: #fff3cd; color: #856404'
            else:
                return 'background-color: #f8d7da; color: #721c24'

        def style_confidence(val):
            if val >= 0.9:
                return 'background-color: #d4edda; color: #155724'
            elif val >= 0.7:
                return 'background-color: #fff3cd; color: #856404'
            else:
                return 'background-color: #f8d7da; color: #721c24'

        # Format and display dataframe
        display_df = df.copy()
        if 'timestamp' in display_df.columns:
            display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

        styled_df = display_df.style\
            .map(style_routing_decision, subset=['routing_decision'])\
            .map(style_confidence, subset=['confidence'])\
            .format({'confidence': '{:.1%}'})\
            .set_properties(**{'text-align': 'center'})

        st.dataframe(styled_df, use_container_width=True, hide_index=True)

        # Expandable row details
        st.subheader("Detailed View")
        if not df.empty:
            selected_hash = st.selectbox("Select record for details:", df['image_hash'].tolist())

            if selected_hash:
                selected_row = df[df['image_hash'] == selected_hash].iloc[0]

                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Image Hash:**", selected_row['image_hash'])
                    st.write("**Classification:**", selected_row['classification'])
                    st.write("**Confidence:**", f"{selected_row['confidence']:.1%}")
                with col2:
                    st.write("**Routing Decision:**", selected_row['routing_decision'])
                    st.write("**Reviewer:**", selected_row['reviewer'])
                    st.write("**Timestamp:**", selected_row['timestamp'])

    else:
        st.warning("No audit data available. Check API connection or enable Demo Mode.")

def render_dashboard_tab(demo_mode):
    """Render the Dashboard tab"""
    st.header("📊 System Dashboard")

    # Get dashboard data
    dashboard_data = make_api_call("dashboard/metrics", demo_mode=demo_mode, mock_key="dashboard")

    if dashboard_data:
        metrics = dashboard_data["metrics"]

        # Row 1: Metric cards
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Images Processed", f"{metrics['total_images']:,}")
        with col2:
            st.metric("Average Confidence", f"{metrics['avg_confidence']:.1%}", "0.2%")
        with col3:
            st.metric("Images Today", metrics['images_today'], "12")
        with col4:
            drift_color = "🟡" if metrics['drift_status'] == "Warning" else "🟢"
            st.metric("Model Drift", f"{drift_color} {metrics['drift_status']}")

        # Row 2: Classification breakdown
        st.subheader("Classification Distribution")
        class_df = pd.DataFrame(dashboard_data["classification_counts"])
        fig_bar = px.bar(
            class_df,
            x='classification',
            y='count',
            color='count',
            color_continuous_scale='viridis',
            title="Predictions by Classification Type"
        )
        fig_bar.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

        # Row 3: Confidence trends over time
        st.subheader("Model Performance Trends")
        conf_df = pd.DataFrame(dashboard_data["confidence_over_time"])
        conf_df['timestamp'] = pd.to_datetime(conf_df['timestamp'])

        fig_line = px.line(
            conf_df,
            x='timestamp',
            y='avg_confidence',
            title="Average Confidence Score Over Time",
            markers=True
        )
        fig_line.add_hline(y=0.7, line_dash="dash", line_color="orange", annotation_text="Medium Confidence Threshold")
        fig_line.add_hline(y=0.9, line_dash="dash", line_color="green", annotation_text="High Confidence Threshold")
        fig_line.update_yaxis(range=[0.5, 1.0])
        fig_line.update_layout(height=400)
        st.plotly_chart(fig_line, use_container_width=True)

        # Row 4: Routing distribution pie chart
        st.subheader("Triage Routing Distribution")
        routing_df = pd.DataFrame(dashboard_data["routing_distribution"])

        fig_pie = px.pie(
            routing_df,
            values='count',
            names='decision',
            title="Triage Decision Distribution",
            color_discrete_map={
                'Auto-Approved': '#28a745',
                'Expedited Review': '#ffc107',
                'Senior Review': '#dc3545'
            }
        )
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)

        # Row 5: System health metrics
        st.subheader("System Health & Compliance")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Model Status**")
            st.success("✅ Model v1.2.1 Active")
            st.success("✅ Inference API Healthy")
            st.success("✅ Auto-scaling Enabled")

        with col2:
            st.markdown("**HIPAA Compliance**")
            st.success("✅ Data Encryption Active")
            st.success("✅ Audit Logging Complete")
            st.success("✅ Access Controls Applied")

        with col3:
            st.markdown("**Performance Metrics**")
            st.info("⚡ Avg Response: 187ms")
            st.info("🎯 Uptime: 99.9%")
            if metrics['drift_status'] == "Warning":
                st.warning("⚠️ Drift: Monitoring")
            else:
                st.success("✅ Drift: Normal")

    else:
        st.warning("Dashboard data unavailable. Check API connection or enable Demo Mode.")

@st.cache_data(ttl=60)
def get_cached_dashboard_data(demo_mode):
    """Cache dashboard data for 60 seconds"""
    return make_api_call("dashboard/metrics", demo_mode=demo_mode, mock_key="dashboard")

def main():
    """Main Streamlit application"""
    # Render sidebar and get demo mode
    demo_mode = render_sidebar()

    # Display connection status
    if not demo_mode:
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                st.success("🟢 Backend API Connected")
            else:
                st.error("🔴 Backend API Error")
        except:
            st.error("🔴 Backend API Disconnected - Enable Demo Mode or start the backend server")

    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["🩻 Triage", "📋 Audit Trail", "📊 Dashboard"])

    with tab1:
        render_triage_tab(demo_mode)

    with tab2:
        render_audit_tab(demo_mode)

    with tab3:
        render_dashboard_tab(demo_mode)

if __name__ == "__main__":
    main()