#!/usr/bin/env python3
"""
CoreRec Demo Frontends Application

This is the main entry point for demonstrating CoreRec's recommendation capabilities
through various platform-specific frontends.

Usage:
    streamlit run corerec/demo_frontends/demo_app.py

Or with platform selection:
    streamlit run corerec/demo_frontends/demo_app.py -- --platform spotify
    streamlit run corerec/demo_frontends/demo_app.py -- --platform youtube
    streamlit run corerec/demo_frontends/demo_app.py -- --platform netflix

This module provides multiple ways to run the demo frontends:
1. Streamlit interface (original)
2. Modern web frontend with FastAPI backend (new)
"""

import streamlit as st
import argparse
import sys
import os
from pathlib import Path

# Add the parent directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_streamlit_app():
    """Run the original Streamlit application."""
    # Configure Streamlit page
    st.set_page_config(
        page_title="CoreRec Demo Frontends",
        page_icon="🎵",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Import the frontend manager
    from frontend_manager import FrontendManager

    # Initialize the frontend manager
    frontend_manager = FrontendManager()

    # Main title
    st.title("🎵 CoreRec Demo Frontends")
    st.markdown("---")

    # Sidebar for navigation
    st.sidebar.title("Navigation")

    # Platform selection
    st.sidebar.markdown("### Choose Platform")
    platform_options = list(frontend_manager.get_available_platforms().keys())

    if platform_options:
        selected_platform = st.sidebar.selectbox(
            "Select a platform:",
            platform_options,
            format_func=lambda x: frontend_manager.get_available_platforms()[x]["name"],
        )

        # Main content area
        if selected_platform:
            platform_info = frontend_manager.get_available_platforms()[
                selected_platform]

            col1, col2 = st.columns([2, 1])

            with col1:
                st.header(f"🎯 {platform_info['name']} Demo")
                st.markdown(f"**Description:** {platform_info['description']}")

                # Launch button
                if st.button(
                        f"🚀 Launch {platform_info['name']} Demo",
                        key=f"launch_{selected_platform}"):
                    st.success(f"Launching {platform_info['name']} demo...")

                    # Load and run the platform
                    try:
                        with st.spinner(f"Loading {platform_info['name']} interface..."):
                            frontend_manager.run_platform(selected_platform)
                    except Exception as e:
                        st.error(
                            f"Error launching {platform_info['name']}: {str(e)}")

            with col2:
                # Platform stats or additional info
                st.markdown("### Platform Features")
                features = platform_info.get("features", [])
                for feature in features:
                    st.markdown(f"• {feature}")
    else:
        st.error("No platforms available. Please check your installation.")

    # Documentation section
    st.markdown("---")

    # Tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📖 Documentation", "⚙️ Configuration", "📊 Analytics", "🆘 Help"]
    )

    with tab1:
        st.markdown(
            """
        ## 📖 Documentation

        ### What are Demo Frontends?

        CoreRec Demo Frontends allow you to showcase your recommendation models through beautiful,
        platform-specific interfaces that mimic real-world applications like Spotify, YouTube, and Netflix.

        ### Available Platforms

        - **🎵 Spotify**: Music recommendation interface with playlists, artists, and tracks
        - **📺 YouTube**: Video recommendation interface with channels, videos, and categories
        - **🎬 Netflix**: Movie/TV show interface with genres, ratings, and watch history

        ### How to Use

        1. **Select a Platform**: Choose from the available platforms in the sidebar
        2. **Launch Demo**: Click the launch button to start the demo interface
        3. **Interact**: Browse recommendations, search content, and test your model
        4. **Analyze**: Use the analytics tab to see recommendation performance

        ### Integration with CoreRec

        These frontends integrate seamlessly with your CoreRec recommendation engines:

        ```python
        from corerec.engines import CollaborativeFilteringEngine
        from corerec.demo_frontends import SpotifyFrontend

        # Create your engine
        engine = CollaborativeFilteringEngine()
        engine.fit(user_item_matrix)

        # Use with demo frontend
        frontend = SpotifyFrontend(recommendation_engine=engine)
        frontend.run()
        ```
        """
        )

    with tab2:
        st.markdown("### ⚙️ Configuration")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Display Settings")
            items_per_row = st.slider("Items per row", 2, 8, 4)
            show_ratings = st.checkbox("Show ratings", True)
            show_descriptions = st.checkbox("Show descriptions", True)

        with col2:
            st.subheader("Data Settings")
            sample_size = st.slider("Sample data size", 100, 2000, 500)
            enable_caching = st.checkbox("Enable caching", True)
            use_mock_data = st.checkbox("Use mock data", False)

        if st.button("💾 Save Configuration"):
            st.success("Configuration saved successfully!")

    with tab3:
        st.markdown("### 📊 Analytics Dashboard")

        # Mock analytics data
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Users", "1,234", "12%")

        with col2:
            st.metric("Recommendations", "45,678", "8%")

        with col3:
            st.metric("Click Rate", "23.4%", "2.1%")

        with col4:
            st.metric("Engagement", "4.2 min", "0.3 min")

        # Charts
        import numpy as np
        import pandas as pd

        # Sample data for charts
        chart_data = pd.DataFrame(
            np.random.randn(
                20,
                3),
            columns=[
                "Spotify",
                "YouTube",
                "Netflix"])

        st.line_chart(chart_data)

        # Platform usage
        platform_usage = pd.DataFrame(
            {"Platform": ["Spotify", "YouTube", "Netflix"], "Usage": [45, 30, 25]}
        )

        st.bar_chart(platform_usage.set_index("Platform"))

    with tab4:
        st.markdown(
            """
        ### 🆘 Help & Support

        #### Common Issues

        **Q: Demo not loading?**
        A: Make sure you have all required dependencies installed. Run `pip install -r requirements.txt`

        **Q: No recommendations showing?**
        A: Check that your recommendation engine is properly configured and trained.

        **Q: Styling looks wrong?**
        A: Clear your browser cache and refresh the page.

        #### Troubleshooting

        1. **Check Dependencies**: Ensure all required packages are installed
        2. **Verify Data**: Make sure your data files are in the correct format
        3. **Check Logs**: Look at the console output for error messages
        4. **Restart App**: Try restarting the Streamlit application

        #### Contact & Resources

        - 📚 [Documentation](https://github.com/your-repo/corerec)
        - 🐛 [Report Issues](https://github.com/your-repo/corerec/issues)
        - 💬 [Community Forum](https://github.com/your-repo/corerec/discussions)
        - 📧 [Email Support](mailto:support@corerec.com)

        #### Version Information

        - **CoreRec Version**: 1.0.0
        - **Demo Frontends**: 1.0.0
        - **Python Version**: {sys.version.split()[0]}
        - **Streamlit Version**: {st.__version__}
        """
        )


def run_web_app():
    """Run the new web frontend application."""
    try:
        from launch import main as launch_main

        print("🌐 Starting web frontend...")
        launch_main()
    except ImportError:
        print("❌ Error: Web frontend dependencies not found.")
        print("Please ensure FastAPI and uvicorn are installed:")
        print("pip install fastapi uvicorn")
        sys.exit(1)


def main():
    """Main entry point for the demo application."""
    parser = argparse.ArgumentParser(description="CoreRec Demo Frontends")
    parser.add_argument(
        "--mode",
        choices=["streamlit", "web"],
        default="streamlit",
        help="Choose the frontend mode (streamlit or web)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=3000,
        help="Port for web frontend (default: 3000)")
    parser.add_argument(
        "--backend-only",
        action="store_true",
        help="Start only the backend API server (web mode only)",
    )
    parser.add_argument(
        "--frontend-only",
        action="store_true",
        help="Start only the frontend server (web mode only)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't automatically open browser (web mode only)")

    # Handle Streamlit execution
    if len(sys.argv) == 1 or (len(sys.argv) ==
                              2 and sys.argv[1] in ["--mode", "streamlit"]):
        # Default to Streamlit mode
        run_streamlit_app()
        return

    args = parser.parse_args()

    if args.mode == "streamlit":
        print("🚀 Starting Streamlit demo frontends...")
        run_streamlit_app()
    elif args.mode == "web":
        print("🌐 Starting web demo frontends...")

        # Set up arguments for the web launcher
        web_args = []
        if args.backend_only:
            web_args.append("--backend-only")
        if args.frontend_only:
            web_args.append("--frontend-only")
        if args.no_browser:
            web_args.append("--no-browser")
        if args.port != 3000:
            web_args.extend(["--port", str(args.port)])

        # Temporarily modify sys.argv for the web launcher
        original_argv = sys.argv[:]
        sys.argv = ["launch.py"] + web_args

        try:
            run_web_app()
        finally:
            sys.argv = original_argv


if __name__ == "__main__":
    main()
