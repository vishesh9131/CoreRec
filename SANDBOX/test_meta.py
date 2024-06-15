import sys
import os
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


import streamlit as st

# Set the page config to widen the layout and set a title
# st.set_page_config(page_title='test_A', layout='wide')

# Import pages
from pages import father, credits

# Disable deprecation warning
st.set_option('deprecation.showPyplotGlobalUse', False)

# Define pages
PAGES = {
    "Core Recommendation System": father.app,
    "Credits": credits.app
}

# Simulate tabs using buttons
st.title('CoreRec by Vishesh')
tab1, tab2 = st.columns(2)

# Use session state to track active tab
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "Core Recommendation System"

# Define button actions
def set_tab(tab):
    st.session_state.active_tab = tab

with tab1:
    st.button("Core Recommendation System", on_click=set_tab, args=("Core Recommendation System",))
with tab2:
    st.button("Credits", on_click=set_tab, args=("Credits",))

# Load the selected page based on active tab
page = PAGES[st.session_state.active_tab]
page()