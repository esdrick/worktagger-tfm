import streamlit as st

# NOTE: This tutorial uses the global session keys aligned with the app:
# ONBOARDING_ACTIVE (bool), ONBOARDING_STEP (int), ONBOARDING_COMPLETED (bool)
# Optional helpers: ONBOARDING_AUTO_SHOWN (bool), ONBOARDING_EXPANDER (bool)

def show_classification_tutorial():
    """Simplified interactive tutorial — without automatic file detection"""
    
    # Initialize tutorial state if it doesn't exist (aligned with app_core_act_db)
    if "ONBOARDING_COMPLETED" not in st.session_state:
        st.session_state["ONBOARDING_COMPLETED"] = False
    if "ONBOARDING_STEP" not in st.session_state:
        st.session_state["ONBOARDING_STEP"] = 0
    if "ONBOARDING_ACTIVE" not in st.session_state:
        st.session_state["ONBOARDING_ACTIVE"] = False

    # Only auto-show if it's the first time AND there is unclassified data
    if (not st.session_state["ONBOARDING_COMPLETED"] and 
        "df_original" in st.session_state and 
        st.session_state.df_original is not None and 
        not st.session_state.df_original.empty):
        if 'Activity' in st.session_state.df_original.columns:
            if (st.session_state.df_original['Activity'].isna().all() and 
                not st.session_state.get("ONBOARDING_AUTO_SHOWN", False)):
                st.session_state["ONBOARDING_ACTIVE"] = True
                st.session_state["ONBOARDING_AUTO_SHOWN"] = True

    # Show tutorial if it is active
    if st.session_state["ONBOARDING_ACTIVE"]:
        _render_simple_tutorial()

@st.fragment
def _render_simple_tutorial():
    """Tutorial using native Streamlit components with fragment to avoid full reruns"""
    
    steps = [
        {
            "title": "🎯 Welcome to the Classification System",
            "content": """
            I'll help you understand how to classify your activities.
            
            **Why classify?**
            - 📊 Analyze your productivity
            - 🎯 Identify work patterns
            - ⚡ Optimize your time
            
            This tutorial takes **2 minutes** and will save you hours of confusion.
            """,
        },
        {
            "title": "1️⃣ Select Activities",
            "content": """
            **First step:** Select the activities you want to classify.
            
            **Selection methods:**
            - ✅ **Select all**: Selects the current page
            - 🚫 **Select none**: Deselects everything
            - 🔄 **Invert selection**: Inverts the selection
            - ☑️ **Individual**: Use each row's checkbox
            
            💡 **Tip:** Start by selecting a few activities to practice.
            """,
        },
        {
            "title": "2️⃣ Automatic Classification with GPT",
            "content": """
            **The easiest way to start:**
            
            **Steps:**
            1. 🔑 **IMPORTANT:** Enter your OpenAI API key
            2. 📊 Choose which data to classify
            3. ▶️ Click "Start classification"
            4. ⏳ Wait (it may take several minutes)
            
            **Advantages:**
            - 🚀 Very fast for many activities
            - 🎯 Consistent results
            - 🧠 Learns from your patterns
            
            ⚠️ **Note:** You need a paid OpenAI API key. Without it, use manual or heuristic classification.
            """,
        },
        {
            "title": "3️⃣ Heuristic Classification",
            "content": """
            **A free and fast alternative:**
            
            **What it does:**
            - 🧠 Uses predefined rules based on app names
            - 📱 Detects patterns in window titles
            - ⚡ Classifies automatically at no cost
            - 🎯 Works well for common apps
            
            **Available types:**
            - **Heuristic Prediction**: Predicts activity and subactivity
            - **Heuristic Eisenhower classification**: Assigns quadrants automatically
            
            💡 **Ideal for:** A quick initial pass before manual refinement.
            """,
        },
        {
            "title": "4️⃣ Manual Classification",
            "content": """
            **For full control over labels:**
            
            **How it works:**
            1. 📝 Select activities
            2. 🎯 Choose a main category (e.g., "Work")
            3. 🔍 Select a specific subcategory
            4. ✅ It is applied automatically
            
            **Main categories:**
            - 💼 **Work** - Work-related activities
            - 🎓 **Study** - Study and learning
            - 🌐 **Internet** - Web browsing
            - 📱 **Social** - Social networks
            - 🎮 **Entertainment** - Entertainment
            
            💡 **Pro tip:** Recent labels appear at the top for quick reuse.
            """,
        },
        {
            "title": "5️⃣ Eisenhower Matrix",
            "content": """
            **For productivity analysis:**
            
            **The 4 quadrants:**
            - 🔴 **I: Urgent & Important** - Crises, emergencies
            - 🟢 **II: Not urgent but Important** - Planning, self-improvement
            - 🟡 **III: Urgent but Not important** - Interruptions, urgent emails
            - ⚪ **IV: Not urgent & Not important** - Distractions, social media
            
            **Goal:** Maximize time in quadrant II, minimize IV.
            
            💡 **After classifying:** Go to the "Eisenhower Matrix" to see your full analysis.
            """,
        },
        {
            "title": "6️⃣ Cases/Projects",
            "content": """
            **To organize by context or project:**
            
            **What are cases?**
            - 🏢 They group activities by specific project
            - 📚 Enable context-based analysis
            - 🎯 Useful for tracking time by client/task
            
            **Example cases:**
            - "Project Alpha" — Specific work
            - "Python Course" — Learning
            - "Household tasks" — Personal
            - "Freelance Client X" — External work
            
            **How to create:**
            1. 📝 Type the case name
            2. ✅ Click "Assign"
            3. 🔄 Reuse existing cases with the buttons
            
            💡 **Useful for:** Billing and project-based productivity analysis.
            """,
        },
        {
            "title": "🎉 Ready to Start!",
            "content": """
            **Next steps:**
            1. 🚀 Start by classifying a few activities
            2. 📊 Visit "Activities Dashboard" to see charts
            3. 🧭 Go to the "Eisenhower Matrix" for productivity analysis
            4. 🤖 Use the "Productivity Assistant" for personalized tips
            
            **Remember:**
            - 🔄 You can use "Undo" to revert changes
            - 💾 Download your labeled data with "Download CSV"
            - ❓ Each tool has explanatory tooltips
            - 🎯 Combine methods: heuristic + manual for the best results
            
            **Let's classify!** 🎯
            """,
        }
    ]
    
    current_step = st.session_state["ONBOARDING_STEP"]
    
    # Use native Streamlit expander for the tutorial
    with st.expander("🎓 Classification Tutorial", expanded=st.session_state.get("ONBOARDING_EXPANDER", True)):
        # Show current step
        st.markdown(f"### {steps[current_step]['title']}")
        st.markdown(steps[current_step]['content'])
        
        # Progress bar
        progress = (current_step + 1) / len(steps)
        st.progress(progress, text=f"Step {current_step + 1} of {len(steps)} ({int(progress * 100)}%)")
        
        # Navigation buttons
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if current_step > 0:
                if st.button("⬅️ Previous", key="tut_prev"):
                    st.session_state["ONBOARDING_STEP"] -= 1
                    st.rerun()
        
        with col2:
            if st.button("❌ Close", key="tut_close"):
                _close_tutorial()
                st.rerun()
        
        with col3:
            if st.button("⏭️ Skip", key="tut_skip"):
                _close_tutorial()
                st.rerun()
        
        with col4:
            if current_step < len(steps) - 1:
                if st.button("Next ➡️", key="tut_next"):
                    st.session_state["ONBOARDING_STEP"] += 1
                    st.rerun()
            else:
                if st.button("🎉 Start!", key="tut_start"):
                    _close_tutorial()
                    st.rerun()

def _close_tutorial():
    """Closes the tutorial and marks it as completed"""
    st.session_state["ONBOARDING_ACTIVE"] = False
    st.session_state["ONBOARDING_COMPLETED"] = True
    st.session_state["ONBOARDING_STEP"] = 0

@st.fragment
def show_tutorial_button():
    """Button to show the tutorial again"""
    if st.button("❓ View Tutorial", help="Step-by-step classification guide"):
        st.session_state["ONBOARDING_ACTIVE"] = True
        st.session_state["ONBOARDING_COMPLETED"] = False
        st.session_state["ONBOARDING_STEP"] = 0
        st.rerun()

@st.fragment
def show_quick_help():
    """Quick help panel — only if the tutorial is NOT active"""
    # Only show if tutorial is not active
    if not st.session_state.get("ONBOARDING_ACTIVE", False):
        with st.expander("🆘 Quick Help", expanded=False):
            st.markdown("""
            **🚀 Quick Start:**
            1. ☑️ Select activities (checkboxes)
            2. 🛠️ Choose a classification tool
            3. ⚡ Apply labels
            
            **🔧 Available tools:**
            - 🤖 **Automatic**: AI classifies for you (requires API key)
            - 🧠 **Heuristic**: Fast and free rules
            - ✋ **Manual**: Full control
            - 🎯 **Eisenhower**: For productivity analysis
            - 📋 **Cases**: Project-based organization
            
            **❓ First time classifying?**
            """)
            
            show_tutorial_button()

# Simplified function to integrate
def integrate_tutorial_in_classification():
    """Integrates the simplified tutorial"""
    # Show tutorial if active
    show_classification_tutorial()
    # Quick help panel (only if the tutorial is not active)
    if not st.session_state.get("ONBOARDING_ACTIVE", False):
        show_quick_help()