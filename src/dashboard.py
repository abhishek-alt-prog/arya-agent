"""
Streamlit Parent Dashboard — for monitoring Arya's learning progress.
Run: streamlit run src/dashboard.py
"""
from __future__ import annotations

import sys
import os

# Ensure src is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from src.bff_client import BFFClient
from src.config import DEFAULT_CHILD_ID

st.set_page_config(
    page_title="Arya's Learning Dashboard",
    page_icon="🦉",
    layout="wide",
)

# ── Sidebar ──────────────────────────────────────────────────────────

st.sidebar.title("🦉 Arya's Dashboard")
child_id = st.sidebar.text_input("Child ID", value=DEFAULT_CHILD_ID)

bff = BFFClient()

# ── Connection status ────────────────────────────────────────────────

if not child_id:
    st.warning("Please enter a Child ID in the sidebar.")
    st.stop()

try:
    dashboard_data = bff.get_dashboard(child_id)
except Exception as exc:
    st.error(f"Could not connect to BFF: {exc}")
    st.info("Make sure the Spring Boot BFF is running on the Mac Mini.")
    st.stop()

# ── Header ───────────────────────────────────────────────────────────

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Level", f"{dashboard_data['level']} — {dashboard_data['levelTitle']}")
with col2:
    st.metric("Total XP", f"{dashboard_data['totalXp']} ⭐")
with col3:
    st.metric("Current Streak", f"{dashboard_data['currentStreak']} days 🔥")
with col4:
    st.metric("XP to Next Level", f"{dashboard_data.get('xpToNextLevel', 0)}")

st.divider()

# ── Subject Progress ─────────────────────────────────────────────────

st.header("📚 Subject Progress")

subjects = dashboard_data.get("subjects", [])
if subjects:
    cols = st.columns(len(subjects))
    for idx, subj in enumerate(subjects):
        with cols[idx]:
            st.subheader(f"{subj['displayName']}")
            progress = subj.get("progressPercent", 0)
            st.progress(progress / 100, text=f"{progress:.0f}%")
            st.caption(
                f"{subj['lessonsCompleted']}/{subj['lessonsTotal']} lessons · "
                f"Avg {subj.get('avgStarRating', 0):.1f}★"
            )
else:
    st.info("No subject data available yet. Run the agent setup first.")

st.divider()

# ── Recent Rewards ───────────────────────────────────────────────────

st.header("🏆 Recent Rewards")

rewards = dashboard_data.get("recentRewards", [])
if rewards:
    for reward in rewards:
        st.success(f"**{reward['name']}** — {reward['description']}")
else:
    st.info("No rewards earned yet. Arya will earn badges and medals as she completes lessons!")

st.divider()

# ── Detailed Progress ───────────────────────────────────────────────

st.header("📊 Detailed Progress")

try:
    progress_data = bff.get_progress(child_id)
    if progress_data:
        import pandas as pd

        rows = []
        for p in progress_data:
            rows.append({
                "Subject": p.subject.value,
                "Unit": p.unit_name,
                "Topic": p.topic_name,
                "Mastery": f"{p.mastery_score:.1f}★",
                "Lessons Done": p.lessons_completed,
                "Difficulty": (p.current_difficulty.value if p.current_difficulty else "—"),
            })

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No detailed progress data available yet.")
except Exception:
    st.warning("Could not fetch detailed progress data.")

# ── Agent Actions ────────────────────────────────────────────────────

st.divider()
st.header("🤖 Agent Controls")

col_a, col_b = st.columns(2)
with col_a:
    if st.button("🚀 Run Initial Setup", use_container_width=True):
        st.info("Run the agent from the terminal: `python -m src.main --child-id <ID> setup`")

with col_b:
    if st.button("🔄 Run Adaptation", use_container_width=True):
        st.info("Run the agent from the terminal: `python -m src.main --child-id <ID> adapt`")

# ── Footer ───────────────────────────────────────────────────────────

st.divider()
st.caption("🦉 Arya's Learning System — Parent Dashboard v0.1")
