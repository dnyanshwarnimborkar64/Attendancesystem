import streamlit as st
import sqlite3
import pandas as pd
import time

# 📌 Streamlit App Title
st.set_page_config(page_title="Attendance Dashboard", layout="wide")
st.title("📊 Attendance Records Dashboard")

# 📌 Function to fetch attendance data
def get_attendance_data():
    conn = sqlite3.connect("attendance.db")
    query = "SELECT name, class_name, entry_time, exit_time FROM attendance ORDER BY entry_time DESC"
    df = pd.read_sql(query, conn)
    conn.close()
    
    # Convert timestamps to readable format
    df["entry_time"] = pd.to_datetime(df["entry_time"]).dt.strftime("%d %b %Y, %H:%M:%S")
    df["exit_time"] = df["exit_time"].apply(lambda x: "🔴 Still in Class" if x is None else pd.to_datetime(x).strftime("%d %b %Y, %H:%M:%S"))
    
    return df

# 📌 Sidebar Filters
st.sidebar.header("🔍 Filters")
df = get_attendance_data()
selected_name = st.sidebar.selectbox("Select Name", ["All"] + list(df["name"].unique()))
selected_class = st.sidebar.selectbox("Select Class", ["All"] + list(df["class_name"].unique()))

# 📌 Apply Filters
if selected_name != "All":
    df = df[df["name"] == selected_name]
if selected_class != "All":
    df = df[df["class_name"] == selected_class]

# 📌 Display Data
st.write("### 📝 Attendance Data")
st.dataframe(df, height=500, use_container_width=True)

# 📌 Download Button
st.download_button("📥 Download CSV", df.to_csv(index=False), "attendance_records.csv", "text/csv")

# 📌 Summary Statistics
st.write("### 📊 Summary Statistics")
total_students = df["name"].nunique()
total_classes = df["class_name"].nunique()
total_records = df.shape[0]
students_in_class = df[df["exit_time"] == "🔴 Still in Class"].shape[0]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Students", total_students)
col2.metric("Total Classes", total_classes)
col3.metric("Total Attendance Records", total_records)
col4.metric("Students Currently in Class", students_in_class)

time.sleep(10)
st.rerun()  # ✅ Updated to avoid deprecated `st.experimental_rerun()`
