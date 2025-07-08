import json

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.title("Translation Quality Dashboard")

data = json.load(open("eval_results/latest.json", encoding="utf-8"))
st.write(f"Run time: {data['timestamp']}")

# Display raw scores for reference
st.write(f"BLEU Score: {data['BLEU']:.2f}")
st.write(f"COMET Score: {data['COMET']:.4f}")

# Normalize scores for better visualization (both as percentages)
df = pd.DataFrame({
    "Metric": ["BLEU (%)", "COMET (%)"],
    "Score": [data["BLEU"], data["COMET"] * 100]
})

# Create a custom bar chart with value labels
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(df["Metric"], df["Score"], color=['#1f77b4', '#ff7f0e'])

# Add value labels on top of bars
for i, (bar, score) in enumerate(zip(bars, df["Score"])):
    height = bar.get_height()
    if i == 0:  # BLEU score
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{data["BLEU"]:.2f}', ha='center', va='bottom', fontweight='bold')
    else:  # COMET score
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{data["COMET"]:.4f}', ha='center', va='bottom', fontweight='bold')

ax.set_ylabel('Score')
ax.set_title('Translation Quality Metrics')
ax.set_ylim(0, 105)  # Set y-axis limit to give space for labels

st.pyplot(fig)

st.subheader("Hypotheses")
for i, hyp in enumerate(data["hypotheses"], 1):
    st.write(f"{i}. {hyp}")
