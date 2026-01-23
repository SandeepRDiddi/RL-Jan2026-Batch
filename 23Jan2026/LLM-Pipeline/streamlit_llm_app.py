import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
import json

# Page configuration
st.set_page_config(
    page_title="LLM Training Pipeline Visualizer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    * {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .main {
        padding-top: 0;
    }
    
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
        font-weight: 600;
    }
    
    .section-header {
        border-bottom: 3px solid #FF6B6B;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    
    .code-block {
        background-color: #f0f0f0;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #FF6B6B;
        margin: 10px 0;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.title("🧠 LLM Training Pipeline")
    st.markdown("---")
    
    page = st.radio(
        "Navigate to:",
        [
            "🏠 Home",
            "📚 Part 1: Pretraining",
            "📖 Part 2: Supervised Fine-Tuning",
            "🎯 Part 3: Reward Modeling",
            "⚙️ Part 4: RLHF & PPO",
            "💊 Pharmaceutical Applications",
            "🚨 Challenges & Solutions",
            "🎮 Interactive Simulator"
        ]
    )
    
    st.markdown("---")
    st.markdown("""
    ### 📖 About This App
    
    A comprehensive visualization of the 4-stage LLM training pipeline:
    1. **Pretraining** - Learn from massive data
    2. **SFT** - Teach structure & instructions
    3. **Reward Modeling** - Learn human preferences
    4. **RLHF** - Align with human values
    
    For Research and Study do not use for Production!
    """)

# ==================== HOME PAGE ====================
if page == "🏠 Home":
    st.markdown("""
    <div style='text-align: center; padding: 40px 0;'>
        <h1 style='font-size: 3.5rem; background: linear-gradient(135deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 20px;'>
            🧠 LLM Training Pipeline Explorer
        </h1>
        <h3 style='color: #666;'>From Raw Data to Aligned AI: A Complete Visual Guide</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📚 Stage 1: Pretraining
        **Learn general patterns from billions of tokens**
        
        - Language modeling
        - Domain knowledge capture
        - Pattern discovery
        """)
    
    with col2:
        st.markdown("""
        ### 📖 Stage 2: SFT
        **Teach instruction-following & structure**
        
        - Expert demonstrations
        - Behavioral cloning
        - Format consistency
        """)
    
    with col3:
        st.markdown("""
        ### 🎯 Stage 3 & 4: RM + RLHF
        **Align with human preferences**
        
        - Preference learning
        - Reward optimization
        - Human-AI alignment
        """)
    
    st.markdown("---")
    
    # The Pipeline Flow
    col1, col2, col3, col4, col5 = st.columns(5)
    
    stages = [
        ("Pretraining", "🔄", "Billions of Tokens", "Broad Patterns"),
        ("SFT", "📝", "10k-100k Pairs", "Structured Behavior"),
        ("RM Training", "🎯", "Preference Data", "Learn Preferences"),
        ("PPO/RLHF", "⚙️", "Policy Update", "Aligned Model"),
    ]
    
    with col1:
        st.info("**Input Data**\n\nRaw internet text, code, and domain knowledge")
    
    for i, (stage, emoji, detail, output) in enumerate(stages):
        col = st.columns(5)[i + 1]
        with col:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #667eea, #764ba2); padding: 20px; border-radius: 10px; color: white; text-align: center;'>
                <h3>{emoji} {stage}</h3>
                <p style='font-size: 0.85rem;'>{detail}</p>
                <p style='margin-top: 10px; padding-top: 10px; border-top: 1px solid white; font-weight: bold;'>{output}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Key Metrics
    st.subheader("📊 Training Scale Comparison")
    
    comparison_data = {
        "Stage": ["Pretraining", "SFT", "Reward Modeling", "RLHF"],
        "Data Size": ["Billions", "10k-100k", "5k-50k", "Variable"],
        "Training Time": ["Weeks", "Days", "Days", "Hours"],
        "Compute Cost": ["💰💰💰💰💰", "💰💰", "💰", "💰💰"],
        "Output Quality": ["Broad", "Good", "Good", "Excellent"]
    }
    
    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison, use_container_width=True)
    
    st.markdown("---")
    
    # Interactive Navigation
    st.subheader("🎯 Start Exploring")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📚 Pretraining", use_container_width=True):
            st.session_state.page = "📚 Part 1: Pretraining"
    
    with col2:
        if st.button("📖 SFT", use_container_width=True):
            st.session_state.page = "📖 Part 2: Supervised Fine-Tuning"
    
    with col3:
        if st.button("🎯 Reward Model", use_container_width=True):
            st.session_state.page = "🎯 Part 3: Reward Modeling"
    
    with col4:
        if st.button("⚙️ RLHF", use_container_width=True):
            st.session_state.page = "⚙️ Part 4: RLHF & PPO"

# ==================== PART 1: PRETRAINING ====================
elif page == "📚 Part 1: Pretraining":
    st.title("📚 Stage 1: Pretraining - Learning from Scale")
    st.markdown("Building the foundation with billions of tokens of data")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("What is Pretraining?")
        st.markdown("""
        Pretraining teaches the model to predict the next word (next token prediction) from massive amounts of unlabeled text data.
        
        **Key Insight:** By learning patterns from billions of tokens, the model acquires:
        - 📝 Language structure and grammar
        - 🧠 World knowledge
        - 💊 Domain-specific patterns (e.g., pharmaceutical knowledge)
        - 🔗 Relationships between concepts
        """)
    
    with col2:
        st.metric("Data Size", "Billions", "of tokens")
        st.metric("Training Time", "Weeks", "on massive clusters")
        st.metric("Loss Function", "Cross-Entropy", "Next token prediction")
    
    # Data Flow Visualization
    st.subheader("📊 Pretraining Data Flow")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style='background: #E3F2FD; padding: 20px; border-radius: 10px; text-align: center;'>
            <h3>📚 Input</h3>
            <p>Raw text corpus</p>
            <p style='font-size: 0.8rem;'>Wikipedia, books, code, medical journals...</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background: #F3E5F5; padding: 20px; border-radius: 10px; text-align: center;'>
            <h3>🔄 Tokenization</h3>
            <p>Split into tokens</p>
            <p style='font-size: 0.8rem;'>Words/subwords</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='background: #F1F8E9; padding: 20px; border-radius: 10px; text-align: center;'>
            <h3>🧠 Model Training</h3>
            <p>Predict next token</p>
            <p style='font-size: 0.8rem;'>Gradient descent</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style='background: #FFF3E0; padding: 20px; border-radius: 10px; text-align: center;'>
            <h3>✨ Output</h3>
            <p>Foundation model</p>
            <p style='font-size: 0.8rem;'>Generalizable patterns</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Loss Function
    st.subheader("📐 The Loss Function")
    
    st.markdown("""
    ### Cross-Entropy Loss for Next Token Prediction
    
    **Formula:**
    ```
    L_pretrain = -Σ log P_θ(w_t | w_<t)
    ```
    
    **What this means:**
    - **w_t**: The word at position t
    - **w_<t**: All previous words (context)
    - **P_θ(w_t | w_<t)**: Probability the model assigns to the correct next word
    - **-log()**: We want to maximize probability (minimize -log probability)
    
    **Goal:** Make the model assign higher probability to correct continuations
    """)
    
    # Visualization: Loss over time
    st.subheader("📈 Training Progress Example")
    
    epochs = np.arange(0, 101, 1)
    loss = 5.0 * np.exp(-epochs / 20) + 0.5
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs, y=loss,
        mode='lines+markers',
        name='Training Loss',
        fill='tozeroy',
        line=dict(color='#667eea', width=3),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title="Pretraining Loss Curve",
        xaxis_title="Epoch",
        yaxis_title="Cross-Entropy Loss",
        template='plotly_white',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Real Example
    st.subheader("🔍 Real Example: Pharmaceutical Text")
    
    st.markdown("**Input Context:** 'Patients with chronic kidney disease taking statins show'")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    predictions = [
        ("increased", 0.45),
        ("elevated", 0.30),
        ("higher", 0.15),
        ("reduced", 0.07),
        ("stable", 0.03)
    ]
    
    for col, (word, prob) in zip([col1, col2, col3, col4, col5], predictions):
        with col:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, rgba(102, 126, 234, {prob}), rgba(118, 75, 162, {prob})); 
                        padding: 20px; border-radius: 10px; text-align: center; color: white;'>
                <h4>{word}</h4>
                <p style='font-size: 1.2rem; font-weight: bold;'>{prob*100:.0f}%</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Code Example
    st.subheader("💻 Code Example (Simplified)")
    
    st.markdown("""
    ```python
    import torch
    import torch.nn as nn
    
    class PretrainedModel(nn.Module):
        def __init__(self, vocab_size, d_model):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.transformer = nn.TransformerEncoder(...)
            self.lm_head = nn.Linear(d_model, vocab_size)
        
        def forward(self, input_ids):
            # Embed tokens
            x = self.embedding(input_ids)  # [batch, seq_len, d_model]
            
            # Pass through transformer
            x = self.transformer(x)  # [batch, seq_len, d_model]
            
            # Predict next tokens
            logits = self.lm_head(x)  # [batch, seq_len, vocab_size]
            
            return logits
    
    # Training
    loss_fn = nn.CrossEntropyLoss()
    
    for batch in dataloader:
        input_ids = batch['input_ids']
        target_ids = batch['target_ids']  # Next token
        
        logits = model(input_ids)
        loss = loss_fn(logits.view(-1, vocab_size), target_ids.view(-1))
        
        loss.backward()
        optimizer.step()
    ```
    """)
    
    st.info("💡 **Key Insight:** Pretraining learns amazing things without explicit labels! Just from predicting the next word.")

# ==================== PART 2: SFT ====================
elif page == "📖 Part 2: Supervised Fine-Tuning":
    st.title("📖 Stage 2: Supervised Fine-Tuning (SFT)")
    st.markdown("Teaching the model to follow instructions and maintain structure")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("What is SFT?")
        st.markdown("""
        SFT (Behavioral Cloning) teaches the model to imitate expert demonstrations by learning instruction-following patterns.
        
        **Key Differences from Pretraining:**
        - ✅ Smaller, curated dataset (not billions of random tokens)
        - ✅ Task-specific demonstrations from experts
        - ✅ Teaches FORMAT and STRUCTURE
        - ✅ Behavioral cloning: match expert outputs exactly
        """)
    
    with col2:
        st.metric("Data Size", "10k-100k", "expert pairs")
        st.metric("Training Time", "Days", "on single GPU")
        st.metric("Loss Function", "KL Divergence", "Behavioral cloning")
    
    # SFT Data Flow
    st.subheader("📊 SFT Data Flow")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown("""
        <div style='background: #E8F5E9; padding: 15px; border-radius: 10px; text-align: center; font-size: 0.9rem;'>
            <h4>📝 Expert Data</h4>
            <p>Curated Q&A pairs</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.text("→")
    
    with col3:
        st.markdown("""
        <div style='background: #E3F2FD; padding: 15px; border-radius: 10px; text-align: center; font-size: 0.9rem;'>
            <h4>🔄 Training</h4>
            <p>Match expert outputs</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.text("→")
    
    with col5:
        st.markdown("""
        <div style='background: #FFF3E0; padding: 15px; border-radius: 10px; text-align: center; font-size: 0.9rem;'>
            <h4>✨ Result</h4>
            <p>Instruction-following model</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Pharmaceutical Example
    st.subheader("💊 Real Pharmaceutical Example")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ❌ Before SFT (After Pretraining)")
        st.markdown("""
        **Question:** "Is warfarin safe with ibuprofen?"
        
        **Model Output:**
        ```
        warfarin ibuprofen interaction bleeding 
        risk platelets nsaid anticoagulation 
        adverse event clinical trial study
        ```
        
        ❌ **Problems:**
        - No clear structure
        - No direct answer
        - Rambling and confusing
        - No actionable guidance
        """)
    
    with col2:
        st.markdown("### ✅ After SFT")
        st.markdown("""
        **Question:** "Is warfarin safe with ibuprofen?"
        
        **Model Output:**
        ```
        CONTRAINDICATED.
        
        Reason: NSAIDs inhibit platelet 
        aggregation and enhance warfarin 
        anticoagulation, increasing bleeding risk.
        
        ALTERNATIVES:
        - Acetaminophen
        - Topical diclofenac
        ```
        
        ✅ **Improvements:**
        - Clear structure
        - Direct answer first
        - Safety-focused
        - Actionable alternatives
        """)
    
    st.markdown("---")
    
    # Loss Function
    st.subheader("📐 SFT Loss Function")
    
    st.markdown("""
    ### Behavioral Cloning / Instruction-Following Loss
    
    **Formula:**
    ```
    L_SFT = -Σ log π_θ(y_expert | x)
    ```
    
    **Breaking it down:**
    - **x**: The instruction/question
    - **y_expert**: The expert's correct response
    - **π_θ**: The model's policy (probability distribution)
    - **-log()**: We want high probability for expert response
    
    **This is different from pretraining because:**
    - Pretraining: "Predict next token in any text"
    - SFT: "Given a question, output this specific response"
    
    We're teaching the model to match expert behavior exactly!
    """)
    
    # Example Training Data
    st.subheader("📚 SFT Training Data Examples")
    
    sft_examples = pd.DataFrame({
        "Question (Instruction)": [
            "Is warfarin safe with ibuprofen?",
            "What's the renal dosing for metformin with CKD stage 4?",
            "Can I take grapefruit juice with atorvastatin?",
            "What are contraindications for ACE inhibitors?"
        ],
        "Expert Response": [
            "CONTRAINDICATED. NSAIDs increase bleeding risk with warfarin.",
            "CKD Stage 4 (eGFR 15-29): Use half dose or avoid.",
            "AVOID. Grapefruit inhibits CYP3A4, increasing atorvastatin levels.",
            "Pregnancy, hyperkalemia, ACE inhibitor cough history."
        ],
        "Category": [
            "Drug Interaction",
            "Renal Dosing",
            "Food Interaction",
            "Contraindications"
        ]
    })
    
    st.dataframe(sft_examples, use_container_width=True)
    
    st.markdown("---")
    
    # Comparison: Pretraining vs SFT
    st.subheader("🔄 Comparison: Pretraining vs SFT")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📚 Pretraining
        
        | Aspect | Details |
        |--------|---------|
        | Data | Billions of tokens |
        | Task | Next token prediction |
        | Supervision | None (unsupervised) |
        | Learning | Statistical patterns |
        | Result | Broad knowledge |
        | Knowledge | Implicit in weights |
        """)
    
    with col2:
        st.markdown("""
        ### 📖 SFT
        
        | Aspect | Details |
        |--------|---------|
        | Data | 10k-100k expert pairs |
        | Task | Match expert response |
        | Supervision | Expert labels |
        | Learning | Behavioral cloning |
        | Result | Structured outputs |
        | Knowledge | Explicit examples |
        """)
    
    st.info("💡 **Key Insight:** SFT can't teach the model what it doesn't already know from pretraining. It just teaches it HOW to express its knowledge!")

# ==================== PART 3: REWARD MODELING ====================
elif page == "🎯 Part 3: Reward Modeling":
    st.title("🎯 Stage 3: Reward Modeling (RM)")
    st.markdown("Learning to score responses based on human preferences")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("What is Reward Modeling?")
        st.markdown("""
        The Reward Model learns to predict human preferences by comparing pairs of responses.
        
        **Key Insight:** 
        - We don't show the model "the best answer"
        - We show it two answers and ask "which is better?"
        - The RM learns patterns in human preferences
        """)
    
    with col2:
        st.metric("Data Size", "5k-50k", "preference pairs")
        st.metric("Training Time", "Days", "on GPU")
        st.metric("Loss Function", "Bradley-Terry", "Preference ordering")
    
    # RM Data Flow
    st.subheader("📊 Reward Model Data Flow")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style='background: #FCE4EC; padding: 15px; border-radius: 10px; text-align: center;'>
            <h4>📝 Two Responses</h4>
            <p>Response A & B</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background: #F3E5F5; padding: 15px; border-radius: 10px; text-align: center;'>
            <h4>👥 Human Eval</h4>
            <p>"Which is better?"</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='background: #E1F5FE; padding: 15px; border-radius: 10px; text-align: center;'>
            <h4>🔄 RM Training</h4>
            <p>Learn to predict choice</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style='background: #F1F8E9; padding: 15px; border-radius: 10px; text-align: center;'>
            <h4>✨ Reward Model</h4>
            <p>Scores any response</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Interactive Example
    st.subheader("💊 Real Example: Drug Interaction Assessment")
    
    st.markdown("**Question:** Drug-drug interaction: Warfarin + Ibuprofen?")
    
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col1:
        st.markdown("""
        ### Response A
        
        "Warfarin and ibuprofen can interact. 
        NSAIDs can increase bleeding risk when 
        taken with warfarin. Be careful and 
        consider alternatives like acetaminophen."
        
        ⭐ **Decent answer**
        - Identifies interaction
        - Suggests alternative
        - Generic guidance
        """)
    
    with col2:
        st.markdown("**vs**")
    
    with col3:
        st.markdown("""
        ### Response B
        
        "CONTRAINDICATED. NSAIDs inhibit 
        platelet aggregation AND enhance 
        warfarin's anticoagulant effect via 
        protein binding displacement → 
        2-3x bleeding risk. USE: Acetaminophen 
        or topical diclofenac. MONITOR: INR 
        if unavoidable."
        
        ⭐⭐⭐ **Excellent answer**
        - Clear contraindication
        - Mechanism explained
        - Risk quantified
        - Specific alternatives
        """)
    
    st.markdown("---")
    
    # Human Preference Distribution
    st.subheader("📊 Human Preference Example")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**For Medical Professionals:**")
        
        preference_data = {
            "Preference": ["Mechanism Explained", "Risk Quantified", "Specific Alternatives", 
                          "Treatment Guidelines", "Safety Warnings", "General Guidance"],
            "% Prefer Response B": [95, 92, 88, 87, 93, 45]
        }
        
        df_pref = pd.DataFrame(preference_data)
        
        fig = px.bar(df_pref, x="% Prefer Response B", y="Preference", 
                     orientation='h', color="% Prefer Response B",
                     color_continuous_scale="RdYlGn")
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("""
        ### 📐 Bradley-Terry Loss Function
        
        **Formula:**
        ```
        L_RM = -E[log σ(r(x,y_w) - r(x,y_l))]
        ```
        
        **Variables:**
        - **r(x,y_w)**: Reward for winning response
        - **r(x,y_l)**: Reward for losing response
        - **σ**: Sigmoid function
        
        **Interpretation:**
        - If winning response has higher reward: loss ≈ 0 ✅
        - If losing response has higher reward: loss ≈ 1 ❌
        
        The RM learns the reward difference!
        """)
    
    st.markdown("---")
    
    # RM Scaling Example
    st.subheader("📈 Example: Reward Scores Across Responses")
    
    responses_data = {
        "Response": [
            "Warfarin... ibuprofen... bleeding...",
            "Don't use ibuprofen with warfarin",
            "CONTRAINDICATED. Mechanism: NSAIDs inhibit platelets...",
            "No interaction detected (WRONG)"
        ],
        "Reward Score": [2.1, 5.8, 8.7, -3.2],
        "Quality": ["Poor", "Good", "Excellent", "Dangerous"]
    }
    
    df_scores = pd.DataFrame(responses_data)
    
    fig = go.Figure()
    
    colors = ['#ff6b6b' if r <= 0 else '#ffd93d' if r <= 5 else '#6bcf7f' 
              for r in df_scores["Reward Score"]]
    
    fig.add_trace(go.Bar(
        y=df_scores["Response"],
        x=df_scores["Reward Score"],
        orientation='h',
        marker=dict(color=colors),
        text=df_scores["Quality"],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="Reward Scores for Different Responses",
        xaxis_title="Reward Score",
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.success("✅ Higher reward = Model prefers this response")
    st.error("❌ Negative reward = Model learns to avoid this response")

# ==================== PART 4: RLHF ====================
elif page == "⚙️ Part 4: RLHF & PPO":
    st.title("⚙️ Stage 4: RLHF & PPO - The Final Alignment")
    st.markdown("Using reinforcement learning to optimize the model based on the reward model")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("What is RLHF?")
        st.markdown("""
        **RLHF** = Reinforcement Learning from Human Feedback
        
        We use the Reward Model as a signal to guide the SFT model towards better outputs.
        
        **PPO** (Proximal Policy Optimization) is the algorithm that does this optimization safely.
        """)
    
    with col2:
        st.metric("RL Algorithm", "PPO", "Proximal Policy")
        st.metric("Update Signal", "Reward Model", "Preference scores")
        st.metric("Constraint", "KL Divergence", "Stay close to SFT")
    
    # The Complete Pipeline
    st.subheader("🔄 The Complete 4-Stage Pipeline")
    
    pipeline_fig = go.Figure()
    
    stages = ['Pretraining', 'SFT', 'RM Training', 'PPO/RLHF']
    data_sizes = ['Billions', '100k', '50k', 'Variable']
    outputs = ['Foundation', 'Structured', 'Reward Scorer', 'Aligned Model']
    
    for i, (stage, size, output) in enumerate(zip(stages, data_sizes, outputs)):
        pipeline_fig.add_trace(go.Bar(
            y=[i],
            x=[1],
            orientation='h',
            name=stage,
            marker=dict(
                color=['#667eea', '#764ba2', '#f093fb', '#4facfe'][i],
                line=dict(color='white', width=2)
            ),
            text=f"<b>{stage}</b><br>{size} tokens<br>{output}",
            textposition='inside',
            hovertemplate=f"<b>{stage}</b><br>Data: {size}<br>Output: {output}<extra></extra>"
        ))
    
    pipeline_fig.update_layout(
        barmode='stack',
        showlegend=False,
        height=300,
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False),
        title="The 4-Stage Training Pipeline"
    )
    
    st.plotly_chart(pipeline_fig, use_container_width=True)
    
    st.markdown("---")
    
    # PPO Algorithm
    st.subheader("📐 PPO Loss Function")
    
    st.markdown("""
    ### Proximal Policy Optimization (PPO)
    
    **Formula:**
    ```
    L_PPO = E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)] - β * KL(π_new || π_sft)
    ```
    
    **Breaking it down:**
    
    1. **r_t**: The ratio of new policy probability to old policy probability
    2. **A_t**: The advantage (difference between reward and baseline)
    3. **clip()**: Prevents the policy from changing too drastically
    4. **KL term**: Keeps the new policy close to SFT (stability)
    5. **β**: Controls how much we care about staying close to SFT
    
    **Two competing goals:**
    - ✅ Maximize reward (first term)
    - ✅ Stay close to SFT model (KL term)
    
    This is the "trust region" approach: improve, but not too drastically!
    """)
    
    # Interactive PPO Visualization
    st.subheader("🎮 Interactive: Effect of β (KL coefficient)")
    
    beta = st.slider("β (KL coefficient)", 0.01, 1.0, 0.5, step=0.05)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### β = 0.1 (Low)
        
        ❌ **Problems:**
        - Large policy changes
        - Can diverge from SFT
        - Unstable training
        - Reward hacking
        
        ✅ **Benefits:**
        - Faster improvement
        - Exploits RM
        """)
    
    with col2:
        st.markdown(f"""
        ### β = {beta:.2f} (Your Choice)
        
        This balances:
        - Policy improvement
        - Training stability
        - Staying close to SFT
        """)
    
    with col3:
        st.markdown("""
        ### β = 1.0 (High)
        
        ✅ **Benefits:**
        - Stable training
        - Stays close to SFT
        - Prevents divergence
        - No reward hacking
        
        ❌ **Problems:**
        - Slower improvement
        - Less effective alignment
        - Underutilizes RM
        """)
    
    # Example: Pharmaceutical Decision
    st.subheader("💊 Example: Optimizing Medical Responses")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Before PPO/RLHF")
        st.markdown("""
        **Question:** "Warfarin + ibuprofen?"
        
        **SFT Model Output:**
        ```
        Warfarin and ibuprofen can interact. 
        NSAIDs may increase bleeding risk. 
        Consider alternatives like acetaminophen.
        ```
        
        **Reward:** 5.2/10
        - Missing mechanism
        - No risk quantification
        - Not specific enough
        """)
    
    with col2:
        st.markdown("### After PPO/RLHF")
        st.markdown("""
        **Question:** "Warfarin + ibuprofen?"
        
        **Optimized Model Output:**
        ```
        CONTRAINDICATED.
        
        Mechanism: NSAIDs inhibit platelet 
        aggregation AND enhance warfarin via 
        protein binding displacement.
        
        Risk: 2-3x increased bleeding.
        
        Alternatives:
        1. Acetaminophen (first-line)
        2. Topical diclofenac
        3. Monitor INR if necessary
        ```
        
        **Reward:** 8.9/10
        - Clear contraindication
        - Mechanism explained  
        - Risk quantified
        - Ranked alternatives
        """)
    
    st.markdown("---")
    
    # Training Dynamics
    st.subheader("📈 PPO Training Dynamics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Reward improvement
        iterations = np.arange(0, 101)
        reward = 5 + 3 * (1 - np.exp(-iterations / 15)) + 0.5 * np.random.randn(len(iterations)).cumsum() / 20
        
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=iterations, y=reward,
            mode='lines',
            name='Average Reward',
            fill='tozeroy',
            line=dict(color='#4facfe', width=3)
        ))
        
        fig1.update_layout(
            title="Reward Improvement Over Training",
            xaxis_title="PPO Iteration",
            yaxis_title="Average Reward Score",
            height=400,
            template='plotly_white'
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # KL Divergence
        kl_div = 2.0 * np.exp(-iterations / 10)
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=iterations, y=kl_div,
            mode='lines',
            name='KL Divergence',
            fill='tozeroy',
            line=dict(color='#f093fb', width=3)
        ))
        
        fig2.update_layout(
            title="KL Divergence From SFT (Stability Check)",
            xaxis_title="PPO Iteration",
            yaxis_title="KL(π_new || π_sft)",
            height=400,
            template='plotly_white'
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("---")
    
    # Algorithm Pseudocode
    st.subheader("💻 PPO Algorithm (Simplified)")
    
    st.code("""
# Initialize: SFT model, Reward Model, Reference policy (copy of SFT)
policy = sft_model
rm = reward_model
reference_policy = copy(sft_model)

for ppo_iteration in range(num_iterations):
    # 1. Generate rollouts: Sample from current policy
    prompts = sample_prompts(batch_size=256)
    responses = policy.generate(prompts)
    
    # 2. Get rewards from trained RM
    rewards = rm.score(prompts, responses)
    
    # 3. Compute advantages (rewards relative to baseline)
    advantages = rewards - baseline
    
    # 4. Update policy with PPO loss
    for mini_batch in create_mini_batches(data):
        # Compute policy ratio
        log_probs_new = policy.log_prob(mini_batch)
        log_probs_old = reference_policy.log_prob(mini_batch)
        ratio = exp(log_probs_new - log_probs_old)
        
        # PPO loss with clipping
        loss_rl = -min(ratio * advantages, 
                       clip(ratio, 1-ε, 1+ε) * advantages)
        
        # KL divergence regularization
        kl_loss = kl_divergence(policy, reference_policy)
        
        # Total loss
        total_loss = loss_rl + β * kl_loss
        
        # Gradient update
        total_loss.backward()
        optimizer.step()
    
    # 5. Update reference policy periodically
    if iteration % update_freq == 0:
        reference_policy = copy(policy)
    """, language='python')

# ==================== PHARMACEUTICAL APPLICATIONS ====================
elif page == "💊 Pharmaceutical Applications":
    st.title("💊 Real-World: Pharmaceutical Applications")
    st.markdown("How the complete pipeline ensures safety, accuracy, and alignment in medical AI")
    
    st.subheader("🔬 Case Study 1: Adverse Event Discovery in Real-World Evidence")
    
    col1, col2 = st.columns([2, 2])
    
    with col1:
        st.markdown("""
        ### The Clinical Problem
        
        **Challenge:** Finding drug-induced liver injury (DILI) in patients with CKD on statins from unstructured medical data.
        
        **Why it's hard:**
        - Millions of patient records
        - Complex interactions
        - Rare events
        - Multiple confounders
        """)
    
    with col2:
        st.markdown("""
        ### How the Pipeline Helps
        
        **Input Query:**
        "Drug-induced liver injury in patients with chronic kidney disease on statins"
        
        **What each stage does:**
        1. **Pretraining:** Knows CKD + statin + liver biochemistry
        2. **SFT:** Structures response as clinical query
        3. **RM:** Ranks results by clinical relevance
        4. **RLHF:** Optimizes for safety warnings
        """)
    
    st.markdown("---")
    
    st.subheader("📊 Data Flow Through All Three Stages")
    
    # Create a visual flow
    col1, col2, col3, col4 = st.columns(4)
    
    stages_detail = [
        {
            "name": "Stage 1: Pretraining",
            "learned": "CKD + statin → elevated liver enzymes",
            "color": "#667eea"
        },
        {
            "name": "Stage 2: SFT",
            "learned": "Structure as FAIR principles (Findable, Accessible, Interoperable, Reusable)",
            "color": "#764ba2"
        },
        {
            "name": "Stage 3: RM",
            "learned": "Rank by clinical relevance + data quality",
            "color": "#f093fb"
        },
        {
            "name": "Stage 4: RLHF",
            "learned": "Safety warnings + context prioritized by experts",
            "color": "#4facfe"
        }
    ]
    
    for i, stage in enumerate(stages_detail):
        col = st.columns(4)[i]
        with col:
            st.markdown(f"""
            <div style='background: {stage["color"]}; padding: 15px; border-radius: 10px; color: white;'>
                <h4>{stage["name"]}</h4>
                <p style='font-size: 0.85rem;'>{stage["learned"]}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Comparison Table
    st.subheader("📋 Integration Table: All Three Stages Combined")
    
    integration_data = {
        "Stage": ["Pretraining", "SFT", "RM Training", "RLHF"],
        "Objective": [
            "Learn language + domain patterns",
            "Teach query structuring",
            "Rank clinical relevance",
            "Align with expert preferences"
        ],
        "Loss Function": [
            "Cross-entropy (next token)",
            "KL divergence (behavioral cloning)",
            "Bradley-Terry (preference)",
            "PPO (policy optimization)"
        ],
        "Data Size": [
            "Billions of tokens",
            "10k-100k examples",
            "5k-50k preferences",
            "Variable"
        ],
        "Output Quality": [
            "Broad but misaligned",
            "Good but capped",
            "Good scoring function",
            "Excellent & aligned"
        ]
    }
    
    df_integration = pd.DataFrame(integration_data)
    st.dataframe(df_integration, use_container_width=True)
    
    st.markdown("---")
    
    # Case Study 2: Drug Interaction
    st.subheader("🔍 Case Study 2: Adverse Drug Interaction Detection")
    
    st.markdown("**Query:** Patient on warfarin + ibuprofen. Risk assessment?")
    
    col1, col2, col3 = st.columns(3)
    
    stages_output = [
        {
            "stage": "Pretraining Only",
            "output": '"...warfarin ibuprofen bleeding risk NSAIDs platelets..." (word salad)',
            "right": "Learned associations between keywords",
            "gap": "❌ No structure. No answer. Rambling.",
            "rating": "1/5"
        },
        {
            "stage": "After SFT",
            "output": '"Warfarin + NSAID = increased bleeding risk. CONTRAINDICATED. Use acetaminophen."',
            "right": "Clear structure. Safe answer.",
            "gap": "⚠️ Limited to training data. No mechanism.",
            "rating": "3/5"
        },
        {
            "stage": "After RLHF",
            "output": '"Contraindicated. NSAIDs inhibit platelets + enhance warfarin → 2-3x risk. ALTERNATIVES: Acetaminophen, topical diclofenac. MONITOR: INR if necessary."',
            "right": "Nuanced. Mechanism explained. Specific.",
            "gap": "✅ Exceeds expectations",
            "rating": "5/5"
        }
    ]
    
    for col, stage_info in zip([col1, col2, col3], stages_output):
        with col:
            st.markdown(f"""
            <div style='border-left: 4px solid #667eea; padding: 15px; background: #f8f9fa; border-radius: 5px;'>
                <h4>{stage_info["stage"]}</h4>
                <p style='font-size: 0.85rem; font-style: italic; margin: 10px 0;'>
                    Output: {stage_info["output"]}
                </p>
                <hr style='margin: 10px 0;'>
                <p style='font-size: 0.8rem; color: green;'><b>✓</b> {stage_info["right"]}</p>
                <p style='font-size: 0.8rem; color: #666;'>{stage_info["gap"]}</p>
                <p style='font-size: 1.2rem; font-weight: bold; color: #667eea; margin-top: 10px;'>{stage_info["rating"]}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Pharmaceutical Alignment Metrics
    st.subheader("📈 Alignment Evaluation: Medical AI Standards")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### ⚠️ The Alignment Challenge
        
        Unlike general-purpose AI, pharmaceutical AI has **life-or-death consequences**.
        
        **Misalignment = Patient Harm**
        
        We need THREE alignment pillars:
        """)
    
    with col2:
        alignment_pillars = {
            "Metric": ["Accuracy", "Safety", "Humility"],
            "Definition": [
                "Medical info correct vs gold standard",
                "Identifies contraindications",
                "Admits uncertainty appropriately"
            ],
            "Measurement": [
                "Expert review of 100 responses",
                "100 dangerous drug pairs",
                "Review for overconfident claims"
            ],
            "Target": [
                "≥ 95%",
                "100% (no false negatives)",
                "Zero absolute claims on unsettled issues"
            ]
        }
        
        df_alignment = pd.DataFrame(alignment_pillars)
        st.dataframe(df_alignment, use_container_width=True)
    
    st.markdown("---")
    
    # Why All Stages Matter
    st.subheader("🎯 Why All Four Stages Are Essential")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### ❌ What Each Stage ALONE Cannot Achieve
        
        **Pretraining Only:**
        - Knows the facts but rambles
        - No structure or format
        - Can't follow instructions
        
        **SFT Only:**
        - Limited to training distribution
        - Can't improve beyond examples
        - Capped performance
        
        **Without RLHF:**
        - No alignment with safety priorities
        - Model doesn't prioritize what experts care about
        - No optimization for preferences
        """)
    
    with col2:
        st.markdown("""
        ### ✅ What ALL FOUR Stages Together Achieve
        
        **Complete Pipeline:**
        - ✅ Broad knowledge (Pretraining)
        - ✅ Structured behavior (SFT)
        - ✅ Safety prioritization (RM)
        - ✅ Preference alignment (RLHF)
        
        **Clinical Excellence:**
        - Clear, actionable recommendations
        - Safety-focused warnings
        - Mechanism explanations
        - Evidence-based alternatives
        - Appropriate uncertainty
        """)

# ==================== CHALLENGES & SOLUTIONS ====================
elif page == "🚨 Challenges & Solutions":
    st.title("🚨 Real Challenges in LLM Training")
    st.markdown("And how practitioners solve them")
    
    challenges = [
        {
            "title": "Annotation Quality & Inconsistency",
            "problem": "Different pharmacists prioritize different aspects of drug information.",
            "example": "Annotator A: 'Specific eGFR cutoffs (safety-focused)' vs Annotator B: 'Alternative drugs (flexibility-focused)'",
            "solutions": [
                "Inter-Rater Reliability (Fleiss' Kappa ≥ 0.60) threshold",
                "Multi-Annotator Consensus (2/3 agreement required)",
                "Calibration sessions with explicit guidelines"
            ]
        },
        {
            "title": "Reward Hacking",
            "problem": "Reward Model has implicit defects; policy exploits them rather than improving genuinely.",
            "example": "If RM rewards length: Model generates verbose, repetitive text even if content quality drops",
            "solutions": [
                "Human validation: Experts rate PPO-generated outputs independently",
                "KL Regularization: Higher β keeps model closer to SFT",
                "Ensemble RMs: Train multiple RMs; average their scores"
            ]
        },
        {
            "title": "Distribution Shift",
            "problem": "RM trained on SFT-generated examples; PPO produces increasingly different outputs.",
            "example": "Out-of-distribution samples have unreliable RM scores. Model generates novel responses RM hasn't seen.",
            "solutions": [
                "Active Learning: Collect human preferences on PPO-generated samples",
                "Iterative Refinement: Run RM training → PPO → RM training cycles",
                "Continuous retraining on newer distributions"
            ]
        },
        {
            "title": "Scalability of Human Preference Data",
            "problem": "Expert pharmacist time is expensive. Collecting 50k preference pairs is very costly.",
            "example": "High-quality annotation is a bottleneck for larger models",
            "solutions": [
                "Focus on High-Value Data: Collect on diverse, challenging examples only",
                "Unlabeled Data Ranking: Use trained RM to rank large unlabeled datasets",
                "Crowdsourcing with Validation: Use general annotators with expert spot-checks"
            ]
        }
    ]
    
    # Challenge Selector
    challenge_idx = st.radio(
        "Select a challenge to explore:",
        range(len(challenges)),
        format_func=lambda i: challenges[i]["title"],
        horizontal=True
    )
    
    challenge = challenges[challenge_idx]
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(f"""
        ### 🔴 Problem
        
        {challenge['problem']}
        """)
    
    with col2:
        st.markdown(f"""
        ### 📝 Real Example
        
        *{challenge['example']}*
        """)
    
    st.markdown("---")
    
    st.subheader("✅ Solutions")
    
    solution_cols = st.columns(len(challenge['solutions']))
    
    for col, solution in zip(solution_cols, challenge['solutions']):
        with col:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #4facfe, #00f2fe); 
                        padding: 20px; border-radius: 10px; color: white; height: 150px;'>
                <h4 style='margin-top: 0;'>✓</h4>
                <p>{solution}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Challenge-specific details
    if challenge_idx == 0:  # Annotation Quality
        st.subheader("📊 Inter-Rater Reliability Example")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Fleiss' Kappa Interpretation
            
            - **κ < 0.20**: Poor agreement
            - **0.20 - 0.40**: Fair agreement
            - **0.40 - 0.60**: Moderate agreement
            - **0.60 - 0.80**: Substantial agreement ✅
            - **0.80 - 1.00**: Almost perfect agreement
            
            **Target:** κ ≥ 0.60 for medical applications
            """)
        
        with col2:
            # Simulate Kappa across domains
            domains = ['Drug Dosing', 'Contraindications', 'Interactions', 'Side Effects', 'Monitoring']
            kappa_scores = [0.72, 0.68, 0.75, 0.65, 0.71]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=domains,
                y=kappa_scores,
                marker=dict(color=kappa_scores, colorscale='RdYlGn', showscale=True),
                text=[f'{k:.2f}' for k in kappa_scores],
                textposition='auto'
            ))
            
            fig.add_hline(y=0.60, line_dash="dash", line_color="red", 
                         annotation_text="Target Threshold")
            
            fig.update_layout(
                title="Fleiss' Kappa by Domain",
                yaxis_title="Kappa Score",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    elif challenge_idx == 1:  # Reward Hacking
        st.subheader("📈 Reward Hacking Example")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Without Safeguards
            
            If RM only rewards **length**:
            
            ```
            Good Response (100 words):
            "The answer is A"
            Reward: 2/10 ❌
            
            Hacked Response (500 words):
            "The answer could potentially be A 
            if we consider various factors... 
            and in some cases it might be..."
            (repeated verbosity)
            Reward: 9/10 ✅
            ```
            
            Model learns to pad text!
            """)
        
        with col2:
            st.markdown("""
            ### With KL Regularization
            
            PPO Loss = RL Loss + β × KL(π_new || π_sft)
            
            **Effect of different β values:**
            
            | β | Length Reward | Actual Quality | Overall Score |
            |---|---|---|---|
            | 0.1 | 8/10 | 2/10 | Hacked ❌ |
            | 0.3 | 6/10 | 5/10 | Better ⚠️ |
            | 0.5 | 5/10 | 7/10 | Good ✅ |
            | 0.8 | 4/10 | 8/10 | Excellent ✅ |
            
            Higher β prevents exploitation!
            """)

# ==================== INTERACTIVE SIMULATOR ====================
elif page == "🎮 Interactive Simulator":
    st.title("🎮 Interactive Simulator: Build Your Own Model!")
    st.markdown("Adjust hyperparameters and see how they affect training outcomes")
    
    st.subheader("⚙️ Training Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Pretraining Settings")
        pretrain_data = st.slider("Data Size (tokens)", 1, 100, 50, step=5)
        pretrain_epochs = st.slider("Epochs", 1, 20, 10)
    
    with col2:
        st.markdown("### SFT Settings")
        sft_data = st.slider("Expert Pairs", 1000, 100000, 50000, step=10000)
        sft_lr = st.select_slider("Learning Rate", [0.0001, 0.0005, 0.001, 0.005, 0.01], value=0.001)
    
    with col3:
        st.markdown("### RLHF Settings")
        rlhf_beta = st.slider("β (KL coefficient)", 0.01, 1.0, 0.5, step=0.05)
        rlhf_iters = st.slider("PPO Iterations", 10, 200, 100, step=10)
    
    st.markdown("---")
    
    # Simulate training
    if st.button("🚀 Run Training Simulation", use_container_width=True):
        st.success("Training started! Simulating results...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Simulation
        for i in range(100):
            progress = i / 100
            progress_bar.progress(progress)
            status_text.text(f"Training: {i}%")
        
        progress_bar.progress(1.0)
        status_text.text("✅ Training Complete!")
        
        st.markdown("---")
        st.subheader("📊 Results")
        
        # Simulated results based on hyperparameters
        base_accuracy = 0.7
        accuracy_boost = (pretrain_data / 100) * 0.15
        sft_boost = (sft_data / 100000) * 0.10
        rlhf_boost = (rlhf_iters / 200) * 0.08 - (rlhf_beta * 0.02)
        
        final_accuracy = min(0.95, base_accuracy + accuracy_boost + sft_boost + rlhf_boost)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Final Accuracy", f"{final_accuracy*100:.1f}%", 
                     f"+{(final_accuracy - base_accuracy)*100:.1f}%")
        
        with col2:
            safety_score = min(1.0, 0.7 + (rlhf_beta * 0.25))
            st.metric("Safety Score", f"{safety_score*100:.0f}%", "Excellent" if safety_score > 0.85 else "Good")
        
        with col3:
            stability = min(1.0, 0.5 + (rlhf_beta * 0.5))
            st.metric("Training Stability", f"{stability*100:.0f}%", "Stable" if stability > 0.8 else "Moderate")
        
        with col4:
            inference_speed = 100 / (1 + sft_data / 50000)
            st.metric("Inference Speed", f"{inference_speed:.0f} tokens/sec", "Fast" if inference_speed > 80 else "Moderate")
        
        st.markdown("---")
        
        # Performance graphs
        col1, col2 = st.columns(2)
        
        with col1:
            # Accuracy progression
            epochs_sim = np.arange(0, pretrain_epochs + rlhf_iters // 10)
            acc_curve = 0.5 + 0.45 * (1 - np.exp(-epochs_sim / 5))
            
            fig_acc = go.Figure()
            fig_acc.add_trace(go.Scatter(
                x=epochs_sim, y=acc_curve,
                mode='lines+markers',
                fill='tozeroy',
                name='Accuracy',
                line=dict(color='#4facfe', width=3)
            ))
            
            fig_acc.update_layout(
                title="Accuracy During Training",
                xaxis_title="Training Step",
                yaxis_title="Accuracy",
                height=400
            )
            
            st.plotly_chart(fig_acc, use_container_width=True)
        
        with col2:
            # Reward progression
            reward_curve = -2 * np.exp(-epochs_sim / 8) + 8 + 0.5 * np.sin(epochs_sim / 3)
            
            fig_reward = go.Figure()
            fig_reward.add_trace(go.Scatter(
                x=epochs_sim, y=reward_curve,
                mode='lines+markers',
                fill='tozeroy',
                name='Reward',
                line=dict(color='#4ade80', width=3)
            ))
            
            fig_reward.update_layout(
                title="Reward Model Score During RLHF",
                xaxis_title="PPO Iteration",
                yaxis_title="Average Reward",
                height=400
            )
            
            st.plotly_chart(fig_reward, use_container_width=True)
        
        st.markdown("---")
        
        st.subheader("💡 Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            insights = []
            
            if pretrain_data < 30:
                insights.append("⚠️ Low pretraining data: Consider scaling up for better foundation")
            else:
                insights.append("✅ Good pretraining scale: Foundation is solid")
            
            if sft_data < 30000:
                insights.append("⚠️ Limited SFT pairs: May limit structured behavior")
            else:
                insights.append("✅ Sufficient SFT data: Good instruction coverage")
            
            if rlhf_beta < 0.2:
                insights.append("⚠️ Low β: Risk of reward hacking and instability")
            elif rlhf_beta > 0.8:
                insights.append("⚠️ High β: May limit improvement from RL")
            else:
                insights.append("✅ Balanced β: Good trade-off between improvement and stability")
            
            for insight in insights:
                st.info(insight)
        
        with col2:
            st.markdown("""
            ### 🎯 Recommendations
            
            For pharmaceutical applications:
            
            1. **Pretraining:** 50B+ tokens recommended for domain knowledge
            2. **SFT:** 50k+ expert examples for comprehensive coverage
            3. **RM:** 20k+ preference pairs from domain experts
            4. **RLHF:** β = 0.5-0.7 for safety + performance balance
            
            **Golden Rule:** Don't neglect any stage! Each is essential.
            """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #999; padding: 20px;'>
    <p>📚 Educational Tool for LLM Training Pipeline</p>
    <p>Perfect for students, researchers, and practitioners</p>
</div>
""", unsafe_allow_html=True)
