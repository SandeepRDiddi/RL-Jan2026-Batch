# 🧠 LLM Training Pipeline Visualizer - Streamlit App

A comprehensive, interactive educational tool that visualizes the complete 4-stage LLM training pipeline with real-world pharmaceutical applications.

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [App Pages](#app-pages)
- [Requirements](#requirements)
- [Educational Use](#educational-use)

## ✨ Features

### 🏠 Home Page
- Visual overview of the 4-stage pipeline
- Pipeline flow diagram
- Training scale comparison table
- Quick navigation to each stage

### 📚 Part 1: Pretraining
- **What is Pretraining?** - Foundational concepts
- **Data Flow Visualization** - Input → Tokenization → Training → Output
- **Loss Function Explanation** - Cross-entropy loss with mathematical formula
- **Training Progress Chart** - Real loss curve visualization
- **Real Pharmaceutical Example** - Next token probability predictions
- **Code Example** - Simplified PyTorch implementation

### 📖 Part 2: Supervised Fine-Tuning (SFT)
- **What is SFT?** - Behavioral cloning fundamentals
- **Data Flow Visualization** - Expert pairs to structured outputs
- **Before/After Comparison** - Pretraining vs SFT outputs
- **Real Pharmaceutical Examples** - Drug interaction queries
- **SFT Training Data** - Example Q&A pairs with categories
- **Pretraining vs SFT Comparison Table** - Key differences

### 🎯 Part 3: Reward Modeling
- **What is Reward Modeling?** - Preference learning fundamentals
- **Data Flow Visualization** - Response pairs to reward scores
- **Interactive Example** - Drug-drug interaction assessment
- **Human Preference Distribution** - Visualization of expert preferences
- **Bradley-Terry Loss Function** - Mathematical foundation
- **Reward Scores Visualization** - Bar chart of response quality scores

### ⚙️ Part 4: RLHF & PPO
- **What is RLHF?** - Reinforcement learning concepts
- **Complete Pipeline Visualization** - All 4 stages together
- **PPO Loss Function** - Formula with detailed breakdown
- **Interactive β Parameter Tuning** - Explore KL coefficient effects
- **Pharmaceutical Optimization Example** - Before/after RLHF outputs
- **Training Dynamics Charts** - Reward improvement and KL divergence
- **PPO Algorithm Pseudocode** - Simplified implementation

### 💊 Pharmaceutical Applications
- **Case Study 1:** Adverse event discovery in real-world evidence
- **Case Study 2:** Adverse drug interaction detection
- **Integration Table:** All stages compared
- **Data Flow Visualization:** Through all 4 stages
- **Clinical Example:** Warfarin + Ibuprofen interaction assessment
- **Alignment Metrics:** Accuracy, Safety, and Humility targets

### 🚨 Challenges & Solutions
- **Challenge 1:** Annotation Quality & Inconsistency
  - Problem: Different experts prioritize different aspects
  - Solutions: Fleiss' Kappa, consensus, calibration
  
- **Challenge 2:** Reward Hacking
  - Problem: Model exploits RM defects
  - Solutions: Human validation, KL regularization, ensemble RMs
  
- **Challenge 3:** Distribution Shift
  - Problem: RM trained on different distribution than PPO
  - Solutions: Active learning, iterative refinement
  
- **Challenge 4:** Scalability of Human Preference Data
  - Problem: Expert time is expensive
  - Solutions: High-value data focus, uncertainty sampling, crowdsourcing

### 🎮 Interactive Simulator
- **Hyperparameter Configuration:**
  - Pretraining: Data size and epochs
  - SFT: Expert pairs and learning rate
  - RLHF: β coefficient and PPO iterations
  
- **Training Simulation:** Visual progress bar
- **Results Dashboard:** 
  - Final accuracy
  - Safety score
  - Training stability
  - Inference speed
  
- **Performance Visualizations:**
  - Accuracy progression curve
  - Reward improvement during RLHF
  
- **Automated Insights:** Recommendations based on configuration

## 📦 Installation

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Step 1: Clone or Download
```bash
# If you have the file
cp streamlit_llm_app.py ~/
cd ~
```

### Step 2: Install Dependencies
```bash
pip install streamlit plotly pandas numpy
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

## 🚀 How to Run

### Simple Way (Recommended)
```bash
streamlit run streamlit_llm_app.py
```

The app will automatically open in your default browser at `http://localhost:8501`

### Advanced: Run on Specific Port
```bash
streamlit run streamlit_llm_app.py --server.port 8000
```

### Run in Headless Mode (Server)
```bash
streamlit run streamlit_llm_app.py --server.headless true --server.enableCORS false
```

## 📄 App Pages

### Navigation
Use the sidebar to navigate between pages:

1. **🏠 Home** - Overview and quick navigation
2. **📚 Part 1: Pretraining** - Foundation stage deep-dive
3. **📖 Part 2: SFT** - Instruction-following stage
4. **🎯 Part 3: Reward Modeling** - Preference learning stage
5. **⚙️ Part 4: RLHF & PPO** - Alignment stage
6. **💊 Pharmaceutical Applications** - Real-world use cases
7. **🚨 Challenges & Solutions** - Common pitfalls and solutions
8. **🎮 Interactive Simulator** - Hands-on hyperparameter tuning

## 📋 Requirements

### Python Packages
```
streamlit>=1.28.0
plotly>=5.17.0
pandas>=2.0.0
numpy>=1.24.0
```

### System Requirements
- 2GB RAM minimum
- Modern web browser (Chrome, Firefox, Safari, Edge)
- Internet connection (for Plotly library rendering)

## 🎓 Educational Use

### For Students
- **Concept Learning:** Each section explains one stage with examples
- **Visual Learning:** Charts, diagrams, and interactive visualizations
- **Real-World Context:** Pharmaceutical applications throughout
- **Hands-On Practice:** Interactive simulator for experimentation

### For Instructors
- **Classroom Presentation:** Full-screen mode for projectors
- **Student Assignments:** "Create your own model" with the simulator
- **Guided Exploration:** Guide students through each page
- **Code Examples:** Copy-paste ready code snippets

### For Researchers
- **Algorithm Reference:** Mathematical formulas with explanations
- **Hyperparameter Effects:** Visualize how β affects training
- **Benchmark Understanding:** See how challenges affect outcomes
- **Implementation Patterns:** Simplified code showing key concepts

## 🔧 Customization

### Modify Colors
Edit the CSS section in the code:
```python
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        ...
    }
</style>
""", unsafe_allow_html=True)
```

### Add Your Own Examples
Add to the respective sections:
```python
custom_example = {
    "title": "Your Case Study",
    "description": "...",
    "output": "..."
}
```

### Adjust Simulator Ranges
Modify the slider parameters:
```python
st.slider("Parameter Name", min_value, max_value, default_value)
```

## 📊 Learning Path Recommendation

### Week 1: Foundations
1. Visit Home page
2. Go through Part 1: Pretraining
3. Understand loss functions

### Week 2: Practical Applications
1. Study Part 2: SFT
2. Study Part 3: Reward Modeling
3. Examine pharmaceutical case studies

### Week 3: Advanced Topics
1. Deep-dive into Part 4: RLHF & PPO
2. Explore Challenges & Solutions
3. Experiment with Interactive Simulator

### Week 4: Integration
1. Review complete pipeline
2. Study all case studies
3. Discuss real-world applications

## 🤝 Contributing

To add new content:
1. Add new page in sidebar navigation
2. Create section with st.subheader()
3. Add visualizations with plotly
4. Include code examples
5. Test all interactive elements

## 📚 References

- Christiano et al. (2017): "Deep Reinforcement Learning from Human Preferences"
- Ouyang et al. (2022): "Training language models to follow instructions with human feedback"
- Schulman et al. (2017): "Proximal Policy Optimization Algorithms"
- Bradley & Terry (1952): "Rank Analysis of Incomplete Block Designs"
- Anthropic Research: Constitutional AI and alignment methodology

## 💡 Tips for Best Experience

1. **Full Screen:** Press F11 for full-screen browsing
2. **Wide Display:** Use a wide monitor for better layout
3. **Slow Interaction:** Wait for charts to fully render before clicking
4. **Reset App:** Click ⋮ → Rerun in top-right
5. **Save Outputs:** Use browser's print function to save pages

## 🐛 Troubleshooting

### App doesn't start
```bash
# Make sure Streamlit is installed
pip install --upgrade streamlit
```

### Charts not displaying
```bash
# Clear cache
rm -rf ~/.streamlit/cache
streamlit run streamlit_llm_app.py --logger.level=debug
```

### Port already in use
```bash
# Use different port
streamlit run streamlit_llm_app.py --server.port 8502
```

## 📧 Support

For questions or issues:
1. Check this README first
2. Review the in-app explanations
3. Consult the code comments
4. Visit Streamlit documentation: https://docs.streamlit.io

## 📄 License

Educational tool - Free to use for learning and teaching purposes.

## 🙏 Acknowledgments

Built for students and educators learning about modern LLM training techniques.

---

**Happy Learning! 🎓**

For the latest version and updates, refer to the accompanying documentation files.
