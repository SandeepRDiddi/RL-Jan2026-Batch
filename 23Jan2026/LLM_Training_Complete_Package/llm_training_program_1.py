"""
LLM Training Pipeline - Program 1: Pretraining, SFT, and Reward Model Training
Real-world pharmaceutical examples with mathematical demonstrations
"""

import numpy as np
from typing import List, Dict
import matplotlib.pyplot as plt
from scipy.special import expit  # Logistic function
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("LLM TRAINING PIPELINE - PROGRAM 1")
print("Stages: Pretraining → SFT → Reward Model Training")
print("="*80)

# ============================================================================
# PART 1: PRETRAINING SIMULATOR
# ============================================================================

class PretrainingSimulator:
    """Pretraining: Next-token prediction on medical sequences"""
    
    def __init__(self):
        self.sequences = [
            ("Patient fever body aches fatigue", "viral"),
            ("Chest X-ray infiltrate lower lobe", "pneumonia"),
            ("Blood glucose 126 mg/dL", "diabetes"),
            ("Rash fever eosinophilia", "DRESS"),
            ("Warfarin NSAIDs", "bleeding"),
            ("ACE inhibitor potassium diuretic", "hyperkalemia"),
            ("eGFR 28 metformin", "lactic_acidosis"),
            ("Codeine CYP2D6 poor metabolizer", "no_effect"),
            ("Statin muscle pain elevated CK", "myopathy"),
            ("Penicillin anaphylaxis beta-lactam", "danger"),
        ]
        self.logits = np.random.randn(100, 100) * 0.1
        self.learning_rate = 0.01
        
    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)
    
    def train_epoch(self):
        epoch_loss = 0
        for context, target in self.sequences:
            ctx_vec = np.random.randn(100) * 0.1  # Simplified context encoding
            probs = self.softmax(ctx_vec.reshape(1, -1))[0]
            target_idx = hash(target) % 100
            loss = -np.log(probs[target_idx] + 1e-10)
            epoch_loss += loss
            # Simple update
            gradient = probs.copy()
            gradient[target_idx] -= 1
            self.logits[hash(context) % 100] -= self.learning_rate * gradient
        
        return epoch_loss / len(self.sequences)
    
    def display_examples(self):
        print("\n" + "="*80)
        print("STAGE 1: PRETRAINING - NEXT TOKEN PREDICTION")
        print("="*80)
        print("\nPretraining learns: Context → Next Token (statistical patterns)")
        print("Mathematical: Loss = -Σ log P(w_t | w_<t)\n")
        print("Example 1: Fever + Aches + Fatigue → 'Viral'")
        print("  Model learns: These symptoms often precede viral diagnosis")
        print("Example 2: eGFR 28 + Metformin → 'Lactic Acidosis'")
        print("  Model learns: Low GFR + metformin = danger pattern")
        print("\nLimitation: No understanding of SAFETY or human values")

# ============================================================================
# PART 2: SFT SIMULATOR
# ============================================================================

class SFTSimulator:
    """SFT: Learn from expert demonstrations"""
    
    def __init__(self):
        self.sft_pairs = [
            ("How to adjust for renal impairment?", 
             "eGFR >60: 100%, eGFR 30-59: 75%, eGFR <30: 50%. Monitor always."),
            ("Metformin in cirrhosis?",
             "Contraindicated. Lactic acidosis risk. Use insulin, GLP-1, SGLT2i."),
            ("Warfarin + NSAIDs?",
             "Avoid. NSAIDs inhibit platelets + bleeding. Use acetaminophen."),
            ("Penicillin allergy + cephalosporin?",
             "Depends on type. Non-IgE: maybe safe. IgE: avoid beta-lactams."),
            ("CYP2D6 poor metabolizer + codeine?",
             "Ineffective. No morphine conversion. Use morphine, oxycodone, tramadol."),
        ]
        self.W = np.random.randn(10, 5) * 0.01
        self.learning_rate = 0.001
        self.losses = []
        
    def forward(self, x):
        return np.dot(x, self.W)
    
    def train(self, epochs=10):
        print("\n" + "="*80)
        print("STAGE 2: SUPERVISED FINE-TUNING (SFT)")
        print("="*80)
        print("\nSFT learns: Instruction → Expert Response")
        print("Mathematical: L_SFT = -E[Σ log π(expert_token | context)]\n")
        
        for epoch in range(epochs):
            epoch_loss = 0
            for instruction, response in self.sft_pairs:
                # Simple feature encoding
                inst_vec = np.array([hash(instruction) % 10 / 10 for _ in range(10)])
                resp_target = np.array([hash(response) % 5 / 5 for _ in range(5)])
                
                output = self.forward(inst_vec)
                loss = np.mean((output - resp_target) ** 2)
                epoch_loss += loss
                
                # Gradient descent
                gradient = 2 * (output - resp_target)
                self.W -= self.learning_rate * np.outer(inst_vec, gradient)
            
            avg_loss = epoch_loss / len(self.sft_pairs)
            self.losses.append(avg_loss)
            
            if (epoch + 1) % 2 == 0:
                print(f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}")
    
    def display_examples(self):
        print("\nSFT Example: Expert Demonstrations")
        print("-" * 60)
        print("1. Instruction: How to adjust for renal impairment?")
        print("   Expert: eGFR >60: normal, 30-59: 75%, <30: 50%")
        print("   Learns: Specific, safe, evidence-based response\n")
        print("2. Instruction: Warfarin + NSAIDs?")
        print("   Expert: Avoid. 2-3x bleeding risk. Use acetaminophen.")
        print("   Learns: Safety-first recommendation with mechanism")

# ============================================================================
# PART 3: REWARD MODEL TRAINER
# ============================================================================

class RewardModelTrainer:
    """Reward Model: Learn human preferences via Bradley-Terry loss"""
    
    def __init__(self):
        # 20 pharmaceutical preference pairs
        self.preferences = [
            {
                "prompt": "Renal dosing adjustment?",
                "pref": "eGFR >60: 100%, 30-59: 75%, <30: 50% with monitoring",
                "dispref": "Just reduce dose if kidneys are low",
            },
            {
                "prompt": "Warfarin + NSAIDs safe?",
                "pref": "No. 2-3x bleeding risk. Use acetaminophen instead.",
                "dispref": "One NSAID shouldn't matter.",
            },
            {
                "prompt": "Metformin eGFR 28?",
                "pref": "Contraindicated. Lactic acidosis. Use insulin, GLP-1, SGLT2i.",
                "dispref": "Fine to use. Monitor kidneys.",
            },
            {
                "prompt": "ACE inhibitor + K-sparing diuretic?",
                "pref": "Both retain K+. Combined effect: severe hyperkalemia. Monitor closely.",
                "dispref": "Both are kidney-protective. No special concern.",
            },
            {
                "prompt": "CYP2D6 poor metabolizer codeine?",
                "pref": "Ineffective. No morphine conversion. Use morphine/oxycodone directly.",
                "dispref": "Should still work. Maybe try higher dose.",
            },
            {
                "prompt": "Pregnancy + ACE inhibitor?",
                "pref": "Contraindicated 2nd/3rd trimester. Fetal renal dysgenesis. Switch to methyldopa.",
                "dispref": "Category C so probably fine.",
            },
            {
                "prompt": "Grapefruit + statin?",
                "pref": "Yes. Inhibits CYP3A4. 10-16x statin levels. Muscle toxicity risk.",
                "dispref": "Minor interaction. One glass is fine.",
            },
            {
                "prompt": "Lithium toxicity?",
                "pref": "Therapeutic 0.6-1.2. Toxicity >2: tremor, confusion. Hemodialysis if severe.",
                "dispref": "Just reduce dose if patient feels bad.",
            },
            {
                "prompt": "Tricyclics in elderly?",
                "pref": "Caution: anticholinergic (confusion, falls). Consider SSRIs.",
                "dispref": "Safe. No special precautions needed.",
            },
            {
                "prompt": "Macrolides + QT prolongation?",
                "pref": "Yes. Block cardiac K+ channels. Risky with other QT drugs.",
                "dispref": "Unlikely to be major problem.",
            },
            {
                "prompt": "SSRI discontinuation?",
                "pref": "Taper over 4 weeks. Dizziness, nausea, electric shocks expected.",
                "dispref": "Can stop immediately. Not serious.",
            },
            {
                "prompt": "NSAIDs + renal disease?",
                "pref": "Yes. Reduce perfusion. eGFR <30: avoid. eGFR 30-60: monitor.",
                "dispref": "NSAIDs protect kidneys. Safe with renal issues.",
            },
            {
                "prompt": "Stevens-Johnson Syndrome?",
                "pref": "URGENT: Discontinue. ICU/burn center. Report to FDA MedWatch.",
                "dispref": "Continue with antihistamine. Will resolve.",
            },
            {
                "prompt": "Theophylline range?",
                "pref": "10-20 μg/mL therapeutic. >20 toxicity: seizures, arrhythmias.",
                "dispref": "No special monitoring needed.",
            },
            {
                "prompt": "Penicillin anaphylaxis + cephalosporin?",
                "pref": "Avoid all beta-lactams. Use fluoroquinolone or macrolide.",
                "dispref": "3rd gen cephalosporin probably okay.",
            },
            {
                "prompt": "Statin myopathy CK >1000?",
                "pref": "Discontinue. Check rhabdomyolysis. Monitor electrolytes.",
                "dispref": "Mild elevation normal. Continue therapy.",
            },
            {
                "prompt": "Blood glucose 126 fasting?",
                "pref": "Repeat fasting or HbA1c. Discuss lifestyle, medication options.",
                "dispref": "Definitely diabetes. Start metformin now.",
            },
            {
                "prompt": "St. John's Wort + warfarin?",
                "pref": "Yes. Induces CYP3A4/2C9. Decreases warfarin. Avoid or monitor INR.",
                "dispref": "Natural so safe. No interaction.",
            },
            {
                "prompt": "Hepatic impairment dosing?",
                "pref": "Child-Pugh: Mild 50% reduction, Moderate 25%, Severe contraindicated.",
                "dispref": "Normal dose okay. Liver is resilient.",
            },
            {
                "prompt": "Polypharmacy >10 drugs elderly?",
                "pref": "High risk. Deprescribe lowest-benefit drugs. Taper TCAs, benzos.",
                "dispref": "Keep all meds. Stopping increases disease risk.",
            }
        ]
        
        self.rm_w = np.random.randn(10) * 0.1
        self.rm_b = 0.0
        self.learning_rate = 0.01
        
        self.history = {
            "epoch": [], "loss": [], "margin": [], "accuracy": [],
            "pref_score": [], "dispref_score": []
        }
    
    def text_to_features(self, text: str) -> np.ndarray:
        """Convert text to feature vector"""
        features = np.zeros(10)
        for i, char in enumerate(text):
            features[ord(char) % 10] += 1
        return features / (np.sum(features) + 1e-10)
    
    def reward(self, text: str) -> float:
        features = self.text_to_features(text)
        return np.dot(features, self.rm_w) + self.rm_b
    
    def sigmoid(self, x):
        return expit(np.clip(x, -500, 500))
    
    def train_epoch(self):
        epoch_loss = 0
        margins = []
        correct = 0
        pref_scores = []
        dispref_scores = []
        
        for pref in self.preferences:
            r_pref = self.reward(pref["pref"])
            r_dispref = self.reward(pref["dispref"])
            
            pref_scores.append(r_pref)
            dispref_scores.append(r_dispref)
            
            margin = r_pref - r_dispref
            margins.append(margin)
            
            prob = self.sigmoid(margin)
            loss = -np.log(prob + 1e-10)
            epoch_loss += loss
            
            if margin > 0:
                correct += 1
            
            # Update weights
            gradient = prob - 1.0
            f_pref = self.text_to_features(pref["pref"])
            f_dispref = self.text_to_features(pref["dispref"])
            
            self.rm_w += self.learning_rate * gradient * (f_pref - f_dispref)
            self.rm_b += self.learning_rate * gradient
        
        return {
            "loss": epoch_loss / len(self.preferences),
            "margin": np.mean(margins),
            "accuracy": correct / len(self.preferences),
            "pref_score": np.mean(pref_scores),
            "dispref_score": np.mean(dispref_scores)
        }
    
    def train(self, epochs=100):
        print("\n" + "="*80)
        print("STAGE 3: REWARD MODEL TRAINING")
        print("="*80)
        print("\nRM learns: Preference Ranking (safer responses score higher)")
        print("Mathematical: Loss = -log σ(r_pref - r_dispref)\n")
        print("Training on 20 pharmaceutical preference pairs...\n")
        
        for epoch in range(epochs):
            metrics = self.train_epoch()
            
            self.history["epoch"].append(epoch + 1)
            self.history["loss"].append(metrics["loss"])
            self.history["margin"].append(metrics["margin"])
            self.history["accuracy"].append(metrics["accuracy"])
            self.history["pref_score"].append(metrics["pref_score"])
            self.history["dispref_score"].append(metrics["dispref_score"])
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1:3d}: Loss={metrics['loss']:.4f}, "
                      f"Margin={metrics['margin']:+.3f}, Acc={metrics['accuracy']:.0%}, "
                      f"r(pref)={metrics['pref_score']:.2f}, r(dispref)={metrics['dispref_score']:.2f}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    # Stage 1: Pretraining
    pretrain = PretrainingSimulator()
    pretrain.display_examples()
    
    print("\nTraining pretraining model...")
    for epoch in range(5):
        loss = pretrain.train_epoch()
        print(f"  Epoch {epoch+1}/5: Loss = {loss:.4f}")
    
    # Stage 2: SFT
    sft = SFTSimulator()
    sft.display_examples()
    sft.train(epochs=10)
    
    # Stage 3: Reward Model
    rm = RewardModelTrainer()
    rm.train(epochs=100)
    
    # Visualization
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss
    axes[0, 0].plot(rm.history["epoch"], rm.history["loss"], 'b-', linewidth=2)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Bradley-Terry Loss")
    axes[0, 0].set_title("RM: Loss Over Training")
    axes[0, 0].grid(True, alpha=0.3)
    
    # Margin
    axes[0, 1].plot(rm.history["epoch"], rm.history["margin"], 'g-', linewidth=2)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Margin")
    axes[0, 1].set_title("RM: Margin (Pref - Dispref)")
    axes[0, 1].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1, 0].plot(rm.history["epoch"], [a*100 for a in rm.history["accuracy"]], 'r-', linewidth=2)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy (%)")
    axes[1, 0].set_title("RM: Preference Prediction Accuracy")
    axes[1, 0].set_ylim([0, 105])
    axes[1, 0].grid(True, alpha=0.3)
    
    # Scores
    axes[1, 1].plot(rm.history["epoch"], rm.history["pref_score"], 'b-', label="Preferred", linewidth=2)
    axes[1, 1].plot(rm.history["epoch"], rm.history["dispref_score"], 'r-', label="Dispreferred", linewidth=2)
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Score")
    axes[1, 1].set_title("RM: Learning to Separate Responses")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("llm_program_1_results.png", dpi=150, bbox_inches='tight')
    print("✓ Saved: llm_program_1_results.png\n")
    
    # Summary
    print("="*80)
    print("PROGRAM 1 SUMMARY")
    print("="*80)
    print("\n✓ PRETRAINING:")
    print("  - 10 pharmaceutical sequences")
    print("  - Learned next-token patterns")
    print("  - Problem: No value alignment\n")
    
    print("✓ SFT:")
    print("  - 5 expert (instruction, response) pairs")
    print("  - Learned instruction-following")
    print("  - Problem: Capped by expert quality\n")
    
    print("✓ REWARD MODEL:")
    print("  - 20 pharmaceutical preferences")
    print("  - Bradley-Terry loss training")
    print(f"  - Final Loss: {rm.history['loss'][-1]:.4f}")
    print(f"  - Final Accuracy: {rm.history['accuracy'][-1]:.1%}")
    print(f"  - Final Margin: {rm.history['margin'][-1]:+.3f}\n")
    
    print("="*80)
    print("✓ PROGRAM 1 COMPLETE")
    print("→ Next: Run Program 2 for Policy Optimization (PPO) and Integration")
    print("="*80)

if __name__ == "__main__":
    main()
