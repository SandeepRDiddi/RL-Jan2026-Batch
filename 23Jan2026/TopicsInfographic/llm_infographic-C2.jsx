import React, { useState, useEffect } from 'react';
import { ChevronDown, AlertCircle, CheckCircle, Zap, Brain } from 'lucide-react';

export default function LLMTrainingInfographic() {
  const [activeChallenge, setActiveChallenge] = useState(0);
  const [scrollProgress, setScrollProgress] = useState(0);

  useEffect(() => {
    const handleScroll = () => {
      const winScroll = document.documentElement.scrollTop;
      const height = document.documentElement.scrollHeight - document.documentElement.clientHeight;
      setScrollProgress((winScroll / height) * 100);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const challenges = [
    {
      title: "Annotation Quality & Inconsistency",
      problem: "Different experts prioritize different aspects",
      example: "Annotator A: Specific eGFR cutoffs (safety) vs Annotator B: Alternative drugs (flexibility)",
      solutions: ["Inter-Rater Reliability (Fleiss' Kappa ≥ 0.60)", "Multi-Annotator Consensus (2/3 agreement)", "Calibration Sessions with explicit guidelines"]
    },
    {
      title: "Reward Hacking",
      problem: "Reward Model exploited rather than genuinely improved",
      example: "If RM rewards length: Model generates verbose, repetitive text to maximize length",
      solutions: ["Human Validation of PPO-generated outputs", "KL Regularization (higher β = stay closer to SFT)", "Ensemble RMs to reduce single model biases"]
    },
    {
      title: "Distribution Shift",
      problem: "RM trained on SFT data; PPO produces increasingly different outputs",
      example: "Out-of-distribution samples have unreliable RM scores",
      solutions: ["Active Learning on PPO-generated samples", "Iterative Refinement cycles (RM → PPO → RM)", "Continuous retraining on newer distributions"]
    },
    {
      title: "Scalability of Human Preference Data",
      problem: "Expert time is expensive; collecting 50k pairs is costly",
      example: "Pharmacist expertise is limited resource",
      solutions: ["Focus on High-Value Data (diverse, challenging examples)", "Unlabeled Data Ranking with uncertainty sampling", "Crowdsourcing with Expert spot-checks"]
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-950 via-slate-900 to-slate-950 text-white overflow-x-hidden">
      {/* Progress Bar */}
      <div className="fixed top-0 left-0 h-1 bg-gradient-to-r from-cyan-400 via-blue-500 to-purple-600 z-50" style={{ width: `${scrollProgress}%` }}></div>

      {/* Header */}
      <section className="relative min-h-screen flex items-center justify-center px-6 pt-20">
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute top-20 left-10 w-72 h-72 bg-blue-500 rounded-full mix-blend-screen filter blur-3xl opacity-20 animate-pulse"></div>
          <div className="absolute bottom-10 right-10 w-72 h-72 bg-purple-500 rounded-full mix-blend-screen filter blur-3xl opacity-20 animate-pulse" style={{ animationDelay: '1s' }}></div>
        </div>
        
        <div className="relative z-10 text-center max-w-4xl">
          <h1 className="text-6xl font-bold mb-4 bg-clip-text text-transparent bg-gradient-to-r from-cyan-300 via-blue-400 to-purple-400">
            LLM Training Pipeline & Alignment
          </h1>
          <p className="text-2xl text-slate-300 mb-8">Complete Integration, Challenges & Pharmaceutical Applications</p>
          <div className="inline-block px-8 py-4 rounded-xl bg-gradient-to-r from-blue-600/20 to-purple-600/20 border border-blue-400/30 backdrop-blur">
            <p className="text-lg text-slate-100">From Statistical Foundation to Clinical Excellence</p>
          </div>
        </div>
      </section>

      {/* Section 1: The Three-Stage Pipeline */}
      <section className="relative py-24 px-6">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-5xl font-bold mb-16 text-center text-transparent bg-clip-text bg-gradient-to-r from-cyan-300 to-blue-400">
            8. Complete Pipeline Integration
          </h2>
          
          <div className="mb-16">
            <h3 className="text-2xl font-semibold mb-8 text-cyan-300">8.1 Real-World Case: Adverse Event Discovery</h3>
            <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-8 backdrop-blur">
              <div className="space-y-4">
                <p className="text-lg"><span className="text-cyan-400 font-semibold">Researcher Query:</span> "Drug-induced liver injury in patients with chronic kidney disease on statins"</p>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-8">
                  <div className="bg-gradient-to-br from-blue-600/30 to-blue-700/20 border border-blue-500/30 rounded-lg p-6">
                    <h4 className="font-semibold text-blue-300 mb-2">Stage 1: Pretraining</h4>
                    <p className="text-sm text-slate-300">Model learned correlation between CKD + statin + elevated liver enzymes from internet text patterns</p>
                  </div>
                  <div className="bg-gradient-to-br from-purple-600/30 to-purple-700/20 border border-purple-500/30 rounded-lg p-6">
                    <h4 className="font-semibold text-purple-300 mb-2">Stage 2: SFT</h4>
                    <p className="text-sm text-slate-300">Learned to structure queries following FAIR principles (Findable, Accessible, Interoperable, Reusable)</p>
                  </div>
                  <div className="bg-gradient-to-br from-pink-600/30 to-pink-700/20 border border-pink-500/30 rounded-lg p-6">
                    <h4 className="font-semibold text-pink-300 mb-2">Stage 3: RLHF</h4>
                    <p className="text-sm text-slate-300">Ranked search results by expert preference for clinical relevance + data quality</p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Integration Table */}
          <div className="mb-16">
            <h3 className="text-2xl font-semibold mb-8 text-cyan-300">8.2 Integration Table: All Three Stages Combined</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm border-collapse">
                <thead>
                  <tr className="bg-slate-700/50">
                    <th className="border border-slate-600 px-6 py-4 text-left font-semibold text-cyan-300">Stage</th>
                    <th className="border border-slate-600 px-6 py-4 text-left font-semibold text-cyan-300">Objective</th>
                    <th className="border border-slate-600 px-6 py-4 text-left font-semibold text-cyan-300">Key Loss Function</th>
                    <th className="border border-slate-600 px-6 py-4 text-left font-semibold text-cyan-300">Data Size</th>
                    <th className="border border-slate-600 px-6 py-4 text-left font-semibold text-cyan-300">Output Quality</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="hover:bg-slate-700/30 transition">
                    <td className="border border-slate-700 px-6 py-4 font-semibold text-blue-300">Pretraining</td>
                    <td className="border border-slate-700 px-6 py-4">Learn general language + domain patterns</td>
                    <td className="border border-slate-700 px-6 py-4 font-mono text-sm">-Σ log P(w_t|context)</td>
                    <td className="border border-slate-700 px-6 py-4">Billions of tokens</td>
                    <td className="border border-slate-700 px-6 py-4 text-yellow-400">Broad but misaligned</td>
                  </tr>
                  <tr className="hover:bg-slate-700/30 transition">
                    <td className="border border-slate-700 px-6 py-4 font-semibold text-purple-300">SFT</td>
                    <td className="border border-slate-700 px-6 py-4">Teach instruction-following + structure</td>
                    <td className="border border-slate-700 px-6 py-4 font-mono text-sm">-log π_sft(expert_resp)</td>
                    <td className="border border-slate-700 px-6 py-4">10k-100k pairs</td>
                    <td className="border border-slate-700 px-6 py-4 text-yellow-400">Good but capped</td>
                  </tr>
                  <tr className="hover:bg-slate-700/30 transition">
                    <td className="border border-slate-700 px-6 py-4 font-semibold text-pink-300">RLHF</td>
                    <td className="border border-slate-700 px-6 py-4">Align with expert preferences</td>
                    <td className="border border-slate-700 px-6 py-4 font-mono text-sm">max reward - βKL</td>
                    <td className="border border-slate-700 px-6 py-4">5k-50k prefs</td>
                    <td className="border border-slate-700 px-6 py-4 text-green-400">Excellent & aligned</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </section>

      {/* Section 2: Challenges */}
      <section className="relative py-24 px-6 bg-slate-800/30">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-5xl font-bold mb-16 text-center text-transparent bg-clip-text bg-gradient-to-r from-red-300 to-orange-400">
            9. Challenges in LLM Training & Solutions
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-12">
            {challenges.map((challenge, idx) => (
              <button
                key={idx}
                onClick={() => setActiveChallenge(idx)}
                className={`p-4 rounded-lg border-2 transition-all ${
                  activeChallenge === idx
                    ? 'border-red-400 bg-red-600/20 text-red-200'
                    : 'border-slate-600 bg-slate-700/20 text-slate-300 hover:border-slate-500'
                }`}
              >
                <AlertCircle className="inline mr-2 w-4 h-4" />
                {challenge.title.split(' &')[0]}
              </button>
            ))}
          </div>

          <div className="bg-gradient-to-br from-slate-800/60 to-slate-900/60 border border-red-500/30 rounded-2xl p-12 backdrop-blur-xl">
            <div className="animate-fade-in">
              <h3 className="text-3xl font-bold mb-6 text-red-300">{challenges[activeChallenge].title}</h3>
              
              <div className="mb-8 p-6 bg-red-900/20 border-l-4 border-red-500 rounded">
                <p className="font-semibold text-red-200 mb-2">Problem:</p>
                <p className="text-slate-200">{challenges[activeChallenge].problem}</p>
              </div>

              <div className="mb-8 p-6 bg-orange-900/20 border-l-4 border-orange-500 rounded">
                <p className="font-semibold text-orange-200 mb-2">Example:</p>
                <p className="text-slate-200">{challenges[activeChallenge].example}</p>
              </div>

              <div>
                <p className="font-semibold text-green-300 mb-4">Solutions:</p>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {challenges[activeChallenge].solutions.map((solution, idx) => (
                    <div key={idx} className="flex items-start gap-3 p-4 bg-green-900/20 border border-green-600/30 rounded-lg">
                      <CheckCircle className="w-5 h-5 text-green-400 flex-shrink-0 mt-1" />
                      <p className="text-slate-200 text-sm">{solution}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Section 3: Medical Example */}
      <section className="relative py-24 px-6">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-5xl font-bold mb-16 text-center text-transparent bg-clip-text bg-gradient-to-r from-emerald-300 to-teal-400">
            10. Why All Three Stages Matter: Medical Example
          </h2>

          <div className="mb-12">
            <div className="bg-slate-800/50 border border-emerald-600/30 rounded-xl p-8 mb-12">
              <h3 className="text-2xl font-semibold mb-4 text-emerald-300">Case: Adverse Drug Interaction Detection</h3>
              <p className="text-lg text-slate-100"><span className="font-semibold">Query:</span> "Patient on warfarin + ibuprofen. Risk assessment?"</p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {[
                {
                  stage: "Pretraining",
                  output: '"...warfarin ibuprofen bleeding risk NSAIDs platelets..." (word salad)',
                  correct: "Learned associations between keywords",
                  gap: "No structure. No clear answer. Rambling.",
                  color: "from-blue-600/30 to-blue-700/20"
                },
                {
                  stage: "SFT",
                  output: '"Warfarin + NSAID = increased bleeding risk. CONTRAINDICATED. Use acetaminophen instead."',
                  correct: "Clear structure. Safe answer. Follows format.",
                  gap: "Limited to training data. No mechanism to improve.",
                  color: "from-purple-600/30 to-purple-700/20"
                },
                {
                  stage: "RLHF",
                  output: '"Contraindicated. NSAIDs inhibit platelet aggregation + enhance warfarin anticoagulation → 2-3x bleeding risk. ALTERNATIVES: Acetaminophen, topical diclofenac. Monitor INR if absolutely necessary."',
                  correct: "Nuanced. Mechanism explained. Specific alternatives.",
                  gap: "None—exceeds expectations",
                  color: "from-emerald-600/30 to-emerald-700/20"
                }
              ].map((item, idx) => (
                <div key={idx} className={`bg-gradient-to-br ${item.color} border border-slate-700 rounded-xl p-6`}>
                  <h4 className="text-lg font-bold mb-4 text-cyan-300">{item.stage}</h4>
                  <div className="space-y-4 text-sm">
                    <div>
                      <p className="text-slate-400 font-semibold mb-1">Output:</p>
                      <p className="text-slate-200 italic">"{item.output}"</p>
                    </div>
                    <div>
                      <p className="text-emerald-300 font-semibold mb-1">✓ What It Got Right:</p>
                      <p className="text-slate-200">{item.correct}</p>
                    </div>
                    <div>
                      <p className="text-orange-300 font-semibold mb-1">⚠ Critical Gap:</p>
                      <p className="text-slate-200">{item.gap}</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Section 4: Mathematical Summary */}
      <section className="relative py-24 px-6 bg-slate-800/30">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-5xl font-bold mb-16 text-center text-transparent bg-clip-text bg-gradient-to-r from-cyan-300 to-blue-400">
            11. Mathematical Summary & Key Equations
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {[
              {
                title: "11.1 Pretraining",
                equation: "L_pretrain = -Σ log P_θ(w_t | w_<t)",
                description: "Cross-entropy loss. Gradient descent minimizes; equivalent to maximizing data likelihood."
              },
              {
                title: "11.2 SFT",
                equation: "L_SFT = -Σ log π_θ(y_expert | x)",
                description: "Behavioral cloning. Force model to match expert demonstrations token-by-token."
              },
              {
                title: "11.3 RM Training",
                equation: "L_RM = -E[log σ(r_RM(x,y_w) - r_RM(x,y_l))]",
                description: "Bradley-Terry loss. Preference logit = sigmoid of reward difference."
              },
              {
                title: "11.4 PPO",
                equation: "L_PPO = E[min(r_t*A_t, clip(r_t, 1-ε, 1+ε)*A_t)] - β*KL(π||π_ref)",
                description: "Clipped objective prevents policy from changing too drastically. KL term ensures stability."
              }
            ].map((item, idx) => (
              <div key={idx} className="bg-gradient-to-br from-slate-800 to-slate-900 border border-slate-700 rounded-xl p-8">
                <h3 className="text-lg font-bold text-cyan-300 mb-4">{item.title}</h3>
                <div className="bg-slate-950 rounded-lg p-6 mb-4 border border-slate-700">
                  <code className="text-purple-300 text-lg font-mono">{item.equation}</code>
                </div>
                <p className="text-slate-300">{item.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Section 5: Alignment in Pharma */}
      <section className="relative py-24 px-6">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-5xl font-bold mb-16 text-center text-transparent bg-clip-text bg-gradient-to-r from-emerald-300 to-teal-400">
            12. Alignment in Pharmaceutical AI
          </h2>

          <div className="mb-16">
            <h3 className="text-2xl font-semibold mb-8 text-emerald-300">12.1 Why Alignment Is Critical in Medicine</h3>
            <div className="bg-gradient-to-r from-red-900/20 to-red-800/20 border border-red-500/30 rounded-xl p-8 mb-8">
              <p className="text-lg text-slate-100">
                <span className="text-red-300 font-bold">Unlike general-purpose AI,</span> pharmaceutical AI has life-or-death consequences. 
                <span className="text-red-300 font-bold ml-2">Misalignment = patient harm.</span>
              </p>
            </div>

            <h4 className="text-xl font-semibold mb-6 text-teal-300">Three Alignment Pillars for Pharmaceutical AI</h4>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {[
                {
                  pillar: "Accuracy",
                  icon: "✓",
                  description: "Medical information must be correct; outdated info causes harm",
                  metric: "≥95%"
                },
                {
                  pillar: "Safety",
                  icon: "⚠",
                  description: "Proactively warn about contraindications, dangers, edge cases",
                  metric: "100% (no false negatives)"
                },
                {
                  pillar: "Humility",
                  icon: "?",
                  description: "Admit uncertainty; never overstate confidence",
                  metric: "Zero absolute claims on unsettled questions"
                }
              ].map((item, idx) => (
                <div key={idx} className="bg-slate-800/50 border border-emerald-600/30 rounded-xl p-6 text-center">
                  <div className="text-4xl mb-4">{item.icon}</div>
                  <h5 className="text-xl font-bold text-emerald-300 mb-3">{item.pillar}</h5>
                  <p className="text-slate-300 mb-4 text-sm">{item.description}</p>
                  <div className="bg-emerald-900/30 rounded-lg px-4 py-2 text-emerald-200 font-semibold text-sm">{item.metric}</div>
                </div>
              ))}
            </div>
          </div>

          <div>
            <h3 className="text-2xl font-semibold mb-8 text-emerald-300">12.2 Alignment Evaluation Metrics</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm border-collapse">
                <thead>
                  <tr className="bg-slate-700/50">
                    <th className="border border-slate-600 px-6 py-4 text-left font-semibold text-emerald-300">Metric</th>
                    <th className="border border-slate-600 px-6 py-4 text-left font-semibold text-emerald-300">Definition</th>
                    <th className="border border-slate-600 px-6 py-4 text-left font-semibold text-emerald-300">How to Measure</th>
                    <th className="border border-slate-600 px-6 py-4 text-left font-semibold text-emerald-300">Target</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="hover:bg-slate-700/30 transition">
                    <td className="border border-slate-700 px-6 py-4 font-semibold text-emerald-300">Accuracy</td>
                    <td className="border border-slate-700 px-6 py-4">% correct info vs reference standard</td>
                    <td className="border border-slate-700 px-6 py-4">Expert review of 100 responses</td>
                    <td className="border border-slate-700 px-6 py-4 text-green-400">≥95%</td>
                  </tr>
                  <tr className="hover:bg-slate-700/30 transition">
                    <td className="border border-slate-700 px-6 py-4 font-semibold text-emerald-300">Safety</td>
                    <td className="border border-slate-700 px-6 py-4">% identifies contraindications</td>
                    <td className="border border-slate-700 px-6 py-4">100 dangerous drug pairs; count correct warnings</td>
                    <td className="border border-slate-700 px-6 py-4 text-green-400">100%</td>
                  </tr>
                  <tr className="hover:bg-slate-700/30 transition">
                    <td className="border border-slate-700 px-6 py-4 font-semibold text-emerald-300">Humility</td>
                    <td className="border border-slate-700 px-6 py-4">% admits uncertainty appropriately</td>
                    <td className="border border-slate-700 px-6 py-4">Review for overconfident claims</td>
                    <td className="border border-slate-700 px-6 py-4 text-green-400">Zero absolute claims</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </section>

      {/* Section 6: Key Takeaways */}
      <section className="relative py-24 px-6 bg-gradient-to-b from-slate-800/30 to-slate-900/50">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-5xl font-bold mb-16 text-center text-transparent bg-clip-text bg-gradient-to-r from-yellow-300 to-orange-400">
            13. Key Takeaways
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-16">
            <div className="bg-gradient-to-br from-blue-600/20 to-blue-700/10 border border-blue-500/30 rounded-xl p-8">
              <h3 className="text-2xl font-bold text-blue-300 mb-6">Why Three Stages Are Essential</h3>
              <ul className="space-y-4">
                <li className="flex gap-3">
                  <span className="text-blue-400 font-bold">1.</span>
                  <span><span className="text-blue-300 font-semibold">Pretraining:</span> Statistical foundation. Learn from scale. But no alignment or understanding.</span>
                </li>
                <li className="flex gap-3">
                  <span className="text-blue-400 font-bold">2.</span>
                  <span><span className="text-blue-300 font-semibold">SFT:</span> Behavioral shaping. Teach structure + format. But limited by dataset size and quality.</span>
                </li>
                <li className="flex gap-3">
                  <span className="text-blue-400 font-bold">3.</span>
                  <span><span className="text-blue-300 font-semibold">RLHF:</span> Nuanced alignment. Learn from preferences. Improve beyond expert demonstrations.</span>
                </li>
              </ul>
            </div>

            <div className="bg-gradient-to-br from-emerald-600/20 to-emerald-700/10 border border-emerald-500/30 rounded-xl p-8">
              <h3 className="text-2xl font-bold text-emerald-300 mb-6">For Pharmaceutical Applications</h3>
              <ul className="space-y-4">
                <li className="flex gap-3">
                  <Zap className="w-5 h-5 text-emerald-400 flex-shrink-0 mt-0.5" />
                  <span><span className="text-emerald-300 font-semibold">Safety:</span> RLHF learns which warnings matter most to domain experts</span>
                </li>
                <li className="flex gap-3">
                  <Zap className="w-5 h-5 text-emerald-400 flex-shrink-0 mt-0.5" />
                  <span><span className="text-emerald-300 font-semibold">Accuracy:</span> Pretraining + SFT + RLHF layers ensure correct medical information</span>
                </li>
                <li className="flex gap-3">
                  <Zap className="w-5 h-5 text-emerald-400 flex-shrink-0 mt-0.5" />
                  <span><span className="text-emerald-300 font-semibold">Productivity:</span> Aligned model focuses on relevant data discovery, saving researcher time</span>
                </li>
              </ul>
            </div>
          </div>

          <div className="bg-gradient-to-r from-purple-900/30 to-blue-900/30 border border-purple-500/30 rounded-2xl p-12">
            <h3 className="text-2xl font-bold text-purple-300 mb-8 flex items-center gap-3">
              <Brain className="w-8 h-8" />
              The Bottom Line
            </h3>
            <p className="text-lg text-slate-100 leading-relaxed">
              Each stage of the training pipeline serves a distinct purpose. <span className="text-blue-300 font-semibold">Pretraining</span> builds statistical foundations from massive data. 
              <span className="text-purple-300 font-semibold"> SFT</span> teaches structured behavior. <span className="text-emerald-300 font-semibold">RLHF</span> aligns the model with human values and expert preferences. 
              In pharmaceutical applications, this three-stage approach is essential to ensure safety, accuracy, and appropriately calibrated confidence—ultimately serving clinicians and protecting patients.
            </p>
          </div>
        </div>
      </section>

      {/* Footer */}
      <section className="py-16 px-6 border-t border-slate-700">
        <div className="max-w-6xl mx-auto text-center text-slate-400">
          <p className="mb-4">Further Reading & References</p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
            <p>Christiano et al. (2017): Deep Reinforcement Learning from Human Preferences</p>
            <p>Ouyang et al. (2022): Training language models to follow instructions with human feedback</p>
            <p>Schulman et al. (2017): Proximal Policy Optimization Algorithms</p>
            <p>Bradley & Terry (1952): Rank Analysis of Incomplete Block Designs</p>
          </div>
        </div>
      </section>

      <style jsx>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Mono:wght@400;700&display=swap');
        
        * {
          font-family: 'Inter', sans-serif;
        }
        
        code {
          font-family: 'Space Mono', monospace;
        }
        
        @keyframes fade-in {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        
        .animate-fade-in {
          animation: fade-in 0.5s ease-out;
        }
      `}</style>
    </div>
  );
}
