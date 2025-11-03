#!/usr/bin/env python3
"""
Create Report.pdf from markdown content
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib import colors
from reportlab.pdfgen import canvas
import datetime

def create_report_pdf():
    """Create comprehensive PDF report"""
    
    # Create PDF
    pdf_file = "Report.pdf"
    doc = SimpleDocTemplate(pdf_file, pagesize=letter,
                           rightMargin=72, leftMargin=72,
                           topMargin=72, bottomMargin=18)
    
    # Container for the 'Flowable' objects
    elements = []
    
    # Define styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1a5490'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#2c5aa0'),
        spaceAfter=20,
        alignment=TA_CENTER
    )
    
    heading1_style = ParagraphStyle(
        'CustomHeading1',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#1a5490'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    heading2_style = ParagraphStyle(
        'CustomHeading2',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#2c5aa0'),
        spaceAfter=10,
        spaceBefore=10,
        fontName='Helvetica-Bold'
    )
    
    heading3_style = ParagraphStyle(
        'CustomHeading3',
        parent=styles['Heading3'],
        fontSize=12,
        textColor=colors.black,
        spaceAfter=8,
        spaceBefore=8,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=10,
        alignment=TA_JUSTIFY,
        spaceAfter=6
    )
    
    code_style = ParagraphStyle(
        'Code',
        parent=styles['Code'],
        fontSize=9,
        fontName='Courier',
        textColor=colors.HexColor('#333333'),
        backColor=colors.HexColor('#f5f5f5'),
        leftIndent=20,
        spaceAfter=10
    )
    
    # Title Page
    elements.append(Spacer(1, 2*inch))
    elements.append(Paragraph("Hangman ML Agent", title_style))
    elements.append(Paragraph("Analysis Report", subtitle_style))
    elements.append(Spacer(1, 0.5*inch))
    
    elements.append(Paragraph("<b>Course:</b> UE23CS352A - Machine Learning", body_style))
    elements.append(Paragraph("<b>Project:</b> Hangman Game Solver using HMM and Reinforcement Learning", body_style))
    elements.append(Spacer(1, 0.5*inch))
    
    elements.append(Paragraph("<b>Team Members:</b>", body_style))
    elements.append(Paragraph("PES1UG23AM354", body_style))
    elements.append(Paragraph("PES1UG23AM359", body_style))
    elements.append(Paragraph("PES1UG23AM350", body_style))
    elements.append(Paragraph("PES1UG23AM345", body_style))
    elements.append(Spacer(1, 0.5*inch))
    
    elements.append(Paragraph(f"<b>Submission Date:</b> November 3, 2025", body_style))
    elements.append(Spacer(1, 1*inch))
    elements.append(Paragraph("PES University<br/>Department of Computer Science and Engineering", 
                             ParagraphStyle('Center', parent=body_style, alignment=TA_CENTER)))
    
    elements.append(PageBreak())
    
    # Executive Summary
    elements.append(Paragraph("Executive Summary", heading1_style))
    elements.append(Paragraph(
        "This report presents a comprehensive analysis of our Hangman game-solving agent that combines "
        "Hidden Markov Models (HMM) with Reinforcement Learning (RL). The agent was trained on a corpus "
        "of approximately 50,000 words and evaluated on 1,996 test words with 6 lives per game.",
        body_style
    ))
    elements.append(Spacer(1, 0.2*inch))
    
    # Performance Metrics Table
    elements.append(Paragraph("Final Performance Metrics", heading2_style))
    
    data = [
        ['Metric', 'Value'],
        ['Total Test Words', '1,996'],
        ['Games Won', '587'],
        ['Games Lost', '1,409'],
        ['Success Rate', '29.41%'],
        ['Total Wrong Guesses', '9,182'],
        ['Avg. Wrong Guesses/Game', '4.60'],
        ['Total Repeated Guesses', '0'],
        ['Avg. Repeated Guesses/Game', '0.00'],
        ['Final Score', '-45,321.82'],
    ]
    
    table = Table(data, colWidths=[3*inch, 2*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a5490')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 4), (-1, 4), 'Helvetica-Bold'),
        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
    ]))
    
    elements.append(table)
    elements.append(Spacer(1, 0.2*inch))
    
    elements.append(Paragraph("Score Calculation:", heading3_style))
    elements.append(Paragraph(
        "<font name='Courier' size=9>"
        "Final Score = (Success Rate × 2000) - (Wrong Guesses × 5) - (Repeated × 2)<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= (0.2941 × 2000) - (9182 × 5) - (0 × 2)<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= 588.2 - 45,910 - 0<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= -45,321.82"
        "</font>",
        body_style
    ))
    
    elements.append(PageBreak())
    
    # Section 1: Key Observations
    elements.append(Paragraph("1. Key Observations", heading1_style))
    
    elements.append(Paragraph("1.1 Most Challenging Parts", heading2_style))
    
    # State Space Explosion
    elements.append(Paragraph("State Space Explosion", heading3_style))
    elements.append(Paragraph(
        "The biggest challenge was managing the exponential state space. For a word of length n, "
        "there are 2^n possible mask patterns. Combined with 2^26 ≈ 67 million possible combinations "
        "of guessed letters and 7 life values, the total state space is practically infinite.",
        body_style
    ))
    elements.append(Paragraph(
        "<b>Impact:</b> The Q-table became extremely sparse. Most test states were never seen during "
        "training, forcing heavy reliance on HMM guidance rather than learned Q-values.",
        body_style
    ))
    elements.append(Spacer(1, 0.1*inch))
    
    # Reward Engineering
    elements.append(Paragraph("Reward Engineering", heading3_style))
    elements.append(Paragraph(
        "Balancing multiple objectives proved critical: maximizing win rate, minimizing wrong guesses, "
        "and avoiding repeated guesses. Our final reward structure:",
        body_style
    ))
    
    reward_data = [
        ['Event', 'Reward', 'Rationale'],
        ['Correct Guess', '+0.5', 'Immediate positive feedback'],
        ['Wrong Guess', '-1.0', 'Moderate penalty, allows exploration'],
        ['Repeated Guess', '-0.5', 'Discourage inefficiency'],
        ['Win Game', '+10.0', 'Strong completion incentive'],
        ['Lose Game', '-5.0', 'Penalty for failure'],
    ]
    
    reward_table = Table(reward_data, colWidths=[1.5*inch, 1*inch, 2.5*inch])
    reward_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    
    elements.append(reward_table)
    elements.append(Spacer(1, 0.1*inch))
    
    # HMM Probability Estimation
    elements.append(Paragraph("HMM Probability Estimation", heading3_style))
    elements.append(Paragraph(
        "Accurately predicting letters from partial words was difficult due to context dependency "
        "(e.g., Q→U 99% of time), position sensitivity (different frequencies at start/middle/end), "
        "and smoothing trade-offs (α=0.01 balanced unseen bigrams with common patterns).",
        body_style
    ))
    elements.append(Spacer(1, 0.1*inch))
    
    # Exploration vs Exploitation
    elements.append(Paragraph("Exploration vs. Exploitation Trade-off", heading3_style))
    elements.append(Paragraph(
        "Too much exploration wasted guesses on unlikely letters (X, Z, Q), while too little caused "
        "the agent to get stuck in local optima. Our solution: epsilon-greedy with HMM-guided exploration "
        "(smart exploration instead of random).",
        body_style
    ))
    
    elements.append(PageBreak())
    
    # Insights Gained
    elements.append(Paragraph("1.2 Insights Gained", heading2_style))
    
    elements.append(Paragraph("HMM as Powerful Linguistic Prior", heading3_style))
    elements.append(Paragraph(
        "The HMM captured English language patterns effectively. Top bigrams learned: TH (3.56%), "
        "HE (3.07%), AN (1.99%), ER (1.98%). Common starting letters: T (16%), A (11%), S (9%), C (8%). "
        "<b>Key Insight:</b> HMM alone achieved ~25% success rate, suggesting linguistic patterns matter "
        "more than game-specific strategy for Hangman.",
        body_style
    ))
    elements.append(Spacer(1, 0.1*inch))
    
    elements.append(Paragraph("Tabular Q-Learning Limitations", heading3_style))
    elements.append(Paragraph(
        "Q-learning faced fundamental issues. With 1,000 episodes × 10 states/episode = 10,000 states "
        "visited out of millions possible, coverage was <0.01% of state space. Most test states had no "
        "Q-values and defaulted to HMM. <b>Key Insight:</b> Function approximation (Deep Q-Networks) "
        "would be more suitable than tabular Q-learning for this high-dimensional problem.",
        body_style
    ))
    elements.append(Spacer(1, 0.1*inch))
    
    elements.append(Paragraph("Action Space Management Critical", heading3_style))
    elements.append(Paragraph(
        "Proper filtering achieved zero repeated guesses by maintaining available = "
        "set('abcdefghijklmnopqrstuvwxyz') - guessed_letters. This reduced action space from 26 to "
        "~15-20 mid-game, avoided -2 penalty per repeated guess, and focused computation on valid actions.",
        body_style
    ))
    elements.append(Spacer(1, 0.1*inch))
    
    elements.append(Paragraph("Word Length Impact", heading3_style))
    elements.append(Paragraph(
        "Performance varied by word length: Short (3-5 letters) ~20% success, Medium (6-10 letters) "
        "~35% success, Long (11+ letters) ~25% success. <b>Key Insight:</b> Medium-length words provide "
        "best balance between context and complexity.",
        body_style
    ))
    
    elements.append(PageBreak())
    
    # Section 2: Strategies
    elements.append(Paragraph("2. Strategies and Design Choices", heading1_style))
    
    elements.append(Paragraph("2.1 HMM Design", heading2_style))
    
    elements.append(Paragraph("First-Order Markov Model", heading3_style))
    elements.append(Paragraph(
        "We used bigram (two-letter) transitions with the formula:",
        body_style
    ))
    elements.append(Paragraph(
        "<font name='Courier' size=9>"
        "P(letter_i | letter_{i-1}) = [count(letter_{i-1}, letter_i) + α] / "
        "[Σ count(letter_{i-1}, l) + 26α]"
        "</font>",
        body_style
    ))
    elements.append(Paragraph(
        "where α = 0.01 (Laplace smoothing). <b>Rationale:</b> Simplicity (676 parameters vs. 17,576 for "
        "trigrams), data efficiency (50,000 words provide ~400,000 bigrams), computational cost O(26²) "
        "manageable, and effectiveness (captures most English patterns).",
        body_style
    ))
    elements.append(Spacer(1, 0.1*inch))
    
    elements.append(Paragraph("Smoothing Parameter (α = 0.01)", heading3_style))
    elements.append(Paragraph(
        "We tested α values: 0.001 (zero probabilities for rare bigrams), <b>0.01 (optimal balance)</b>, "
        "0.1 (common patterns weakened), 1.0 (near-uniform distribution). The chosen value preserves "
        "3-4 orders of magnitude difference between common (TH) and rare (QZ) bigrams.",
        body_style
    ))
    elements.append(Spacer(1, 0.1*inch))
    
    elements.append(Paragraph("Probability Aggregation", heading3_style))
    elements.append(Paragraph(
        "For masked word '_PP_E', we: (1) Find blank positions, (2) For each blank, get nearest left "
        "revealed letter, (3) Calculate transition probability, (4) Use <b>maximum</b> probability across "
        "positions (not average), (5) Normalize to probability distribution. Maximum aggregation focuses "
        "on most confident prediction and empirically outperformed averaging by 3-5%.",
        body_style
    ))
    
    elements.append(PageBreak())
    
    elements.append(Paragraph("2.2 RL State and Reward Design", heading2_style))
    
    elements.append(Paragraph("State Representation", heading3_style))
    elements.append(Paragraph(
        "<font name='Courier' size=9>"
        "state = {<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;'masked_word': str,&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# e.g., '_PP_E'<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;'guessed_letters': set,&nbsp;&nbsp;&nbsp;# e.g., {'a', 'p', 'e'}<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;'lives_left': int,&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# 0-6<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;'word_length': int&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# 3-15<br/>"
        "}"
        "</font>",
        body_style
    ))
    elements.append(Paragraph(
        "<b>Design rationale:</b> Masked word (essential for HMM), guessed letters (prevents repeats), "
        "lives left (influences risk-taking), word length (included but not in key to reduce state space).",
        body_style
    ))
    elements.append(Spacer(1, 0.1*inch))
    
    elements.append(Paragraph("Action Selection Strategy", heading3_style))
    elements.append(Paragraph(
        "Epsilon-greedy with HMM guidance: During exploration (random < ε), use HMM probabilities "
        "(smart exploration). During exploitation (random ≥ ε), use Q-values (learned strategy). "
        "<b>Key innovation:</b> Smart exploration using HMM instead of random selection explores promising "
        "actions and avoids obviously bad letters (X, Z, Q).",
        body_style
    ))
    elements.append(Spacer(1, 0.1*inch))
    
    elements.append(Paragraph("Q-Learning Hyperparameters", heading3_style))
    
    hyper_data = [
        ['Parameter', 'Value', 'Rationale'],
        ['Learning rate (α)', '0.1', 'Stable learning without oscillation'],
        ['Discount factor (γ)', '0.95', 'Values future rewards highly'],
        ['Initial epsilon', '1.0', '100% exploration at start'],
        ['Epsilon decay', '0.995', 'Gradual shift to exploitation'],
        ['Minimum epsilon', '0.01', '1% exploration maintained'],
        ['Training episodes', '1,000', 'Balance training time and coverage'],
    ]
    
    hyper_table = Table(hyper_data, colWidths=[1.5*inch, 1*inch, 2.5*inch])
    hyper_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    
    elements.append(hyper_table)
    
    elements.append(PageBreak())
    
    # Section 3: Exploration
    elements.append(Paragraph("3. Exploration vs. Exploitation Trade-off", heading1_style))
    
    elements.append(Paragraph(
        "We used epsilon-greedy with exponential decay: ε_t = max(0.01, 1.0 × 0.995^t). "
        "After 1000 episodes: ε ≈ 0.007.",
        body_style
    ))
    elements.append(Spacer(1, 0.1*inch))
    
    elements.append(Paragraph("Exploration Phase (Episodes 1-500)", heading3_style))
    elements.append(Paragraph(
        "High epsilon (ε > 0.3) primarily uses HMM probabilities, explores diverse word patterns, "
        "and populates Q-table with initial values. Benefits: Discovers effective letter sequences, "
        "learns which HMM predictions work best, builds foundation for exploitation.",
        body_style
    ))
    elements.append(Spacer(1, 0.1*inch))
    
    elements.append(Paragraph("Transition Phase (Episodes 500-800)", heading3_style))
    elements.append(Paragraph(
        "Medium epsilon (0.1 < ε < 0.3) balances HMM guidance and Q-values, refines Q-table estimates, "
        "and begins exploiting learned patterns.",
        body_style
    ))
    elements.append(Spacer(1, 0.1*inch))
    
    elements.append(Paragraph("Exploitation Phase (Episodes 800-1000)", heading3_style))
    elements.append(Paragraph(
        "Low epsilon (ε < 0.1) primarily uses Q-values with minimal exploration and converges to "
        "learned policy.",
        body_style
    ))
    elements.append(Spacer(1, 0.1*inch))
    
    elements.append(Paragraph("Why This Worked", heading3_style))
    elements.append(Paragraph(
        "(1) Smart exploration: HMM guidance prevented wasteful random guesses. "
        "(2) Gradual transition: Smooth shift from exploration to exploitation. "
        "(3) Maintained exploration: 1% epsilon prevents getting stuck. "
        "(4) HMM fallback: When Q-values unavailable, HMM provides reasonable action.",
        body_style
    ))
    
    elements.append(PageBreak())
    
    # Section 4: Future Improvements
    elements.append(Paragraph("4. Future Improvements", heading1_style))
    
    elements.append(Paragraph(
        "If we had another week, we would implement the following improvements:",
        body_style
    ))
    elements.append(Spacer(1, 0.1*inch))
    
    improvements = [
        ("Deep Q-Network (DQN)", "+10-15%", 
         "Replace Q-table with neural network to overcome state space explosion. "
         "Input: state features, Output: Q-values. Architecture: 3-layer MLP (256-128-64)."),
        
        ("Higher-Order HMM (Trigrams)", "+5-8%",
         "Model P(letter_i | letter_{i-2}, letter_{i-1}) to capture more context. "
         "Use trigram patterns: 'THE', 'ING', 'AND'. Backoff to bigrams when unavailable."),
        
        ("Word Frequency Weighting", "+3-5%",
         "Weight training by word frequency. Prioritize learning patterns from common words. "
         "Use word frequency database (e.g., Google N-grams)."),
        
        ("Position-Specific Models", "+4-6%",
         "Separate HMM for start/middle/end positions. Learn position-specific bigram patterns. "
         "Combine predictions with position weighting."),
        
        ("Ensemble Methods", "+5-7%",
         "Train multiple HMMs with different smoothing. Train multiple RL agents with different "
         "hyperparameters. Weighted voting for final action selection."),
        
        ("Adaptive Exploration", "+2-4%",
         "Increase epsilon when success rate drops. Decrease epsilon when performance improves. "
         "State-dependent exploration (more for unfamiliar patterns)."),
        
        ("Word Length-Specific Strategies", "+6-10%",
         "Train separate agents for short/medium/long words. Use length-specific reward structures. "
         "Optimize hyperparameters per length category."),
        
        ("Curriculum Learning", "+3-5%",
         "Start training on short, common words. Gradually increase difficulty. "
         "Progressive word length expansion."),
        
        ("Transfer Learning", "+15-20%",
         "Use BERT/GPT embeddings for word patterns. Fine-tune on Hangman task. "
         "Incorporate linguistic knowledge from large corpora. <b>(Most promising)</b>"),
        
        ("Monte Carlo Tree Search (MCTS)", "+8-12%",
         "Simulate future game trajectories. Evaluate expected outcomes. "
         "Select action with best expected value through lookahead."),
    ]
    
    for i, (title, improvement, description) in enumerate(improvements, 1):
        elements.append(Paragraph(f"{i}. {title} (Expected: {improvement})", heading3_style))
        elements.append(Paragraph(description, body_style))
        elements.append(Spacer(1, 0.08*inch))
    
    elements.append(Spacer(1, 0.1*inch))
    elements.append(Paragraph("Priority Improvements (Top 3)", heading3_style))
    elements.append(Paragraph(
        "If limited to one week: (1) <b>Deep Q-Network (DQN)</b> - Biggest impact on state space problem, "
        "(2) <b>Trigram HMM</b> - Significant improvement in predictions, "
        "(3) <b>Transfer Learning</b> - Leverage existing language models.",
        body_style
    ))
    elements.append(Paragraph(
        "<b>Combined expected improvement: +25-35% success rate → Target: 55-65% overall</b>",
        body_style
    ))
    
    elements.append(PageBreak())
    
    # Section 5: Conclusion
    elements.append(Paragraph("5. Conclusion", heading1_style))
    
    elements.append(Paragraph("5.1 Summary", heading2_style))
    elements.append(Paragraph(
        "Our Hangman agent achieved 29.41% success rate using a hybrid HMM-RL approach. "
        "Key achievements: Zero repeated guesses (perfect state tracking), effective linguistic "
        "pattern learning (HMM), and smart exploration strategy (HMM-guided epsilon-greedy).",
        body_style
    ))
    elements.append(Spacer(1, 0.1*inch))
    
    elements.append(Paragraph("5.2 Main Limitations", heading2_style))
    elements.append(Paragraph(
        "(1) State space explosion: Tabular Q-learning insufficient for high-dimensional problem. "
        "(2) Q-table sparsity: Limited generalization to unseen states. "
        "(3) HMM dominance: RL provided marginal improvement over pure HMM.",
        body_style
    ))
    elements.append(Spacer(1, 0.1*inch))
    
    elements.append(Paragraph("5.3 Key Learnings", heading2_style))
    elements.append(Paragraph(
        "(1) Linguistic priors crucial: Domain knowledge (HMM) more valuable than pure RL. "
        "(2) Function approximation needed: Neural networks required for complex state spaces. "
        "(3) Smart exploration effective: HMM-guided exploration better than random. "
        "(4) Reward engineering matters: Dense rewards significantly improved learning.",
        body_style
    ))
    elements.append(Spacer(1, 0.1*inch))
    
    elements.append(Paragraph("5.4 Final Thoughts", heading2_style))
    elements.append(Paragraph(
        "While our success rate is modest, the project provided valuable insights into combining "
        "probabilistic models (HMM) with reinforcement learning, managing exploration-exploitation "
        "trade-offs, handling high-dimensional state spaces, and the importance of domain knowledge "
        "in RL applications.",
        body_style
    ))
    elements.append(Spacer(1, 0.1*inch))
    elements.append(Paragraph(
        "With the proposed improvements (especially DQN and transfer learning), we believe 55-65% "
        "success rate is achievable, representing a 2x improvement over our current performance.",
        body_style
    ))
    
    elements.append(Spacer(1, 0.3*inch))
    elements.append(Paragraph("End of Report", 
                             ParagraphStyle('Center', parent=heading2_style, alignment=TA_CENTER)))
    
    # Build PDF
    doc.build(elements)
    print(f"✅ Report created successfully: {pdf_file}")
    print(f"   Total pages: ~15-20")
    print(f"   File size: Check {pdf_file}")

if __name__ == "__main__":
    try:
        create_report_pdf()
    except Exception as e:
        print(f"❌ Error creating PDF: {e}")
        print("   Make sure reportlab is installed: pip install reportlab")
