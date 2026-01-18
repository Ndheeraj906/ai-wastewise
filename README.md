# ai-wastewise

AI WasteWise — An AI-powered Campus Waste Segregation & Recycling Assistant

Summary
- Project type: Prototype web demo + training pipeline + documentation
- SDG primary alignment: SDG 12 — Responsible Consumption and Production
- Secondary SDGs: SDG 11 (Sustainable Cities and Communities), SDG 13 (Climate Action)
- Objective: Help students, campus staff, and sanitation teams reduce waste contamination by providing on-device and web-based waste classification, pickup optimization suggestions, and actionable recycling guidance.

Key features
- Image-based waste classification (Plastic / Paper / Metal / Glass / Organic / E-waste)
- Web demo (Flask) to upload images and get classification and guidance
- Training script (PyTorch / torchvision) and data layout
- Responsible AI guidance and auditing checklist
- Notebook-style demo and instructions for RAG & agentic workflows for policy retrieval (optional extension)

Quick start (development)
1. Create virtual environment:
   python -m venv .venv && source .venv/bin/activate
2. Install dependencies:
   pip install -r requirements.txt
3. Run the demo locally:
   export FLASK_APP=src/app.py
   flask run
4. Open http://127.0.0.1:5000 and upload a sample image

Project status
- Prototype-ready demo; model training pipeline included.
- Replace placeholder/demo model weights with your fine-tuned model for production use.

Responsible AI
- Data collection guidelines and checks (see docs/project-proposal.md)
- Bias checks for under/over-represented waste types and geographies
- Privacy: no personal data collection; images are processed transiently and not stored unless user opts in
- Model explainability: top-K labels + confidence scores + simple text rationale

Deliverables included
- Project description and SDG mapping (docs/project-proposal.md)
- Prototype/demo: Flask app (src/app.py), sample scripts
- Training pipeline (scripts/train.py) and notebook (notebooks/demo.md)
- Impact statement and Responsible AI considerations

Contact / Author
- Student: Ndheeraj906
- Internship: 1M1B AI + Sustainability (IBM SkillsBuild & AICTE)
