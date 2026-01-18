# Project Proposal — AI WasteWise

Title: AI WasteWise — Campus Waste Segregation & Recycling Assistant

Author
- Name / Login: Ndheeraj906
- College: [Alliance University]

SDG Alignment
- Primary: SDG 12 — Responsible Consumption and Production
- Secondary: SDG 11 — Sustainable Cities and Communities; SDG 13 — Climate Action

Problem Statement
How might we use AI to automatically classify and guide correct segregation of campus waste so that contamination of recyclable streams reduces and recycling rates on campus become more sustainable?

Who is affected
- Primary: Students, hostel residents, campus sanitation workers
- Secondary: Campus administration, recycling vendors, municipal waste managers
- Environmental: Reduced contamination -> higher recycling throughput -> lower landfill

Why AI is needed
- AI enables fast, low-cost, scalable visual classification of diverse waste types (images from phones or campus cameras)
- Can provide real-time guidance and generate analytics for pickup planning and contamination detection

Solution Overview
- Image classification model (MobileNet/ResNet backbone) to classify waste into 6 categories
- A lightweight Flask web demo for upload and result display
- Integration options:
  - Retrieval-Augmented Generation (RAG) for policy and recycling guidelines
  - Agent workflows to schedule pickups and generate routes
- Dashboard / analytics for contamination rates and pickup optimization (future extension)

Prototype Components (submitted)
1. Project description & SDG mapping — this document
2. Prototype/demo — `src/app.py`, `src/model.py`, `templates/*`
3. Training pipeline — `scripts/train.py`
4. Notebook demo — `notebooks/demo.md` (walkthrough for training & evaluation)

Responsible AI Considerations
- Data bias & representation: collect images across campuses, seasons, and lighting
- Privacy: do not collect or store identifiable human data; redact faces if present
- Transparency: return top-3 predictions with confidence and a short rationale
- Model updates: track dataset versions, use validation splits, and maintain audit logs
- Human-in-the-loop: recommend human review for low-confidence predictions (< 0.6)

Expected Impact
- Lower contamination rates in recyclable bins on campus
- Faster sorting, improved recycling revenue for campus vendors
- Awareness among students via an educational UI and feedback loop

Metrics for success
- Reduction in contamination rate (%) over 6 months
- Increase in correct recycling rate (%) in weekly audits
- User adoption: number of uploads or active users

Deployment & Next steps
- Improve model with transfer learning & targeted data augmentation
- On-device inference for offline/edge deployment (mobile)
- Integrate geospatial pickup scheduling and RAG-based policy assistant