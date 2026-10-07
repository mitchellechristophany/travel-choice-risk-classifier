# 🚘 Autonomous Travel Choice & Route Risk Classifier

A machine learning framework designed to model individual travel mode choices (Drive, Transit, Walk) and evaluate route-level risk factors using spatial accident indicators and mobility survey attributes.

## 📌 Key Capabilities
- **Discrete Choice Modeling:** Employs gradient boosted decision trees (LightGBM) to capture complex, non-linear relationships in traveler choice behavior.
- **Behavioral Feature Analysis:** Evaluates the relative impact of spatial accessibility, trip distance, income brackets, and urban density on modal split.
- **Route Risk Scoring:** Integrates spatial crash frequencies and vulnerability indices to score origin-destination corridor safety.

## 📐 System Architecture
```text
[ Mobility Survey Data & Spatial Metrics ]
                   │
                   ▼
     ┌───────────────────────────┐
     │ Feature Engineering &     │
     │ Data Preprocessing        │
     └─────────────┬─────────────┘
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
┌─────────────────┐ ┌──────────────────┐
│ Mode Choice     │ │ Route Risk       │
│ Classifier      │ │ Assessor         │
│ (LightGBM)      │ │ (Spatial Overlay)│
└────────┬────────┘ └────────┬─────────┘
         │                   │
         └─────────┬─────────┘
                   ▼
     [ Integrated Decision & Risk Output ]
