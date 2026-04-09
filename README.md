---
title: CSTI Support Intelligence
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# AI Support Ticket Intelligence (CSTI) Environment

**CSTI** is a real-world decision-making environment for Reinforcement Learning agents, specifically designed for the **OpenEnv Round 1 Challenge**. The agent acts as a Senior Support Triage Engineer in a high-growth SaaS company, making critical routing and prioritization decisions.

## 🌟 Why This Matters
In modern customer service, speed and accuracy are paramount. Misrouting a VIP ticket or failing to identify a high-churn-risk sentiment can lead to significant revenue loss. CSTI provides a rigorous testbed for agents to demonstrate:
- **Linguistic Nuance**: Detecting frustration and complex multi-intent requests.
- **Contextual Awareness**: Utilizing customer history and segment (VIP/Standard) to set SLA priority.
- **Policy Adherence**: Balancing strict company policies with customer satisfaction.

## 🎮 Environment Specs

### Observation Space (The Context)
The agent receives a rich state including:
- `content`: The raw customer inquiry.
- `customer_segment`: `standard` or `vip`.
- `sla_deadline`: Time remaining for a guaranteed response.
- `ticket_history`: Past interaction logs to provide historical context.

### Action Space (The Decision)
The agent classifies the ticket by selecting:
- **Department**: `technical`, `billing`, `sales`, `logistics`, `general`.
- **Priority**: `low`, `medium`, `high`, `urgent`.
- **Tags**: Relevant metadata (e.g., `refund`, `churn-risk`, `broken-item`).

## 📊 Reward Smoothing Logic
CSTI uses a continuous reward function designed to differentiate between "clueless" and "expert" agents:
- **Department Match (50%)**: +0.5 for exact match.
- **Priority Distance (30%)**: 
  - Exact match: +0.3
  - Off-by-one level: +0.15
  - Off-by-two levels: +0.05
- **Tag Precision (20%)**: Proportional points based on correctly identified key metadata tags.

## 📝 Example Interaction (Easy Task)
**Input Content**: *"Can you please help me reset my password? I need to access my files urgently."*
**Expert Agent Action**:
```json
{
  "department": "technical",
  "priority": "medium",
  "tags": ["login", "password-reset"]
}
```
**Reward**: `0.99` (Perfect Match)

## 🏗️ Project Structure
```
decision_env/
├── decision_env/
│   ├── env.py       # Core OpenEnv logic
│   ├── models.py    # Pydantic schemas
│   ├── tasks.py     # Hierarchical tasks (Easy to Hard)
│   ├── grader.py    # Deterministic Partial-Credit Grader
│   └── server.py    # FastAPI Deployment Server
├── openenv.yaml     # Environment Metadata
├── inference.py     # Baseline LLM-Agent Script
├── Dockerfile       # Deployment Configuration
└── README.md        # Documentation
```

## 🚀 Deployment
1. Build the container: `docker build -t csti-env .`
2. Run on port 7860: `docker run -p 7860:7860 csti-env`
3. Test `/reset` endpoint: `curl -X POST "http://localhost:7860/reset?task_id=easy_login_issue"`
