from typing import Dict, Any, List

TASKS: Dict[str, Dict[str, Any]] = {
    "easy_login_issue": {
        "id": "easy_login_issue",
        "name": "Password Reset Help",
        "difficulty": "easy",
        "description": "Clear technical support request for a password reset.",
        "input": {
            "ticket_id": "TKT-001",
            "content": "Hi, I've been trying to log into my account for the past hour and it says my password is incorrect. Can you please help me reset it? I need to access my files urgently.",
            "customer_segment": "standard",
            "sla_deadline": "2 hours",
            "ticket_history": ["2024-03-01: Login successful", "2024-02-15: Password changed"]
        },
        "ground_truth": {
            "department": "technical",
            "priority": "medium",
            "required_tags": ["login", "password-reset"],
        }
    },
    "medium_billing_angry": {
        "id": "medium_billing_angry",
        "name": "Duplicate Charge & Angry Customer",
        "difficulty": "medium",
        "description": "Mixed intent involving billing error and high negative sentiment.",
        "input": {
            "ticket_id": "TKT-002",
            "content": "I am EXTREMELY disappointed! I just checked my bank statement and I've been charged twice for my 'Pro' subscription this month. This is unacceptable. I want a refund for the second charge immediately and some compensation for this stress!",
            "customer_segment": "standard",
            "sla_deadline": "4 hours",
            "ticket_history": ["2024-01-20: Subscription started", "2024-02-20: Payment processed"]
        },
        "ground_truth": {
            "department": "billing",
            "priority": "high",
            "required_tags": ["refund", "angry", "double-charge"],
        }
    },
    "hard_policy_exception": {
        "id": "hard_policy_exception",
        "name": "VIP Damage Claim & Policy Exception",
        "difficulty": "hard",
        "description": "Complex request with multi-intent ambiguity and emotional tone.",
        "input": {
            "ticket_id": "TKT-003",
            "content": "Look, my order #98765 arrived with the glass shattered. It's been a disaster day. I know your 'official' policy says report in 24h, and yeah it's been like 27-28 hours because I was stuck at a funeral. I've been a VIP member for years and spent a fortune here. If I don't get an overnight replacement for free, I'm canceling my Pro plan, removing my credit card, and telling my followers on X about this. Also, the delivery guy left it in the rain so the box is soaked too. Just fix this now.",
            "customer_segment": "vip",
            "sla_deadline": "1 hour",
            "ticket_history": [
                "2023-12-15: Large order #88211 (Success)",
                "2024-01-05: Refund for late delivery (Resolved)",
                "2024-03-10: VIP status renewed"
            ]
        },
        "ground_truth": {
            "department": "logistics",
            "priority": "urgent",
            "required_tags": ["damaged-item", "policy-exception", "replacement", "churn-risk", "rain-damage"],
        }
    }
}

def get_task(task_id: str) -> Dict[str, Any]:
    return TASKS.get(task_id, TASKS["easy_login_issue"])
