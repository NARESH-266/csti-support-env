from typing import Dict, Any, List

class ToolRegistry:
    """Registry of mock diagnostic tools for the support environment."""
    
    def __init__(self):
        self.tools = {
            "lookup_customer_record": {
                "name": "lookup_customer_record",
                "description": "Retrieves more detailed billing and account status for a customer email.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "email": {"type": "string"}
                    },
                    "required": ["email"]
                }
            },
            "check_system_status": {
                "name": "check_system_status",
                "description": "Checks the real-time status of a specific service (e.g., 'billing', 'auth', 'database').",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "service": {"type": "string"}
                    },
                    "required": ["service"]
                }
            }
        }

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Returns metadata for all available tools."""
        return list(self.tools.values())

    def call_tool(self, name: str, arguments: Dict[str, Any], task_data: Dict[str, Any]) -> str:
        """Simulates calling a tool based on the task context."""
        if name == "lookup_customer_record":
            # Check if task has hidden customer data
            hidden = task_data.get("hidden_info", {}).get("customer_record")
            if hidden:
                return f"SUCCESS: Customer Record Found. Status: {hidden['status']}. Billing Note: {hidden['note']}."
            return "SUCCESS: Customer found. Status: Active. No billing issues found in record."
            
        elif name == "check_system_status":
            service = arguments.get("service")
            hidden = task_data.get("hidden_info", {}).get("system_status", {}).get(service)
            if hidden:
                return f"ALERT: Service '{service}' is experiencing {hidden['error']}. Last incident report: {hidden['time']}."
            return f"SUCCESS: Service '{service}' is operational (100% uptime)."
            
        return f"ERROR: Tool '{name}' not found."
