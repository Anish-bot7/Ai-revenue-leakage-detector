import numpy as np

# Service Catalog
SERVICE_CATALOG = {

    "Cloud": {
        "unit_price": 20,
        "plans": ["Basic Cloud", "Business Cloud", "Enterprise Cloud"]
    },

    "Internet": {
        "unit_price": 10,
        "plans": ["Fiber 100", "Fiber 300", "Fiber 1000"]
    },

    "Voice": {
        "unit_price": 5,
        "plans": ["Voice Basic", "Voice Premium"]
    },

    "Security": {
        "unit_price": 15,
        "plans": ["Firewall", "SOC", "Endpoint Security"]
    },

    "SaaS": {
        "unit_price": 25,
        "plans": ["CRM", "ERP", "HRMS"]
    }

}

BILLING_CYCLE = [
    "Monthly",
    "Quarterly",
    "Yearly"
]

SLA_LEVEL = [
    "Bronze",
    "Silver",
    "Gold",
    "Platinum"
]


def generate_contract(contract_id):

    service = np.random.choice(list(SERVICE_CATALOG.keys()))

    service_info = SERVICE_CATALOG[service]

    unit_price = service_info["unit_price"]

    plan_name = np.random.choice(service_info["plans"])

    contract_duration = np.random.choice([12, 24, 36])

    estimated_usage = np.random.randint(500, 3000)

    contract_price = estimated_usage * unit_price

    return {

        "Contract_ID": f"CON{contract_id:05}",

        "Service_Type": service,

        "Plan_Name": plan_name,

        "Unit_Price": unit_price,

        "Estimated_Usage": estimated_usage,

        "Contract_Price": contract_price,

        "Discount_Allowed": np.random.choice([0, 5, 10, 15]),

        "Tax_Rate": 18,

        "Billing_Cycle": np.random.choice(BILLING_CYCLE),

        "Contract_Duration": contract_duration,

        "SLA_Level": np.random.choice(SLA_LEVEL)

    }