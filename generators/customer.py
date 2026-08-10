from faker import Faker
import numpy as np

fake = Faker()

CUSTOMER_TYPES = [
    "Enterprise",
    "SMB",
    "Government",
    "Retail"
]

INDUSTRIES = [
    "IT",
    "Finance",
    "Healthcare",
    "Telecom",
    "Manufacturing",
    "Education",
    "Logistics"
]

REGIONS = [
    "North",
    "South",
    "East",
    "West"
]

COUNTRIES = [
    "India",
    "USA",
    "UK",
    "Singapore",
    "Australia"
]

ACCOUNT_STATUS = [
    "Active",
    "Inactive",
    "Suspended"
]

PAYMENT_TERMS = [
    "Net 15",
    "Net 30",
    "Net 45",
    "Net 60"
]

ACCOUNT_MANAGERS = [
    "Rahul Sharma",
    "Priya Nair",
    "John Thomas",
    "Arjun Kumar",
    "Sneha Patel"
]


def generate_customer(customer_id):

    return {

        "Customer_ID": f"CUST{customer_id:05}",

        "Customer_Name": fake.company(),

        "Customer_Type": np.random.choice(CUSTOMER_TYPES),

        "Industry": np.random.choice(INDUSTRIES),

        "Region": np.random.choice(REGIONS),

        "Country": np.random.choice(COUNTRIES),

        "City": fake.city(),

        "Email": fake.company_email(),

        "Phone": fake.phone_number(),

        "Registration_Date": fake.date_between(
            start_date="-5y",
            end_date="today"
        ),

        "Account_Status": np.random.choice(ACCOUNT_STATUS),

        "Payment_Terms": np.random.choice(PAYMENT_TERMS),

        "Credit_Limit": np.random.randint(50000,1000000),

        "Account_Manager": np.random.choice(ACCOUNT_MANAGERS)

    }