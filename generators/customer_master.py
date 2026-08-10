import pandas as pd
from generators.customer import generate_customer

def generate_customer_master(num_customers=500):

    customers = []

    for i in range(num_customers):
        customers.append(generate_customer(i + 1))

    df = pd.DataFrame(customers)

    return df

