import pandas as pd
from generators.contract import generate_contract

def generate_contract_master(num_contracts=200):

    contracts = []

    for i in range(num_contracts):
        contracts.append(generate_contract(i + 1))

    df = pd.DataFrame(contracts)

    return df