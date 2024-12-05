from itertools import combinations
from collections import defaultdict
import pandas as pd
transactions = [
    ["Milk", "Bread", "Butter"],
    ["Bread", "Butter"],
    ["Milk", "Bread", "Sugar"],
    ["Milk", "Sugar"],
    ["Milk", "Bread", "Butter", "Sugar"],
    ["Bread", "Sugar"],
    ["Milk", "Bread"],
]
min_support = 0.2
min_confidence = 0.6

def calculate_support(itemset, transactions):
    count = sum(1 for transaction in transactions if set(itemset).issubset(set(transaction)))
    return count / len(transactions)

def generate_candidates(frequent_itemsets, k):
    items = set(item for itemset in frequent_itemsets for item in itemset)
    return [tuple(sorted(candidate)) for candidate in combinations(items, k)]

def apriori(transactions, min_support):
    transactions = list(map(set, transactions))
    itemsets = defaultdict(int)
    frequent_itemsets = []

    for transaction in transactions:
        for item in transaction:
            itemsets[frozenset([item])] += 1

    frequent = {itemset for itemset, count in itemsets.items() if count / len(transactions) >= min_support}
    frequent_itemsets.extend(frequent)

    print(itemsets)

    k = 2
    while k<=3:
        candidates = generate_candidates(frequent, k)
        itemsets = defaultdict(int)
        for transaction in transactions:
            for candidate in candidates:
                if set(candidate).issubset(transaction):
                    itemsets[frozenset(candidate)] += 1

        frequent = {itemset for itemset, count in itemsets.items() if count / len(transactions) >= min_support}
        frequent_itemsets.extend(frequent)
        k += 1

    return frequent_itemsets

def generate_rules(frequent_itemsets, transactions, min_confidence):
    rules = []
    transactions_count = len(transactions)

    for itemset in frequent_itemsets:
        itemset = set(itemset)
        if len(itemset) < 2:
            continue
        for consequent in itemset:
            antecedent = itemset - {consequent}
            antecedent_support = calculate_support(antecedent, transactions)
            rule_support = calculate_support(itemset, transactions)
            confidence = rule_support / antecedent_support if antecedent_support > 0 else 0

            if confidence >= min_confidence:
            #     consequent_support = calculate_support({consequent}, transactions)
            #     lift = (rule_support)/( consequent_support*antecedent_support) if consequent_support > 0 else 0
                rules.append((antecedent, consequent, confidence))

    return rules


frequent_itemsets = apriori(transactions, min_support)
frequent_3_itemsets = [itemset for itemset in frequent_itemsets if len(itemset) == 3]
print("Frequent 3-Itemsets:")
print(frequent_3_itemsets)

# ****-----------****

rules = generate_rules(frequent_itemsets, transactions, min_confidence)
print("association rules")
print(rules)