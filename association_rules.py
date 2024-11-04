import itertools
import pandas as pd
import matplotlib.pyplot as plt
transactions = [
    {"Milk", "Bread", "Butter"},
    {"Bread", "Butter"},
    {"Milk", "Bread", "Sugar"},
    {"Milk", "Sugar"},
    {"Milk", "Bread", "Butter", "Sugar"},
    {"Bread", "Sugar"},
    {"Milk", "Bread"}
]
def calculate_support(transactions):
    itemsets = {}
    for transaction in transactions:
        for i in range(1, len(transaction) + 1):
            for itemset in itertools.combinations(transaction, i):
                itemset = frozenset(itemset)
                if itemset in itemsets:
                    itemsets[itemset] += 1
                else:
                    itemsets[itemset] = 1
    num_transactions = len(transactions)
    support = {itemset: count / num_transactions for itemset, count in itemsets.items()}
    return support



support = calculate_support(transactions)

def filter_frequent_itemsets(support, min_support):
    return {itemset: supp for itemset, supp in support.items() if supp >= min_support}

min_support = 0.3  # User-specified threshold
frequent_itemsets = filter_frequent_itemsets(support, min_support)

def calculate_confidence(frequent_itemsets, support):
    rules = []
    for itemset in frequent_itemsets.keys():
        if len(itemset) > 1:
            for antecedent in itertools.combinations(itemset, len(itemset) - 1):
                antecedent = frozenset(antecedent)
                consequent = itemset - antecedent
                confidence = support[itemset] / support[antecedent]
                rules.append((antecedent, consequent, confidence))
    return rules

rules = calculate_confidence(frequent_itemsets, support)

def calculate_lift(rules, support):
    lift_rules = []
    for antecedent, consequent, confidence in rules:
        consequent_support = support[consequent]
        lift = confidence / consequent_support
        lift_rules.append((antecedent, consequent, confidence, lift))
    return lift_rules

lift_rules = calculate_lift(rules, support)

itemsets_df = pd.DataFrame(list(frequent_itemsets.items()), columns=['Itemset', 'Support'])
itemsets_df['Itemset'] = itemsets_df['Itemset'].apply(lambda x: ', '.join(x))

plt.figure(figsize=(10, 6))
plt.barh(itemsets_df['Itemset'], itemsets_df['Support'], color='skyblue')
plt.xlabel('Support')
plt.ylabel('Itemset')
plt.title('Frequent Itemsets with Support')
plt.show()

print("Frequent Itemsets and Support:")
print(itemsets_df)
print("\nAssociation Rules with Confidence and Lift:")
for antecedent, consequent, confidence, lift in lift_rules:
    print(f"Rule: {set(antecedent)} -> {set(consequent)}, Confidence: {confidence:.2f}, Lift: {lift:.2f}")
