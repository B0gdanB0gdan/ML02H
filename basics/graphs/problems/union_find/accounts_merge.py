"""
Given a list of accounts where each element accounts[i] is a list of strings, 
where the first element accounts[i][0] is a name, and the rest of the elements 
are emails representing emails of the account.

Now, we would like to merge these accounts. 
Two accounts definitely belong to the same person if there is some common email to both accounts. 
Note that even if two accounts have the same name, they may belong to different people as people 
could have the same name. A person can have any number of accounts initially, 
but all of their accounts definitely have the same name.

After merging the accounts, return the accounts in the following format: the first element of each 
account is the name, and the rest of the elements are emails in sorted order. The accounts themselves 
can be returned in any order.


Input: 
accounts = [
    ["John","johnsmith@mail.com","john_newyork@mail.com"],
    ["John","johnsmith@mail.com","john00@mail.com"],
    ["Mary","mary@mail.com"],
    ["John","johnnybravo@mail.com"]
]
Output: [
    ["John","john00@mail.com","john_newyork@mail.com","johnsmith@mail.com"],
    ["Mary","mary@mail.com"],
    ["John","johnnybravo@mail.com"]
]
Explanation:
The first and second John's are the same person as they have the common email "johnsmith@mail.com".
The third John and Mary are different people as none of their email addresses are used by other accounts.
We could return these lists in any order, for example the answer [['Mary', 'mary@mail.com'], ['John', 'johnnybravo@mail.com'], 
['John', 'john00@mail.com', 'john_newyork@mail.com', 'johnsmith@mail.com']] would still be accepted.
"""

from ...templates.union_find import UnionFind
from collections import defaultdict


def accounts_merge(accounts: list[list[str]]) -> list[list[str]]:
    uf = UnionFind(len(accounts))

    mapping = {}
    for i, account in enumerate(accounts):
        for email in account[1:]:
            if email in mapping:
                uf.union(mapping[email], i)
            else:
                mapping[email] = i

    merged_accounts = defaultdict(list)
    for email in mapping:
        parent_email = uf.find(mapping[email])
        merged_accounts[parent_email].append(email)

    return [
        [accounts[root][0], *sorted(emails)] for root, emails in merged_accounts.items()
    ]


if __name__ == "__main__":
    accounts = [
        ["John","johnsmith@mail.com","john_newyork@mail.com"],
        ["John","johnsmith@mail.com","john00@mail.com"],
        ["Mary","mary@mail.com"],
        ["John","johnnybravo@mail.com"]
    ]
    print(accounts_merge(accounts))