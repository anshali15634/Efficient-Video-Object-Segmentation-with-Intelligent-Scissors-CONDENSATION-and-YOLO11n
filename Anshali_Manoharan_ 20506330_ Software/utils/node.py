# utils/node.py
class Node:
    def __init__(self, row, column):
        self.cost = float('inf')
        self.location = (row, column)
        self.prevNode = None
        self.expanded = False
    
    def __eq__(self, other):
        return self.location == other.location

    def __lt__(self, other):
        return self.cost < other.cost