class ContourList:
    class Node:
        def __init__(self, min_x, max_x, offset):
            self.min_x = min_x
            self.max_x = max_x
            self.offset = offset
            self.prev = None
            self.next = None

        @staticmethod
        def OffsetLess(a, b):
            return a.offset < b.offset

    def __init__(self):
        self.head = None
        self.tail = None

    def Reset(self):
        self.head = None
        self.tail = None

    def Insert(self, min_x, max_x, offset):
        node = ContourList.Node(min_x, max_x, offset)
        if not self.head:
            self.head = self.tail = node
        else:
            self.tail.next = node
            node.prev = self.tail
            self.tail = node
        return node

    def Remove(self, node):
        if node.prev:
            node.prev.next = node.next
        else:
            self.head = node.next
        if node.next:
            node.next.prev = node.prev
        else:
            self.tail = node.prev

    def MaxElement(self, min_x, max_x, comparator):
        max_node = None
        max_offset = float('-inf')
        current = self.head
        while current:
            if current.min_x <= min_x and current.max_x >= max_x:
                if not max_node or comparator(current, max_node):
                    max_node = current
            current = current.next
        return max_node if max_node else self.head

    def begin(self):
        return self.head

    def end(self):
        return None

    def Print(self):
        current = self.head
        while current:
            print(f"[{current.min_x}, {current.max_x}] offset: {current.offset}", end=" -> ")
            current = current.next
        print("None")