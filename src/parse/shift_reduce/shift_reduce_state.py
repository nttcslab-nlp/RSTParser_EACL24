from data.tree import AttachTree


class ShiftReduceState(object):
    # TODO:
    # https://github.com/lianghuang3/lineardpparser/blob/master/code/newstate.py
    def __init__(self, n_edus: int):
        self.n_edus = n_edus
        self.stack = []
        self.queue = list(map(str, range(n_edus)[::-1]))
        self.score = 0

    def copy(self):
        # make new object
        x = ShiftReduceState(self.n_edus)
        # copy params
        x.stack = self.stack.copy()
        x.queue = self.queue.copy()
        x.score = self.score
        return x

    def operate(
        self,
        action: str,
        nuc: str | None = None,
        rel: str | None = None,
        score: float = 0,
    ):
        self.score = self.score + score

        if action == "shift":
            edu_idx = self.queue.pop()
            node = AttachTree("text", [edu_idx])
            self.stack.append(node)
        elif action == "reduce":
            assert nuc is not None and rel is not None
            r_node = self.stack.pop()
            l_node = self.stack.pop()
            label = ":".join([nuc, rel])
            new_node = AttachTree(label, [l_node, r_node])
            self.stack.append(new_node)
        else:
            raise ValueError("unexpected action: {}".format(action))

    def is_end(self):
        return len(self.stack) == 1 and len(self.queue) == 0

    def get_tree(self):
        if self.is_end():
            return self.stack[0]
        else:
            raise ValueError

    def get_state(self):
        def get_edu_span(node: AttachTree | str):
            if isinstance(node, AttachTree):
                leaves = node.leaves()
                span = (int(leaves[0]), int(leaves[-1]) + 1)
            else:
                edu_idx = node
                span = (int(edu_idx), int(edu_idx) + 1)

            return span

        # stack top1, top2
        s1 = get_edu_span(self.stack[-1]) if len(self.stack) > 0 else (-1, -1)
        s2 = get_edu_span(self.stack[-2]) if len(self.stack) > 1 else (-1, -1)
        # queue first
        q1 = get_edu_span(self.queue[-1]) if len(self.queue) > 0 else (-1, -1)
        return s1, s2, q1

    def allowed_actions(self):
        return [
            action for action in ["shift", "reduce"] if self.is_allowed_action(action)
        ]

    def is_allowed_action(self, action: str):
        if action == "shift":
            return len(self.queue) >= 1
        elif action == "reduce":
            return len(self.stack) >= 2
        else:
            raise ValueError
