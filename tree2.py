import random
import networkx as nx
import matplotlib.pyplot as plt
import math

from contour_list import ContourList

################################################################################
class BuildingBlock:
    def __init__(self, name):
        self.name = name
        self.merge_device_names = []
        self.patterns = []
        self.min_x = 0.0
        self.min_y = 0.0
        self.max_x = 0.0
        self.max_y = 0.0
        self.width = 0.0
        self.height = 0.0
        self.column_multiple = 1
        self.row_multiple = 1
        self.variants = []

    def MoveCenterXTo(self, x):
        w = self.get_width()
        self.min_x = x - w * 0.5
        self.max_x = x + w * 0.5

    def MoveMinXTo(self, x):
        w = self.get_width()
        self.min_x = x
        self.max_x = x + w

    def MoveMaxXTo(self, x):
        w = self.get_width()
        self.max_x = x
        self.min_x = x - w

    def MoveMinYTo(self, y):
        h = self.get_height()
        self.min_y = y
        self.max_y = y + h

    def get_min_x(self):
        return self.min_x

    def get_max_x(self):
        return self.max_x

    def get_min_y(self):
        return self.min_y

    def get_max_y(self):
        return self.max_y

    def GetCenterX(self):
        return (self.min_x + self.max_x) * 0.5

    def GetCenterY(self):
        return (self.min_y + self.max_y) * 0.5

    def get_width(self):
        return self.max_x - self.min_x if self.max_x > self.min_x else self.width

    def get_height(self):
        return self.max_y - self.min_y if self.max_y > self.min_y else self.height

    def SetW(self, w):
        self.width = w

    def SetH(self, h):
        self.height = h

    def GetColumnMultiple(self):
        return self.column_multiple

    def GetRowMultiple(self):
        return self.row_multiple

class SymmetryUnit:
    def __init__(self, r_half, l_half):
        self.r_half = r_half
        self.l_half = l_half
        self.l_child = None
        self.r_child = None
        self.parent = None
        self.l_hint = None
        self.r_hint = None
        if r_half != l_half:
            r_half.patterns.sort()
            l_half.patterns.sort()
            # assert l_half.patterns and r_half.patterns  # ZAKOMENTUJTE TUTO ŘÁDKU
            # assert l_half.patterns == r_half.patterns   # ZAKOMENTUJTE TUTO ŘÁDKU
            l_half.SetW(l_half.patterns[0].width  if l_half.patterns else l_half.width)
            l_half.SetH(l_half.patterns[0].height if l_half.patterns else l_half.height)
            r_half.SetW(r_half.patterns[0].width  if r_half.patterns else r_half.width)
            r_half.SetH(r_half.patterns[0].height if r_half.patterns else r_half.height)

class topology_BStarTree:
    def __init__(self, block_or_group, group=None):
        if group is not None:
            self.name = str(block_or_group)
            self.units = [SymmetryUnit(b1, b2) for b1, b2 in group]
        elif isinstance(block_or_group, BuildingBlock):
            self.name = block_or_group.name
            self.units = [SymmetryUnit(block_or_group, block_or_group)]
        else:
            self.name = str(block_or_group)
            self.units = [SymmetryUnit(b1, b2) for b1, b2 in block_or_group]
        self.root = self.units[0]
        self.l_child = None
        self.r_child = None
        self.parent = None
        self.min_x = 0.0
        self.min_y = 0.0
        self.max_x = 0.0
        self.max_y = 0.0
        self.begin = None
        self.end = None
        self.bbox = BBox()

class Loader:
    ############################################################################
    def __init__(self):
        self.block_name_to_block = {}
        self.net_name_to_blocks = {}
        self.sym_group_name_to_sym_group = {}
        self.blocks = set()

    ############################################################################
    def Load(self, netlist_file_path, symmetry_file_path, block_file_path):
        if not self.LoadBlockFile(block_file_path):
            print(f"Failed to open file: {block_file_path}")
            return False
        if not self.LoadNetlistFile(netlist_file_path):
            print(f"Failed to open file: {netlist_file_path}")
            return False
        if not self.LoadSymmetryConstraintFile(symmetry_file_path):
            print(f"Failed to open file: {symmetry_file_path}")
            return False
        return True

    ############################################################################
    def LoadBlockFile(self, filepath):
        try:
            with open(filepath) as file:
                for line in file:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    tokens = line.split()
                    name = tokens[0]
                    block = BuildingBlock(name)
                    merge_names = []
                    variants = []
                    i = 1
                    while i < len(tokens) and not tokens[i].startswith('('):
                        merge_names.append(tokens[i])
                        i += 1
                    while i < len(tokens):
                        if tokens[i].startswith('('):
                            param_tokens = []
                            param_tokens.append(tokens[i][1:])
                            i += 1
                            while i < len(tokens) and not tokens[i].endswith(')'):
                                param_tokens.append(tokens[i])
                                i += 1
                            if i < len(tokens):
                                param_tokens.append(tokens[i][:-1])
                            param_vals = param_tokens
                            if len(param_vals) >= 4:
                                variants.append((
                                    float(param_vals[0]),
                                    float(param_vals[1]),
                                    int(param_vals[2]),
                                    int(param_vals[3])
                                ))
                        i += 1
                    if not variants:
                        continue  # ignorovat blok bez parametrů
                    block.width = variants[0][0]
                    block.height = variants[0][1]
                    block.row_multiple = variants[0][2]
                    block.column_multiple = variants[0][3]
                    block.variants = variants[1:]
                    block.merge_device_names = merge_names
                    self.block_name_to_block[block.name] = block
                    for merge_name in merge_names:
                        self.block_name_to_block[merge_name] = block
                    self.blocks.add(block)
            return True
        except Exception as e:
            print(f"Chyba při načítání bloků: {e}")
            return False

    ############################################################################
    def LoadNetlistFile(self, filepath):
        try:
            with open(filepath) as file:
                device_name_to_block = {}
                for name, block in self.block_name_to_block.items():
                    if not block.merge_device_names:
                        device_name_to_block[name] = block
                    else:
                        for merge_name in block.merge_device_names:
                            device_name_to_block[merge_name] = block
                for line in file:
                    tokens = line.strip().split()
                    if not tokens:
                        continue
                    device_name = tokens[0]
                    if device_name.startswith('M') and len(tokens) >= 4:
                        d, g, s = tokens[1:4]
                        d = d.lower()
                        g = g.lower()
                        s = s.lower()
                        block = device_name_to_block.get(device_name)
                        if block:
                            self.net_name_to_blocks.setdefault(d, set()).add(block)
                            self.net_name_to_blocks.setdefault(g, set()).add(block)
                            self.net_name_to_blocks.setdefault(s, set()).add(block)
            return True
        except Exception as e:
            print(f"Chyba při načítání netlistu: {e}")
            return False

    ############################################################################
    def LoadSymmetryConstraintFile(self, filepath):
        try:
            with open(filepath) as file:
                for line in file:
                    tokens = line.strip().split()
                    if not tokens:
                        continue
                    group_name = tokens[0]
                    block_name1 = tokens[1]
                    block1 = self.block_name_to_block.get(block_name1)
                    if not block1:
                        continue
                    if len(tokens) > 2:
                        block_name2 = tokens[2]
                        block2 = self.block_name_to_block.get(block_name2)
                        if not block2:
                            continue
                    else:
                        block2 = block1
                    self.sym_group_name_to_sym_group.setdefault(group_name, set()).add((block1, block2))
            return True
        except Exception as e:
            print(f"Chyba při načítání symetrie: {e}")
            return False

    ############################################################################
    def get_nets(self):
        nets = []
        for name, blocks in self.net_name_to_blocks.items():
            nets.append((name, list(blocks)))
        return nets

    ############################################################################
    def get_nodes(self):
        constrained_blocks = set()
        for group in self.sym_group_name_to_sym_group.values():
            for block1, block2 in group:
                constrained_blocks.add(block1)
                if block1 != block2:
                    constrained_blocks.add(block2)
                    
        unconstrained_blocks = self.blocks - constrained_blocks

        nodes = []
        for block in unconstrained_blocks:
            nodes.append(topology_BStarTree(block))

        for group_name, group in self.sym_group_name_to_sym_group.items():
            nodes.append(topology_BStarTree(group_name, group))
        return nodes

################################################################################
class BBox:
    def __init__(self):
        self.min_x = 0.0
        self.min_y = 0.0
        self.max_x = 0.0
        self.max_y = 0.0

    def set_min_x(self, x):
        self.min_x = x
        return self

    def set_min_y(self, y):
        self.min_y = y
        return self

    def set_max_x(self, x):
        self.max_x = x
        return self

    def set_max_y(self, y):
        self.max_y = y
        return self

    def SetW(self, w):
        self.max_x = self.min_x + w
        return self

    def SetH(self, h):
        self.max_y = self.min_y + h
        return self

    def get_min_x(self):
        return self.min_x

    def get_min_y(self):
        return self.min_y

    def get_max_x(self):
        return self.max_x

    def get_max_y(self):
        return self.max_y

    def GetCenterX(self):
        return (self.min_x + self.max_x) / 2

    def GetCenterY(self):
        return (self.min_y + self.max_y) / 2

    def get_width(self):
        return self.max_x - self.min_x

    def get_height(self):
        return self.max_y - self.min_y

class Perturbator:
    def __init__(self, hbtree):
        self.hbtree = hbtree
        self.blocks = hbtree.blocks
        self.asfbtrees_ = hbtree.topology_btrees
        # self.rebuildable_trees_ = [t for t in self.asfbtrees_ if len(t.units) > 1]
        # self.resizeable_units_ = [u for t in self.asfbtrees_ for u in t.units if len(u.r_half.patterns) > 1]
        # self.flipable_units_ = [u for t in self.asfbtrees_ for u in t.units if u.l_half != u.r_half]

        self.rebuildable_trees_ = []
        self.resizeable_units_ = []
        self.flipable_units_ = []

        # collect candidates
        for asfbtree in self.asfbtrees_:
            if len(asfbtree.units) > 1:
                self.rebuildable_trees_.append(asfbtree)

        for asfbtree in self.asfbtrees_:
            for unit in asfbtree.units:
                if unit.l_half != unit.r_half:
                    self.flipable_units_.append(unit)
                if len(unit.r_half.patterns) > 1:
                    self.resizeable_units_.append(unit)

        # perturbation functions
        self.perturb_funcs = [
            self.random_resize_symmetry_unit,
            self.random_swap_asfbstar_tree,
            self.random_swap_asfbstar_tree_child,
            self.random_flip_symmetry_unit,
            self.random_rebuild_asfbstar_tree,
            self.random_transplant_asfbstar_tree
        ]
        self.rng = random.Random()

    def __call__(self):
        if not self.perturb_funcs:
            return

        for block in self.blocks:
            if hasattr(block, "variants") and block.variants:
                variant = random.choice(block.variants)
                block.current_variant = variant
                block.width = variant[0]
                block.height = variant[1]
                block.row_multiple = variant[2]
                block.column_multiple = variant[3]
                # Reset pozice na (0,0) – bude přepočítáno packorem
                block.min_x = 0.0
                block.min_y = 0.0
                block.max_x = block.width
                block.max_y = block.height

        # equivalent of std::discrete_distribution
        num_perturb = self.rng.randint(1, len(self.perturb_funcs))
        for _ in range(num_perturb):
            func = self.rng.choice(self.perturb_funcs)
            func()

    def random_resize_symmetry_unit(self):
        if not self.resizeable_units_:
            return
        unit = self.rng.choice(self.resizeable_units_)
        pattern = self.rng.choice(unit.r_half.patterns)
        unit.r_half.SetW(pattern.width)
        unit.r_half.SetH(pattern.height)
        unit.l_half.SetW(pattern.width)
        unit.l_half.SetH(pattern.height)

    def random_swap_asfbstar_tree(self):
        if len(self.asfbtrees_) < 2:
            return
        t1, t2 = self.rng.sample(self.asfbtrees_, 2)
        self.swap_without_link(t1, t2)

    def random_swap_asfbstar_tree_child(self):
        if not self.asfbtrees_:
            return
        t = self.rng.choice(self.asfbtrees_)
        t.l_child, t.r_child = t.r_child, t.l_child

    def random_flip_symmetry_unit(self):
        if not self.flipable_units_:
            return
        unit = self.rng.choice(self.flipable_units_)
        unit.l_child, unit.r_child = unit.r_child, unit.l_child

    def random_rebuild_asfbstar_tree(self):
        if not self.rebuildable_trees_:
            return
        asfbtree = self.rng.choice(self.rebuildable_trees_)
        units = asfbtree.units
        self.rng.shuffle(units)
        asfbtree.root = units[0]
        units[0].parent = None
        for i in range(len(units) - 1):
            units[i].l_child = None
            units[i].r_child = units[i + 1]
        units[-1].l_child = None
        units[-1].r_child = None
        asfbtree.units = units

    def random_transplant_asfbstar_tree(self):
        if len(self.asfbtrees_) < 2:
            return
        t1, t2 = self.rng.sample(self.asfbtrees_, 2)
        self.transplant(t1, t2)

    def swap_without_link(self, a, b):
        a.bbox, b.bbox = b.bbox, a.bbox
        a.name, b.name = b.name, a.name
        a.units, b.units = b.units, a.units
        a.root, b.root = b.root, a.root
        a.begin, b.begin = b.begin, a.begin
        a.end, b.end = b.end, a.end

    def transplant(self, trans, to):
        child = self.get_rand_child(trans)
        if child:
            grandchild = self.get_rand_child(child)
            while grandchild:
                child = grandchild
                grandchild = self.get_rand_child(child)
            self.swap_without_link(trans, child)
            if to == child:
                to = trans
            trans = child

        # remove trans from its parent
        if trans.parent:
            if trans.parent.l_child == trans:
                trans.parent.l_child = None
            elif trans.parent.r_child == trans:
                trans.parent.r_child = None
            else:
                raise RuntimeError("Unreachable branch in transplant")

        def insert(to_child):
            if not to_child:
                return trans
            to_child.parent = trans
            if self.rng.choice([True, False]):
                trans.l_child = to_child
            else:
                trans.r_child = to_child
            return trans
        if self.rng.choice([True, False]):
            to.l_child = insert(to.l_child)
        else:
            to.r_child = insert(to.r_child)
        trans.parent = to

    def get_rand_child(self, tree):
        if tree.l_child and tree.r_child:
            return self.rng.choice([tree.l_child, tree.r_child])
        if tree.l_child:
            return tree.l_child
        if tree.r_child:
            return tree.r_child
        return None

class Packor:
    def __init__(self, hbtree):
        self.hbtree = hbtree
        self.packor_square = top_BStarTree.PackorSquare(hbtree)

    def __call__(self):
        self.packor_square()

################################################################################
class top_BStarTree:
    def __init__(self):
        self.units = []
        self.blocks = []
        self.nets = []
        self.topology_btrees = []
        self.min_x = 0.0
        self.min_y = 0.0
        self.max_x = 0.0
        self.max_y = 0.0
        self.root = None
        self.rng = random.Random()

    def NaivePlacement(self):
        x0 = 0.0
        x1 = 0.0
        y = 0.0
        for unit in self.units:
            representative = unit.r_half
            if unit.r_half == unit.l_half:
                representative.MoveCenterXTo(0.0)
                representative.MoveMinYTo(y)
                x0 = min(x0, representative.get_min_x())
                x1 = max(x1, representative.get_max_x())
            else:
                unit.r_half.MoveMinXTo(0.0)
                unit.r_half.MoveMinYTo(y)
                unit.l_half.MoveMaxXTo(0.0)
                unit.l_half.MoveMinYTo(y)
                x0 = min(x0, unit.l_half.get_min_x())
                x1 = max(x1, unit.r_half.get_max_x())
            y += representative.get_height()
        self.set_min_x(x0).set_min_y(0.0).set_max_x(x1).set_max_y(y)

    def BuildTree(self):
        random.shuffle(self.topology_btrees)
        self.root = self.topology_btrees[0]
        self.topology_btrees[0].parent = None
        for i in range(len(self.topology_btrees) - 1):
            self.topology_btrees[i + 1].parent = self.topology_btrees[i]
            self.topology_btrees[i].l_child = self.topology_btrees[i + 1]
            self.topology_btrees[i].r_child = None
        self.topology_btrees[-1].l_child = None
        self.topology_btrees[-1].r_child = None
        for topology_btree in self.topology_btrees:
            units = topology_btree.units
            random.shuffle(units)
            topology_btree.root = units[0]
            units[0].parent = None
            for i in range(len(units) - 1):
                units[i + 1].parent = units[i]
                units[i].l_child = None
                units[i].r_child = units[i + 1]
            units[-1].l_child = None
            units[-1].r_child = None
    # wmi TODO: check contour-based packing algorithm
    # here could be problem with placement of the second block
    class PackorSquare:
        def __init__(self, hbtree):
            self.contours_ = ContourList()
            self.hbtree_ = hbtree

        def __call__(self):
            if not self.hbtree_.root:
                return
            self.contours_.Reset()
            self.hbtree_.set_min_x(0.0).set_min_y(0.0).set_max_x(0.0).set_max_y(0.0)
            self.hbtree_.root.bbox.set_min_x(0.0).set_min_y(0.0)
            self.PackDFS(self.hbtree_.root, self.contours_.begin())

        def PackDFS(self, asfbtree, hint):
            asfbtree.bbox.SetH(0.0).SetW(0.0)
            if asfbtree.root.r_half == asfbtree.root.l_half:
                asfbtree.root.r_half.MoveCenterXTo(0.0)
            else:
                asfbtree.root.r_half.MoveMinXTo(0.0)
                asfbtree.root.l_half.MoveMaxXTo(0.0)
            self.PackPass1(asfbtree, asfbtree.root)
            self.PackPass2(asfbtree, asfbtree.root, hint, hint)
            asfbtree.bbox.set_min_y(asfbtree.root.r_half.get_min_y())
            self.hbtree_.set_max_x(max(self.hbtree_.get_width(), asfbtree.bbox.get_max_x() - self.hbtree_.get_min_x()))
            self.hbtree_.set_max_y(max(self.hbtree_.get_height(), asfbtree.bbox.get_max_y() - self.hbtree_.get_min_y()))
            if asfbtree.l_child:
                asfbtree.l_child.bbox.set_min_x(asfbtree.bbox.get_max_x())
                self.PackDFS(asfbtree.l_child, asfbtree.end)
            if asfbtree.r_child:
                asfbtree.r_child.bbox.set_min_x(asfbtree.bbox.get_min_x())
                self.PackDFS(asfbtree.r_child, asfbtree.begin)

        def PackPass1(self, asfbtree, unit):
            if not unit:
                return
            l_child = unit.l_child
            r_child = unit.r_child
            if l_child:
                l_child.r_half.MoveMinXTo(unit.r_half.get_max_x())
                l_child.l_half.MoveMaxXTo(unit.l_half.get_min_x())
            if r_child:
                if r_child.r_half == r_child.l_half:
                    r_child.r_half.MoveCenterXTo(0.0)
                else:
                    if unit.r_half == unit.l_half:
                        r_child.r_half.MoveMinXTo(0.0)
                        r_child.l_half.MoveMaxXTo(0.0)
                    else:
                        r_child.r_half.MoveMinXTo(unit.r_half.get_min_x())
                        r_child.l_half.MoveMaxXTo(unit.l_half.get_max_x())
            asfbtree.bbox.SetW(max(asfbtree.bbox.get_width(), unit.r_half.get_max_x() - unit.l_half.get_min_x()))
            self.PackPass1(asfbtree, l_child)
            self.PackPass1(asfbtree, r_child)

        def PackPass2(self, asfbtree, unit, l_hint, r_hint):
            if not unit:
                return
            if unit.l_half == unit.r_half:
                self.PackSelfSymmetric(asfbtree, unit, r_hint)
            else:
                self.PackSymmetryPair(asfbtree, unit, l_hint, r_hint)
            asfbtree.bbox.SetH(max(asfbtree.bbox.get_height(), unit.r_half.get_max_y() - asfbtree.root.r_half.get_min_y()))
            self.PackPass2(asfbtree, unit.l_child, unit.l_hint, unit.r_hint)
            self.PackPass2(asfbtree, unit.r_child, unit.l_hint, unit.r_hint)

        def PackSelfSymmetric(self, asfbtree, unit, r_hint):
            r_half = unit.r_half
            r_half.MoveCenterXTo(asfbtree.bbox.GetCenterX())
            unit.r_hint = unit.l_hint = self.contours_.MaxElement(r_half.get_min_x(), r_half.get_max_x(), ContourList.Node.OffsetLess)
            feasible_min_y = unit.r_hint.offset if unit.r_hint is not None else 0.0
            #feasible_min_y = asfbtree.root.r_half.get_min_y()
            r_half.MoveMinYTo(feasible_min_y)
            unit.r_hint = unit.l_hint = \
            self.contours_.Insert(r_half.get_min_x(), r_half.get_max_x(), r_half.get_max_y())

        def PackSymmetryPair(self, asfbtree, unit, l_hint, r_hint):
            r_half = unit.r_half
            l_half = unit.l_half
            symmetry_axis_x = asfbtree.bbox.GetCenterX()
            r_half.MoveMinXTo(r_half.get_min_x() + symmetry_axis_x)
            l_half.MoveMaxXTo(l_half.get_max_x() + symmetry_axis_x)
            unit.r_hint = self.contours_.MaxElement(r_half.get_min_x(), r_half.get_max_x(), ContourList.Node.OffsetLess)
            unit.l_hint = self.contours_.MaxElement(l_half.get_min_x(), l_half.get_max_x(), ContourList.Node.OffsetLess)
            feasible_min_y = unit.r_hint.offset if unit.r_hint is not None else 0.0
            #feasible_min_y = asfbtree.root.r_half.get_min_y()
            r_half.MoveMinYTo(feasible_min_y)
            l_half.MoveMinYTo(feasible_min_y)
            unit.r_hint = self.contours_.Insert(r_half.get_min_x(), r_half.get_max_x(), r_half.get_max_y())
            unit.l_hint = self.contours_.Insert(l_half.get_min_x(), l_half.get_max_x(), l_half.get_max_y())


    def Load(self, netlist_file_path, symmetry_file_path, block_file_path):
        loader = Loader()
        if not loader.Load(netlist_file_path, symmetry_file_path, block_file_path):
            return False
        self.nets = loader.get_nets()
        # dba TODO: understand why loader.get_nodes() is needed and what it does
        # it creates topology_btrees from blocks and symmetry constraints
        # it filters block by symmetry constraints
        self.topology_btrees = loader.get_nodes()
        for topology_btree in self.topology_btrees:
            for unit in topology_btree.units:
                self.units.append(unit)
        for unit in self.units:
            self.blocks.append(unit.r_half)
            if unit.l_half != unit.r_half:
                self.blocks.append(unit.l_half)
        self.BuildTree() # it creates random B*tree (root + topology_btrees + units) from (self.topology_btrees)
        # visualize_tree(self.root)
        top_BStarTree.PackorSquare(self)() # it places blocks according to B*tree
        # visualize_block_positions(self.blocks)
        visualize_tree_and_blocks2(self.root, self.blocks)
        return True

    def Dump(self, output_file_path):
        try:
            with open(output_file_path, "w") as file:
                file.write(f"{self.get_total_hpwl():.6f}\n")
                file.write(f"{self.get_area():.6f}\n")
                file.write(f"{self.get_width():.6f} {self.get_height():.6f}\n")
                for block in self.blocks:
                    file.write(f"{block.name} ")
                    for name in block.merge_device_names:
                        file.write(f"{name} ")
                    file.write(f"{block.get_min_x():.6f} {block.get_min_y():.6f} ({block.get_width():.6f} {block.get_height():.6f} {block.GetColumnMultiple()} {block.GetRowMultiple()})\n")
            return True
        except Exception:
            return False

    def Clear(self):
        self.topology_btrees.clear()
        self.units.clear()
        self.blocks.clear()
        self.nets.clear()
        self.min_x = 0.0
        self.min_y = 0.0
        self.max_x = 0.0
        self.max_y = 0.0
        self.root = None

    def get_width(self):
        return self.max_x - self.min_x

    def get_height(self):
        return self.max_y - self.min_y

    def get_area(self):
        return self.get_width() * self.get_height()

    def get_total_hpwl(self):
        total_hpwl = 0.0
        for _, blocks in self.nets:
            x0 = x1 = blocks[0].GetCenterX()
            y0 = y1 = blocks[0].GetCenterY()
            for block in blocks[1:]:
                xc = block.GetCenterX()
                yc = block.GetCenterY()
                x0 = min(x0, xc)
                x1 = max(x1, xc)
                y0 = min(y0, yc)
                y1 = max(y1, yc)
            w = x1 - x0
            h = y1 - y0
            total_hpwl += w + h
        return total_hpwl

    def get_min_x(self):
        return self.min_x

    def get_max_x(self):
        return self.max_x

    def get_min_y(self):
        return self.min_y

    def get_max_y(self):
        return self.max_y

    def set_min_x(self, x):
        self.min_x = x
        return self

    def set_min_y(self, y):
        self.min_y = y
        return self

    def set_max_x(self, x):
        self.max_x = x
        return self

    def set_max_y(self, y):
        self.max_y = y
        return self


    def save_placement(self):
        # Uloží aktuální pozice bloků
        return [(block.name, block.min_x, block.min_y, block.max_x, block.max_y) for block in self.blocks]

    def load_placement(self, placement):
        # Obnoví pozice bloků ze záznamu
        name_to_block = {block.name: block for block in self.blocks}
        for name, min_x, min_y, max_x, max_y in placement:
            block = name_to_block[name]
            block.min_x = min_x
            block.min_y = min_y
            block.max_x = max_x
            block.max_y = max_y

    def optimize(self, cost_fn, timer, patience=10000000):
        placement = self.save_placement()
        perturbator = Perturbator(self)
        packor = Packor(self)
        cost = cost_fn(self)
        reject_count = 0

        while not timer.is_timeout() and reject_count < patience:
            # print(f"aaan\n")
            perturbator()
            packor()
            new_cost = cost_fn(self)
            if new_cost < cost:
                cost = new_cost
                placement = self.save_placement()
                reject_count = 0
            else:
                self.load_placement(placement)
                reject_count += 1

        Packor(self)()  # Znovu umístí bloky podle stromu !!! added by dba



################################################################################
def visualize_tree(root):

    fig, ax = plt.subplots()
    arrow_length_init = 2.0
    angle_left = -40
    angle_right = 40

    def plot_node(node, x, y, depth, arrow_length, label_side="right"):
        # Oranžové kolečko
        circle = plt.Circle((x, y), 0.2, color='orange', ec='black', zorder=2)
        ax.add_patch(circle)
        # Popisek vlevo/vpravo od kolečka
        if label_side == "left":
            ax.text(x + 0.3, y, node.name, ha='left', va='center', fontsize=8, zorder=3)
        else:
            ax.text(x - 0.3, y, node.name, ha='right', va='center', fontsize=8, zorder=3)
        # l_child: doleva a dolů (-40°), modrá šipka, popisek vlevo
        if hasattr(node, 'l_child') and node.l_child:
            dx = arrow_length * math.cos(math.radians(angle_left)) * 0.9
            dy = -arrow_length * math.sin(math.radians(angle_left)) * 0.9
            x_l = x + dx
            y_l = y - abs(dy)
            ax.arrow(x, y, dx, -abs(dy), head_width=0.15, head_length=0.2, fc='green', ec='green', length_includes_head=True)
            plot_node(node.l_child, x_l, y_l, depth + 1, arrow_length * 0.9, label_side="left")
        # r_child: doprava a dolů (+40°), zelená šipka, popisek vpravo
        if hasattr(node, 'r_child') and node.r_child:
            dx = -arrow_length * math.cos(math.radians(angle_right)) * 0.9
            dy = -arrow_length * math.sin(math.radians(angle_right)) * 0.9
            x_r = x + dx
            y_r = y - abs(dy)
            ax.arrow(x, y, dx, -abs(dy), head_width=0.15, head_length=0.2, fc='blue', ec='blue', length_includes_head=True)
            plot_node(node.r_child, x_r, y_r, depth + 1, arrow_length * 0.9, label_side="right")

    plot_node(root, 0, 0, 1, arrow_length_init, label_side="right")
    ax.set_aspect('equal')
    ax.axis('off')
    plt.title("Vizualizace stromu (kořen nahoře)")
    plt.show()

def visualize_block_positions(blocks):
    fig, ax = plt.subplots()
    min_x = min(block.min_x for block in blocks)
    max_x = max(block.min_x + block.width for block in blocks)
    min_y = min(block.min_y for block in blocks)
    max_y = max(block.min_y + block.height for block in blocks)
    for block in blocks:
        x = block.min_x
        y = block.min_y
        w = block.width
        h = block.height
        rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='blue')
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, block.name, ha='center', va='center', fontsize=8)
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_aspect('equal')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title("Blocks positions")
    plt.show()

def visualize_blocksand_interconnections(blocks, nets):
    fig, ax = plt.subplots()
    # vykreslení bloků
    for block in blocks:
        x, y, w, h = block.min_x, block.min_y, block.width, block.height
        rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='blue')
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, block.name, ha='center', va='center', fontsize=8)
    # vykreslení propojení
    for net_name, net_blocks in nets:
        centers = [(b.min_x + b.width/2, b.min_y + b.height/2) for b in net_blocks]
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                x0, y0 = centers[i]
                x1, y1 = centers[j]
                ax.plot([x0, x1], [y0, y1], color='green', linewidth=1, alpha=0.6)
    ax.set_aspect('equal')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title("Blocks and interconnections")
    plt.show()

def visualize_tree_and_blocks(root, blocks):
    fig, (ax_blocks, ax_tree) = plt.subplots(1, 2, figsize=(12, 6))

    # --- Bloky vlevo ---
    min_x = min(block.min_x for block in blocks)
    max_x = max(block.min_x + block.width for block in blocks)
    min_y = min(block.min_y for block in blocks)
    max_y = max(block.min_y + block.height for block in blocks)
    for block in blocks:
        x = block.min_x
        y = block.min_y
        w = block.width
        h = block.height
        rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='blue')
        ax_blocks.add_patch(rect)
        ax_blocks.text(x + w / 2, y + h / 2, block.name, ha='center', va='center', fontsize=8)
    ax_blocks.set_xlim(min_x, max_x)
    ax_blocks.set_ylim(min_y, max_y)
    ax_blocks.set_aspect('equal')
    ax_blocks.set_title("Blocks positions")
    ax_blocks.set_xlabel('X')
    ax_blocks.set_ylabel('Y')

    # --- Strom vpravo ---
    arrow_length_init = 2.0
    angle_left = -40
    angle_right = 40

    def plot_node(node, x, y, depth, arrow_length, label_side="right"):
        circle = plt.Circle((x, y), 0.2, color='orange', ec='black', zorder=2)
        ax_tree.add_patch(circle)
        if label_side == "left":
            ax_tree.text(x + 0.3, y, node.name, ha='left', va='center', fontsize=8, zorder=3)
        else:
            ax_tree.text(x - 0.3, y, node.name, ha='right', va='center', fontsize=8, zorder=3)
        if hasattr(node, 'l_child') and node.l_child:
            dx = arrow_length * math.cos(math.radians(angle_left)) * 0.9
            dy = -arrow_length * math.sin(math.radians(angle_left)) * 0.9
            x_l = x + dx
            y_l = y - abs(dy)
            ax_tree.arrow(x, y, dx, -abs(dy), head_width=0.15, head_length=0.2, fc='green', ec='green', length_includes_head=True)
            plot_node(node.l_child, x_l, y_l, depth + 1, arrow_length * 0.9, label_side="left")
        if hasattr(node, 'r_child') and node.r_child:
            dx = -arrow_length * math.cos(math.radians(angle_right)) * 0.9
            dy = -arrow_length * math.sin(math.radians(angle_right)) * 0.9
            x_r = x + dx
            y_r = y - abs(dy)
            ax_tree.arrow(x, y, dx, -abs(dy), head_width=0.15, head_length=0.2, fc='blue', ec='blue', length_includes_head=True)
            plot_node(node.r_child, x_r, y_r, depth + 1, arrow_length * 0.9, label_side="right")

    plot_node(root, 0, 0, 1, arrow_length_init, label_side="right")
    ax_tree.set_aspect('equal')
    ax_tree.axis('off')
    ax_tree.set_title("Tree structure")

    plt.tight_layout()
    plt.show()

def visualize_tree_and_blocks2(root, blocks):
    fig, (ax_blocks, ax_tree) = plt.subplots(1, 2, figsize=(12, 6))

    # --- Bloky vlevo ---
    min_x = min(block.min_x for block in blocks)
    max_x = max(block.min_x + block.width for block in blocks)
    min_y = min(block.min_y for block in blocks)
    max_y = max(block.min_y + block.height for block in blocks)
    for block in blocks:
        x = block.min_x
        y = block.min_y
        w = block.width
        h = block.height
        rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='blue')
        ax_blocks.add_patch(rect)
        ax_blocks.text(x + w / 2, y + h / 2, block.name, ha='center', va='center', fontsize=8)
    ax_blocks.set_xlim(min_x, max_x)
    ax_blocks.set_ylim(min_y, max_y)
    ax_blocks.set_aspect('equal')
    ax_blocks.set_title("Blocks positions")
    ax_blocks.set_xlabel('X')
    ax_blocks.set_ylabel('Y')

    # --- Strom vpravo ---
    def plot_node(node, x, y, depth, step=2.0, color='orange'):
        circle = plt.Circle((x, y), 0.2, color=color, ec='black', zorder=2)
        ax_tree.add_patch(circle)
        ax_tree.text(x - 0.4, y - 0.4, node.name, ha='center', va='bottom', fontsize=10, zorder=3)
        # l_child: zelená šipka doprava
        if hasattr(node, 'l_child') and node.l_child:
            x_l = x + step
            y_l = y
            ax_tree.arrow(x, y, step, 0, head_width=0.15, head_length=0.5, fc='green', ec='green', length_includes_head=True)
            plot_node(node.l_child, x_l, y_l, depth + 1, step)
        # r_child: modrá šipka nahoru
        if hasattr(node, 'r_child') and node.r_child:
            x_r = x
            y_r = y + step
            ax_tree.arrow(x, y, 0, step, head_width=0.15, head_length=0.5, fc='blue', ec='blue', length_includes_head=True)
            plot_node(node.r_child, x_r, y_r, depth + 1, step)

    # First point with different color from the others
    plot_node(root, 0, 0, 1, color='darkgreen')
    ax_tree.set_aspect('equal')
    ax_tree.axis('off')
    ax_tree.set_title("Tree structure")

    plt.tight_layout()
    plt.show()

################################################################################
if __name__ == "__main__":
    ############################################################################
    #NETLIST_FILE = "C:\\Users\\wolfg\\iCloudDrive\\Documents\\CVUT\\Diplomka\\Codes\\LAB1_py_v1\\case1.netlist"
    #SYMMETRY_FILE = "C:\\Users\\wolfg\\iCloudDrive\\Documents\\CVUT\\Diplomka\\Codes\\LAB1_py_v1\\case1.sym"
    #BLOCK_FILE = "C:\\Users\\wolfg\\iCloudDrive\\Documents\\CVUT\\Diplomka\\Codes\\LAB1_py_v1\\case1.block"

    #SET VARIABLE PATH FOR BETTER PORTABILITY
    import os

    basename = "test"  # Change to your file base name

    NETLIST_FILE = os.path.join(os.path.dirname(__file__), f"{basename}.netlist")
    SYMMETRY_FILE = os.path.join(os.path.dirname(__file__), f"{basename}.sym")
    BLOCK_FILE = os.path.join(os.path.dirname(__file__), f"{basename}.block")

    ############################################################################
    tree = top_BStarTree()
    tree.Load(NETLIST_FILE, SYMMETRY_FILE, BLOCK_FILE)

    ############################################################################
    # visualize_block_positions(tree.blocks)
    # visualize_blocksand_interconnections(tree.blocks, tree.nets)
    # visualize_tree(tree.root)
    # visualize_bstartree_graph(tree.blocks, tree.asfbtrees_)
    # print(f"Počet stromů: {len(tree.asfbtrees_)}")
    # for t in tree.asfbtrees_:
    #     print(f"Strom {t.name} má {len(t.units)} uzlů")

    ############################################################################
    # Volání optimalizace, pokud existuje
    # Pokud není metoda optimize, pouze vypočítá cost
    import threading
    from utils import PA3Cost
    from utils import Timer  # nebo vlastní implementace Timeru

    expected_aspect_ratio = 2
    cost_func = PA3Cost(expected_aspect_ratio)
    timer = Timer(timeout=1)  # nastavte vhodný timeout

    mtx = threading.Lock()

    i=1
    if hasattr(tree, "optimize"):
        tree.optimize(cost_func, timer)
    with mtx:
        print(f"Thread-{i:03} -> Cost={cost_func(tree):.6f}")

    ############################################################################
    # visualize_block_positions(tree.blocks)
    # visualize_blocksand_interconnections(tree.blocks, tree.nets)
    # visualize_tree(tree.root)
    visualize_tree_and_blocks2(tree.root, tree.blocks)
    ############################################################################
    print(f"Optimized tree")

# dba TODO:
#  1) tree maybe is not correctly set, because only one element is visualized
#     during tree vizualization. Is needed to understand the tree structure.
#  2) optimize function - maybe simulated annealing
#  3) cost function - area + wirelength + aspect ratio
#  4) packing - maybe better algorithm
#  5) perturbation - maybe better algorithm

