import ast
import astor
import copy


class __ParentMap__(ast.NodeTransformer):
    # current parent (module)
    parent = None

    def visit(self, node):
        # set parent attribute for this node
        node.parent = self.parent
        # This node becomes the new parent
        self.parent = node
        # Do any work required by super class
        node = super().visit(node)
        # If we have a valid node (ie. node not being removed)
        if isinstance(node, ast.AST):
            # update the parent, since this may have been transformed
            # to a different node by super
            self.parent = node.parent
        return node

    def get_parent(self, node):
        return node.parent


def __get_parent_map__(tree: ast.AST) -> __ParentMap__:
    parent_map = __ParentMap__()
    parent_map.visit(tree)
    return parent_map


class __SuccessorVisitor__(ast.NodeVisitor):
    def __init__(self, successor_node):
        super().__init__()
        self.find = False
        self.succeddor_node = successor_node

    def visit(self, node):
        if node == self.succeddor_node:
            self.find = True
        self.generic_visit(node)


def __is_successor__(node, root):
    searcher = __SuccessorVisitor__(node)
    searcher.visit(root)
    return searcher.find


class __FuncVisitor__(ast.NodeVisitor):
    def __init__(self, tree: ast.AST) -> None:
        super().__init__()
        self.definitions: list[ast.FunctionDef] = []
        self.calls: list[ast.Call] = []
        self.tree = tree
        self.visit(self.tree)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.definitions.append(node)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if (node.func.id != 'range'):
            self.calls.append(node)
        self.generic_visit(node)


class __NameTailer__(ast.NodeVisitor):
    def __init__(self, block_index) -> None:
        super().__init__()
        self.block_index = block_index

    def visit_Name(self, node: ast.Name) -> None:
        node.id = node.id if node.id == 'range' else node.id + ('_' + str(self.block_index))
        self.generic_visit(node)


class __BodyDetector__(ast.NodeVisitor):
    def __init__(self, tree, target_node) -> None:
        super().__init__()
        self.tree = tree
        self.targte_node = target_node
        self.is_in_body = False

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if __is_successor__(self.targte_node, node):
            self.is_in_body = True
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        if __is_successor__(self.targte_node, node):
            self.is_in_body = True
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        if __is_successor__(self.targte_node, node):
            self.is_in_body = True
        self.generic_visit(node)

    def detect(self) -> bool:
        self.visit(self.tree)
        return self.is_in_body


class __ParentBodyFinder__():
    def __init__(self, tree: ast.AST, child: ast.AST) -> None:
        self.tree = tree
        self.child = child
        self.parent = __get_parent_map__(self.tree)

    def find_parent_body(self):
        node = self.child
        while not hasattr(self.parent.get_parent(node), 'body'):
            node = self.parent.get_parent(node)
        return self.parent.get_parent(node)


class FuncUnroller(ast.NodeTransformer):
    def __init__(self, filename: str) -> None:
        super().__init__()
        self.tree = astor.parse_file(filename)
        self.code = astor.to_source(self.tree)
        self.definitions: list[ast.FunctionDef] = __FuncVisitor__(self.tree).definitions
        self.calls: list[ast.Call] = __FuncVisitor__(self.tree).calls
        self.tail = 1
        # print(astor.dump_tree(self.definitions))
        # print(astor.dump_tree(self.calls))

    def __get_definition__(self, node: ast.Call):
        # get definition, actual parameters, formal parameters
        for definition in self.definitions:
            if definition.name == node.func.id:
                return definition, node.keywords, definition.args.args
        return None, None, None

    def __index_in_module__(self, node):
        for statement in self.tree.body:
            if __is_successor__(node, statement):
                return self.tree.body.index(statement)

    def __get_return_body__(self, node: ast.Call, expression: ast.Tuple):
        parent = __get_parent_map__(self.tree)
        tagerts = parent.get_parent(node).targets[0].elts
        values = expression.elts
        return_body = []
        for i in range(len(expression.elts)):
            __NameTailer__(self.tail).visit(values[i])
            statement = ast.Assign(targets=[tagerts[i]], value=values[i])
            return_body.append(statement)
        return return_body

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if node.func.id == 'range':
            return node
        definition, actual_parameters, formal_parameters = self.__get_definition__(node)
        assert definition is not None, "call: '{}' has no definition".format(node.func.id)
        if type(definition.body[-1]) is ast.Return:
            sub_body = copy.deepcopy(definition.body[:-1])
            for statement in sub_body:
                __NameTailer__(self.tail).visit(statement)
            for i in range(len(actual_parameters)):
                statement = ast.Assign(targets=[ast.Name(id=actual_parameters[i].arg + '_' + str(self.tail))],
                                       value=actual_parameters[i].value)
                sub_body.insert(0, statement)
                # print(astor.dump_tree(statement))
            # print(astor.dump_tree(sub_body))
            expression = definition.body[-1].value
            if type(expression) is ast.Tuple:
                return_body = self.__get_return_body__(node, expression)
                if __BodyDetector__(self.tree, node).detect():
                    parent = __ParentBodyFinder__(self.tree, node).find_parent_body()
                    parent.body[-1].targets = [parent.body[-1].targets[0].elts[-1]]
                    body = parent.body[:-1] + sub_body + return_body
                    parent.body = body
                    self.tail += 1
                    return ast.copy_location(return_body[-1].value, node)
                else:
                    index = self.__index_in_module__(node)
                    self.tree.body[index].targets = [self.tree.body[index].targets[0].elts[-1]]
                    body = self.tree.body[:index] + sub_body + return_body + self.tree.body[index + 1:]
                    self.tree.body = body
                    self.tail += 1
                    return ast.copy_location(return_body[-1].value, node)
            else:
                if __BodyDetector__(self.tree, node).detect():
                    parent = __ParentBodyFinder__(self.tree, node).find_parent_body()
                    body = parent.body[:-1] + sub_body + [parent.body[-1]]
                    parent.body = body
                else:
                    index = self.__index_in_module__(node)
                    body = self.tree.body[:index] + sub_body + self.tree.body[index:]
                    self.tree.body = body
                __NameTailer__(self.tail).visit(expression)
                self.tail += 1
                return ast.copy_location(expression, node)
        else:
            return node

    def view_tree(self):
        print(astor.dump_tree(self.tree))

    def view_code(self):
        self.code = astor.to_source(self.tree)
        print(self.code)
        return self.code

    def unroll(self):
        if len(self.tree.body) == 1 and type(self.tree.body[0]) is ast.FunctionDef:
            definition = self.tree.body[0]
            body = definition.body
            if type(body[-1]) is ast.Return:
                body.pop()
            args = definition.args.args
            for arg in args:
                statement = ast.AnnAssign(target=ast.Name(id=arg.arg), annotation=arg.annotation,
                                          value=ast.Constant(value=0), simple=1)
                body.insert(0, statement)
            self.tree.body = body
        else:
            self.visit(self.tree)
            self.tree.body = list(filter(lambda x: (type(x) is not ast.FunctionDef), self.tree.body))
        # self.view_code()
        # self.view_tree()


def main():
    filename = 'testcode.py'
    unroller = FuncUnroller(filename)
    unroller.unroll()


if __name__ == '__main__':
    main()