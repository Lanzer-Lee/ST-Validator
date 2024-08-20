import os
import qianfan
import re
import astor
import ast
import src.lib.interpreter as interpreter
import ply.yacc as yacc
import ply.lex as lex
from src.lib.adaptor import FuncUnroller
import subprocess


class ErnieAPI:
    def __init__(self) -> None:
        os.environ["QIANFAN_AK"] = "your key"
        os.environ["QIANFAN_SK"] = "your key"
        self.response = None
        self.code = None

    def ask(self, prompt: str) -> None:
        self.response = qianfan.ChatCompletion().do(endpoint="ernie-lite-8k",
                                                    messages=[{"role": "user", "content": prompt}],
                                                    temperature=0.95, top_p=0.7, penalty_score=1)
        self.extract_code()

    def extract_code(self):
        self.code = re.findall(r'```st([^`]*)```', self.response.body['result'])[0]
        with open('output/st-code.ST', 'w') as file:
            print(self.code, file=file)
        print(self.code)


class Interpreter:
    def __init__(self, path) -> None:
        self.path = path
        self.tree = None

    def __call__(self) -> ast.Module:
        with open(self.path, 'r') as file:
            code = file.read()
            lexer = lex.lex(module=interpreter)
            lexer.input(code)
            parser = yacc.yacc(module=interpreter)
            tree = parser.parse(code)
            return ast.Module(body=tree) if type(tree) is list else ast.Module(body=[tree])

    def __str__(self):
        return astor.dump_tree(self.tree)

    def __repr__(self):
        return astor.to_source(self.tree)

    def interpret(self):
        self.tree = self.__call__()
        with open('output/python-code.py', 'w') as file:
            print(self.__repr__(), file=file)
        print(self.__str__())


class Unroller:
    def __init__(self, path) -> None:
        self.path = path
        self.code = None

    def __call__(self):
        unroller = FuncUnroller(self.path)
        unroller.unroll()
        return unroller.view_code()

    def __str__(self):
        return self.code

    def unroll(self):
        self.code = self.__call__()
        with open('output/unrolled-python-code.py', 'w') as file:
            print('from pynuxmv.main import *\n', file=file)
            print(self.__str__(), file=file)
        print('\nunrolled code:\n')
        print(self.__str__())


class WindowsCMD:
    def __init__(self) -> None:
        self.output = None
        self.error = None

    def run_command(self, command):
        result = subprocess.run(command, capture_output=True, text=True)
        self.output = result.stdout
        self.error = result.stderr


class PyNuXMV(WindowsCMD):
    def __init__(self, input_path='output/unrolled-python-code.py', output_path='output/model.smv') -> None:
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path

    def build(self):
        self.run_command('bin/pynuXmv '+self.input_path+' '+self.output_path)

