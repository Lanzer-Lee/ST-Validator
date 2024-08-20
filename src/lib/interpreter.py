import ast
import astor
import math
import re

# List of token names.   This is always required
tokens = [
    'ASSIGN',
    'COLON',
    'SEMICOLON',
    'EQ',
    'PLUS',
    'MINUS',
    'TIMES',
    'DIVIDE',
    'LPAREN',
    'RPAREN',
    'COMMA',
    'LT',
    'GT',
    'LE',
    'GE',
    'DOT',
    'LBRACE',
    'RBRACE',
    'SHARP',
    'ID',
    'COMMENT',
    'MULTI_COMMENT',
    'INT_NUMBER',
    'FLOAT_NUMBER',
    'STRING_NUMBER'
]

reserved = {
    'VAR': 'VAR',
    'END_VAR': 'END_VAR',
    'CONST': 'CONST',
    'END_CONST': 'END_CONST',
    'IF': 'IF',
    'END_IF': 'END_IF',
    'THEN': 'THEN',
    'ELSE': 'ELSE',
    'ELSIF': 'ELSIF',
    'WHILE': 'WHILE',
    'END_WHILE': 'END_WHILE',
    'DO': 'DO',
    'FOR': 'FOR',
    'END_FOR': 'END_FOR',
    'TO': 'TO',
    'INT': 'INT',
    'FLOAT': 'FLOAT',
    'BOOL': 'BOOL',
    'FALSE': 'FALSE',
    'TRUE': 'TRUE',
    'AND': 'AND',
    'OR': 'OR',
    'XOR': 'XOR',
    'NOT': 'NOT',
    'BY': 'BY',
    'FUNCTION_BLOCK': 'FUNCTION_BLOCK',
    'END_FUNCTION_BLOCK': 'END_FUNCTION_BLOCK',
    'VAR_INPUT': 'VAR_INPUT',
    'VAR_OUTPUT': 'VAR_OUTPUT',
    'PROGRAM': 'PROGRAM',
    'END_PROGRAM': 'END_PROGRAM',
    'BEGIN': 'BEGIN',
    'REAL': 'REAL',
    'VAR_IN_OUT': 'VAR_IN_OUT',
    'FUNCTION': 'FUNCTION',
    'END_FUNCTION': 'END_FUNCTION',
    'VAR_GLOBAL': 'VAR_GLOBAL',
    'STRING': 'STRING',
    'LOG': 'LOG',
    'EXP': 'EXP',
    'BYTE': 'BYTE',
    'LN': 'LN',
    'TON': 'TON',
    'TP': 'TP',
    'TOF': 'TOF',
    'RTC': 'RTC',
    'TIME': 'TIME',
    'CASE': 'CASE',
    'END_CASE': 'END_CASE',
    'OF': 'OF',
    'DWORD': 'DWORD',
    'EQ_STRING': 'EQ_STRING'
}

tokens = tokens + list(reserved.values())

# Regular expression rules for simple tokens
t_ASSIGN = r':='
t_COLON = r':'
t_SEMICOLON = r';'
t_EQ = r'='
t_PLUS = r'\+'
t_MINUS = r'-'
t_TIMES = r'\*'
t_DIVIDE = r'/'
t_LPAREN = r'\('
t_RPAREN = r'\)'
t_COMMA = r','
t_LT = r'<'
t_GT = r'>'
t_LE = r'<='
t_GE = r'>='
t_DOT = r'\.'
t_LBRACE = r'\{'
t_RBRACE = r'\}'
t_SHARP = r'\#'
# A string containing ignored characters (spaces and tabs)
t_ignore = ' \t'

precedence = (
    ("left", "PLUS", "MINUS"),
    ("left", "TIMES", "DIVIDE"),
    ("left", "LPAREN", "RPAREN"),
    ("right", "NOT", "ASSIGN"),
    ("left", "AND", "OR"),
    ("left", "LT", "GT", "LE", "GE", "EQ"),
    ("left", "VAR", "END_VAR", "CONST", "END_CONST"),
    ("left", "VAR_INPUT", "VAR_OUTPUT", "VAR_IN_OUT", "VAR_GLOBAL")
)

timer_name = []


def t_COMMENT(t):
    r'\/\/.*'
    pass


def t_MULTI_COMMENT(t):
    r'\(\*[\w\W]*?\*\)'
    pass


def t_ID(t):
    r'[a-zA-Z_][a-zA-Z_0-9]*(\#[+-]?[0-9]+(\.[0-9]+)?([mM])?([sS])?)?'
    t.type = reserved.get(t.value, 'ID')  # Check for reserved words
    t.value = t.value.upper()
    if '#' in t.value:
        number = re.findall(r'[0-9\.+-]+', t.value)
        t.value = number[0]
    return t


# A regular expression rule with some action code
def t_INT_NUMBER(t):
    r'\b[0-9]+(?!\.)\b'
    return t


def t_FLOAT_NUMBER(t):
    r'[0-9]+(\.[0-9]+)([Ee][+-]?[0-9]+)?'
    # r'\d+(\.\d+)'
    return t


def t_STRING_NUMBER(t):
    r'\".*\"'
    return t


# Define a rule so we can track line numbers
def t_newline(t):
    r'\n+'
    t.lexer.lineno += len(t.value)


# Error handling rule
def t_error(t):
    print("Illegal character '%s'" % t.value[0])
    t.lexer.skip(1)


def p_module(p):
    '''module : program
              | function_block_definition
              | function_definition
              | module function_block_definition
              | module function_definition
              | module program
    '''
    if (len(p) == 2):
        p[0] = p[1] if type(p[1]) is list else [p[1]]
    elif (len(p) == 3):
        p[0] = p[2] + p[1] if type(p[2]) is list else [p[2]] + p[1]


def p_program(p):
    '''program : PROGRAM ID var_list BEGIN statement_list END_PROGRAM
               | PROGRAM ID var_list statement_list END_PROGRAM
    '''
    var_list = p[3]
    p[0] = []
    for statement in var_list[1]:
        p[0] += statement
    p[0] = p[0] + p[4] if len(p) == 6 else p[0] + p[5]


def p_function_block_definition(p):
    '''function_block_definition : FUNCTION_BLOCK ID var_list BEGIN statement_list END_FUNCTION_BLOCK
                                 | FUNCTION_BLOCK ID var_list statement_list END_FUNCTION_BLOCK
    '''
    # name = ID
    name = p[2].upper()
    # body = statement_list
    body = p[5] if len(p) == 7 else p[4]
    # get different type declare position in var list
    len_output = 0
    len_in_out = 0
    len_input = 0
    len_global = 0
    len_var = 0
    len_const = 0
    var_list = p[3]
    for index, type in enumerate(var_list[0]):
        if type == 'output':
            len_output = len(var_list[1][index])
            index_output = index
        elif type == 'in_out':
            len_in_out = len(var_list[1][index])
            index_in_out = index
        elif type == 'var':
            len_var = len(var_list[1][index])
            index_var = index
        elif type == 'input':
            len_input = len(var_list[1][index])
            index_input = index
        elif type == 'global':
            len_global = len(var_list[1][index])
            index_global = index
        elif type == 'const':
            len_const = len(var_list[1][index])
            index_const = index
    # var_output and var_in_out contribute to return statement
    if (len_output == 1 and len_in_out == 0):
        statement: ast.AnnAssign = var_list[1][index_output][0]
        return_statement = ast.Return(value=statement.target)
    elif (len_in_out == 1 and len_output == 0):
        statement: ast.AnnAssign = var_list[1][index_in_out][0]
        return_statement = ast.Return(value=statement.target)
    elif (len_in_out + len_output > 1):
        elts = []
        if len_output:
            statement_list: list[ast.AnnAssign] = var_list[1][index_output]
            for statement in statement_list:
                elts.append(statement.target)
        if len_in_out:
            statement_list: list[ast.AnnAssign] = var_list[1][index_in_out]
            for statement in statement_list:
                elts.append(statement.target)
        return_statement = ast.Return(value=ast.Tuple(elts=elts))
    else:
        return_statement = None
    body.append(return_statement)
    if len_var:
        for statement in var_list[1][index_var]:
            body.insert(0, statement)
    if len_const:
        for statement in var_list[1][index_const]:
            body.insert(0, statement)
    # var_input and var_in_out contribute to argument
    args = []
    if len_input:
        statement_list: list[ast.AnnAssign] = var_list[1][index_input]
        for statement in statement_list:
            arg = ast.arg(arg=statement.target.id, annotation=statement.annotation)
            args.append(arg)
    if len_in_out:
        statement_list: list[ast.AnnAssign] = var_list[1][index_in_out]
        for statement in statement_list:
            arg = ast.arg(arg=statement.target.id, annotation=statement.annotation)
            args.append(arg)
    args = ast.arguments(posonlyargs=[], args=args, defaults=[])
    p[0] = ast.FunctionDef(name=name, args=args, body=body, decorator_list=[], returns=None)


def p_function_definition(p):
    '''function_definition : FUNCTION ID COLON INT var_list statement_list END_FUNCTION
                           | FUNCTION ID COLON BOOL var_list statement_list END_FUNCTION
                           | FUNCTION ID COLON REAL var_list statement_list END_FUNCTION
                           | FUNCTION ID COLON FLOAT var_list statement_list END_FUNCTION
                           | FUNCTION ID COLON BYTE var_list statement_list END_FUNCTION
    '''
    type = p[4]
    if type == 'INT' or type == 'BOOL' or type == 'FLOAT' or type == 'BYTE':
        returns = ast.Name(id='int')  # only support int
    elif type == 'REAL':
        returns = ast.Name(id='int')
    else:
        returns = None
    name = p[2].upper()
    body: list = p[6]
    # get different type declare position in var list
    len_output = 0
    len_in_out = 0
    len_input = 0
    len_global = 0
    len_var = 0
    len_const = 0
    var_list = p[5]
    for index, type in enumerate(var_list[0]):
        if type == 'output':
            len_output = len(var_list[1][index])
            index_output = index
        elif type == 'in_out':
            len_in_out = len(var_list[1][index])
            index_in_out = index
        elif type == 'var':
            len_var = len(var_list[1][index])
            index_var = index
        elif type == 'input':
            len_input = len(var_list[1][index])
            index_input = index
        elif type == 'global':
            len_global = len(var_list[1][index])
            index_global = index
        elif type == 'const':
            len_const = len(var_list[1][index])
            index_const = index
    return_statement = ast.Return(value=ast.Name(id=name.upper()))
    body.append(return_statement)
    if len_var:
        for statement in var_list[1][index_var]:
            body.insert(0, statement)
    if len_const:
        for statement in var_list[1][index_const]:
            body.insert(0, statement)
    args = []
    if len_input:
        statement_list: list[ast.AnnAssign] = var_list[1][index_input]
        for statement in statement_list:
            arg = ast.arg(arg=statement.target.id, annotation=statement.annotation)
            args.append(arg)
    if len_in_out:
        statement_list: list[ast.AnnAssign] = var_list[1][index_in_out]
        for statement in statement_list:
            arg = ast.arg(arg=statement.target.id, annotation=statement.annotation)
            args.append(arg)
    args = ast.arguments(posonlyargs=[], args=args, defaults=[])
    p[0] = ast.FunctionDef(name=name, args=args, body=body, decorator_list=[], returns=returns)


def p_var_list(p):
    '''var_list : var_input
                | var_output
                | var
                | var_in_out
                | var_global
                | const
                | var_list var_input
                | var_list var_output
                | var_list var
                | var_list var_in_out
                | var_list var_global
                | var_list const
    '''
    if (len(p) == 2):
        p[0] = p[1]
    elif (len(p) == 3):
        p[0] = [p[1][0] + p[2][0], p[1][1] + p[2][1]]


def p_var_output(p):
    '''var_output : VAR_OUTPUT declare_list END_VAR'''
    p[0] = [['output'], [p[2]]]


def p_var_input(p):
    '''var_input : VAR_INPUT declare_list END_VAR'''
    p[0] = [['input'], [p[2]]]


def p_var_in_out(p):
    '''var_in_out : VAR_IN_OUT declare_list END_VAR'''
    p[0] = [['in_out'], [p[2]]]


def p_var(p):
    '''var : VAR declare_list END_VAR'''
    p[0] = [['var'], [p[2]]]


def p_const(p):
    '''const : CONST declare_list END_CONST'''
    p[0] = [['const'], [p[2]]]


def p_var_global(p):
    '''var_global : VAR_GLOBAL declare_list END_VAR'''
    p[0] = [['global'], [p[2]]]


def p_statement_list(p):
    '''statement_list :
                      | statement_list assign_statement
                      | statement_list while_statement
                      | statement_list if_statement
                      | statement_list for_statement
                      | statement_list call_statement
                      | statement_list case_statement
    '''
    if (len(p) == 1):
        p[0] = []
    elif (len(p) == 3):
        if type(p[2]) is list:
            p[0] = p[1] + p[2]
        else:
            p[1].append(p[2])
            p[0] = p[1]


def p_declare_list(p):
    '''declare_list :
                    | declare_list declare_int_statement
                    | declare_list declare_float_statement
                    | declare_list declare_timer_statement
    '''
    if (len(p) == 1):
        p[0] = []
    elif (len(p) == 3):
        if type(p[2]) is list:
            p[0] = p[1] + p[2]
        else:
            p[1].append(p[2])
            p[0] = p[1]


def p_declare_timer_statement(p):
    '''declare_timer_statement : ID COLON TP SEMICOLON
                               | ID COLON TON SEMICOLON
                               | ID COLON TOF SEMICOLON
                               | ID COLON RTC SEMICOLON
    '''
    if len(p) == 5:
        timer_name.append(p[1])
        body = []
        statement = ast.AnnAssign(target=ast.Name(id=p[1] + '_IN'), annotation=ast.Name(id='int'),
                                  value=ast.Constant(value=0), simple=1)
        body.append(statement)
        statement = ast.AnnAssign(target=ast.Name(id=p[1] + '_PT'), annotation=ast.Name(id='int'),
                                  value=ast.Constant(value=0), simple=1)
        body.append(statement)
        statement = ast.AnnAssign(target=ast.Name(id=p[1] + '_Q'), annotation=ast.Name(id='int'),
                                  value=ast.Constant(value=0), simple=1)
        body.append(statement)
        statement = ast.AnnAssign(target=ast.Name(id=p[1] + '_ET'), annotation=ast.Name(id='int'),
                                  value=ast.Constant(value=0), simple=1)
        body.append(statement)
        p[0] = body


def p_declare_int_statement(p):
    '''declare_int_statement : ID COLON INT ASSIGN int_number SEMICOLON
                             | ID COLON BOOL ASSIGN int_number SEMICOLON
                             | ID COLON BYTE ASSIGN int_number SEMICOLON
                             | ID COLON TIME ASSIGN int_number SEMICOLON
                             | ID COLON DWORD ASSIGN int_number SEMICOLON
                             | ID COLON STRING ASSIGN string_number SEMICOLON
                             | ID COLON BYTE ASSIGN ID SEMICOLON
                             | ID COLON TIME ASSIGN ID SEMICOLON
                             | ID COLON INT SEMICOLON
                             | ID COLON BOOL SEMICOLON
                             | ID COLON BYTE SEMICOLON
                             | ID COLON TIME SEMICOLON
                             | ID COLON DWORD SEMICOLON
                             | ID COLON STRING SEMICOLON
    '''
    # variable:INT:=number;
    # variable:INT;
    if (len(p) == 7):
        if type(p[5]) is str and p[5].isdigit():
            number = re.findall(r'\d+', p[5])
            p[5] = ast.Constant(value=int(number[0]))
            print(astor.dump_tree(p[5]))
        if (p[3] == 'INT' or p[3] == 'TIME' or p[3] == 'BYTE' or p[3] == 'DWORD' or p[3] == 'STRING'):
            p[0] = ast.AnnAssign(target=ast.Name(id=p[1]), annotation=ast.Name(id='int'), value=p[5], simple=1)
        elif (p[3] == 'BOOL'):
            p[0] = ast.AnnAssign(target=ast.Name(id=p[1]), annotation=ast.Name(id='bool'), value=p[5], simple=1)
    elif (len(p) == 5):
        if (p[3] == 'INT' or p[3] == 'BYTE' or p[3] == 'TIME' or p[3] == 'DWORD' or p[3] == 'STRING'):
            p[0] = ast.AnnAssign(target=ast.Name(id=p[1]), annotation=ast.Name(id='int'), value=ast.Constant(value=0),
                                 simple=1)
        elif (p[3] == 'BOOL'):
            p[0] = ast.AnnAssign(target=ast.Name(id=p[1]), annotation=ast.Name(id='bool'),
                                 value=ast.Constant(value=False), simple=1)


def p_declare_float_statement(p):
    '''declare_float_statement : ID COLON FLOAT ASSIGN float_number SEMICOLON
                               | ID COLON FLOAT SEMICOLON
                               | ID COLON REAL ASSIGN float_number SEMICOLON
                               | ID COLON REAL SEMICOLON
    '''
    # variable:FLOAT:=number;
    # variable:FLOAT;
    if (len(p) == 7):
        # no support for float
        p[0] = ast.AnnAssign(target=ast.Name(id=p[1]), annotation=ast.Name(id='int'), value=p[5], simple=1)
    elif (len(p) == 5):
        # no support for float
        p[0] = ast.AnnAssign(target=ast.Name(id=p[1]), annotation=ast.Name(id='int'), value=ast.Constant(value=0.0),
                             simple=1)


def p_case_statement(p):
    '''case_statement : CASE ID OF case_list END_CASE SEMICOLON'''
    case_list = p[4]
    case_body = []
    for index in range(len(case_list[0])):
        condition = case_list[0][index]
        body = case_list[1][index]
        statement = ast.If(test=condition, body=body, orelse=[])
        case_body.append(statement)
    p[0] = case_body


def p_case_list(p):
    '''case_list : expression COLON statement_list
                 | case_list expression COLON statement_list
    '''
    if len(p) == 4:
        p[0] = [[p[1]], [p[3]]]
    elif len(p) == 5:
        p[1][0].append(p[2])
        p[1][1].append(p[4])
        p[0] = [p[1][0], p[1][1]]


def p_if_statement(p):
    '''if_statement : IF expression THEN statement_list END_IF SEMICOLON
                    | IF expression THEN statement_list elsif_statement_list END_IF SEMICOLON
                    | IF expression THEN statement_list else_statement END_IF SEMICOLON
                    | IF expression THEN statement_list elsif_statement_list else_statement END_IF SEMICOLON
    '''
    if (len(p) == 7):
        p[0] = ast.If(test=p[2], body=p[4], orelse=[])
    elif (len(p) == 8):
        p[0] = ast.If(test=p[2], body=p[4], orelse=[p[5]]) if type(p[5]) is ast.If else ast.If(test=p[2], body=p[4],
                                                                                               orelse=p[5])
    elif (len(p) == 9):
        p[0] = ast.If(test=p[2], body=p[4], orelse=[p[5]])
        node = p[0]
        while (node.orelse != []):
            node = node.orelse[0]
        node.orelse = p[6]


def p_elsif_statement_list(p):
    '''elsif_statement_list : elsif_statement
                            | elsif_statement_list elsif_statement'''
    if (len(p) == 2):
        p[0] = p[1]
    elif (len(p) == 3):
        if (p[1].orelse == []):
            p[0] = ast.If(test=p[1].test, body=p[1].body, orelse=[p[2]])
        else:
            # node=id(p[1])
            # parent=id(node)
            # while(ctypes.cast(node,ast.If)):
            #     parent=id
            #     node=id(ctypes.cast(node,ast.If).orelse[0])
            # value=ast.If(test=p[2].test,body=p[2].body,orelse=[])
            # ctypes.cast(node,ast.If)=value
            # ctypes.cast(parent,ast.If)=node
            node = p[1]
            # parent=node
            while (node.orelse != []):
                # parent=node
                node = node.orelse[0]
            node = ast.If(test=p[2].test, body=p[2].body, orelse=[])
            # parent.orelse=[node]
            p[0] = p[1]


def p_elsif_statement(p):
    '''elsif_statement : ELSIF expression THEN statement_list'''
    p[0] = ast.If(test=p[2], body=p[4], orelse=[])


def p_else_statement(p):
    '''else_statement : ELSE statement_list'''
    p[0] = p[2]


def p_for_statement(p):
    '''for_statement : FOR ID ASSIGN expression TO expression DO statement_list END_FOR SEMICOLON
                     | FOR ID ASSIGN expression TO expression BY expression DO statement_list END_FOR SEMICOLON
    '''
    if (len(p) == 11):
        p[0] = ast.For(target=ast.Name(id=p[2]),
                       iter=ast.Call(func=ast.Name(id='range'), args=[p[4], p[6]], keywords=[]), body=p[8], orelse=[])
    elif (len(p) == 13):
        p[0] = ast.For(target=ast.Name(id=p[2]),
                       iter=ast.Call(func=ast.Name(id='range'), args=[p[4], p[6], p[8]], keywords=[]), body=p[10],
                       orelse=[])


def p_while_statement(p):
    '''while_statement : WHILE expression DO statement_list END_WHILE'''
    p[0] = ast.While(test=p[2], body=p[4], orelse=[])


def p_assign_statement(p):
    '''assign_statement : ID ASSIGN expression SEMICOLON'''
    p[0] = ast.Assign(targets=[ast.Name(id=p[1])], value=p[3])


def p_expression(p):
    '''expression : ID
                  | int_number
                  | float_number
                  | string_number
                  | log
                  | exp
                  | call
                  | eq_string
                  | expression PLUS expression
                  | expression MINUS expression
                  | expression TIMES expression
                  | expression DIVIDE expression
                  | expression AND expression
                  | expression OR expression
                  | expression XOR expression
                  | expression LT expression
                  | expression GT expression
                  | expression LE expression
                  | expression GE expression
                  | expression EQ expression
                  | LPAREN expression RPAREN
                  | NOT expression
                  | MINUS expression
                  | PLUS expression
    '''
    if (len(p) == 2):
        if (type(p[1]) is str):
            p[0] = ast.Name(id=p[1])
        else:
            p[0] = p[1]
            # p[0]=ast.Expr(value=p[1])
    elif (len(p) == 4):
        if (p[2] == '+'):
            p[0] = ast.BinOp(left=p[1], op=ast.Add(), right=p[3])
        elif (p[2] == '-'):
            p[0] = ast.BinOp(left=p[1], op=ast.Sub(), right=p[3])
        elif (p[2] == '*'):
            p[0] = ast.BinOp(left=p[1], op=ast.Mult(), right=p[3])
        elif (p[2] == '/'):
            p[0] = ast.BinOp(left=p[1], op=ast.Div(), right=p[3])
        elif (p[2] == 'AND'):
            p[0] = ast.BoolOp(op=ast.And(), values=[p[1], p[3]])
        elif (p[2] == 'OR'):
            p[0] = ast.BoolOp(op=ast.Or(), values=[p[1], p[3]])
        elif (p[2] == 'XOR'):
            p[0] = ast.BoolOp(op=ast.BitXor(), values=[p[1], p[3]])
        elif (p[2] == '<'):
            p[0] = ast.Compare(left=p[1], ops=[ast.Lt()], comparators=[p[3]])
        elif (p[2] == '>'):
            p[0] = ast.Compare(left=p[1], ops=[ast.Gt()], comparators=[p[3]])
        elif (p[2] == '<='):
            p[0] = ast.Compare(left=p[1], ops=[ast.LtE()], comparators=[p[3]])
        elif (p[2] == '>='):
            p[0] = ast.Compare(left=p[1], ops=[ast.GtE()], comparators=[p[3]])
        elif (p[2] == '='):
            p[0] = ast.Compare(left=p[1], ops=[ast.Eq()], comparators=[p[3]])
        elif (p[1] == '(' and p[3] == ')'):
            p[0] = p[2]
    elif (len(p) == 3):
        if (p[1] == '+'):
            p[0] == ast.UnaryOp(op=ast.UAdd(), operand=p[2])
        elif (p[1] == '-'):
            p[0] = ast.UnaryOp(op=ast.USub(), operand=p[2])
        elif (p[1] == 'NOT'):
            p[0] = ast.UnaryOp(op=ast.Not(), operand=p[2])


def p_call_statement(p):
    '''call_statement : ID argument RPAREN SEMICOLON'''
    if p[1].upper() in timer_name:
        body = []
        for keyword in p[2]:
            statement = ast.Assign(targets=[ast.Name(id=p[1].upper() + '_' + keyword.arg)], value=keyword.value)
            body.append(statement)
        p[0] = body
    else:
        p[0] = ast.Assign(targets=[ast.Name(id=p[1].lower())],
                          value=ast.Call(func=ast.Name(id=p[1].upper()), args=[], keywords=p[2]))
    # p[0]=ast.Expr(value=ast.Call(func=ast.Name(id=p[1]),args=[],keywords=p[2]))


def p_arugument(p):
    '''argument : LPAREN
                | argument ID ASSIGN expression
                | argument COMMA ID ASSIGN expression
    '''
    if (len(p) == 2):
        p[0] = []
    elif (len(p) == 5):
        p[1].append(ast.keyword(arg=p[2], value=p[4]))
        p[0] = p[1]
    elif (len(p) == 6):
        p[1].append(ast.keyword(arg=p[3], value=p[5]))
        p[0] = p[1]


def p_call(p):
    '''call : ID DOT ID'''
    if p[3] == 'IN' or p[3] == 'Q' or p[3] == 'PT' or p[3] == 'ET':
        p[0] = ast.Name(id=p[1] + '_' + p[3])
    else:
        p[0] = ast.Name(id=p[1])


def p_eq_string(p):
    '''eq_string : EQ_STRING LPAREN ID COMMA ID RPAREN'''
    p[0] = ast.Compare(left=[ast.Name(id=p[3])], ops=[ast.Eq()], comparators=[ast.Name(id=p[5])])


def p_exp(p):
    '''exp : EXP LPAREN expression RPAREN'''
    if type(p[3]) is not ast.Constant:
        p[0] = ast.BinOp(left=p[3], op=ast.Add(), right=ast.Constant(value=1))
    else:
        p[0] = ast.Constant(value=math.exp(float(p[3].value)))


def p_log(p):
    '''log : LOG LPAREN expression RPAREN
           | LN LPAREN expression RPAREN
    '''
    if p[1] == 'LOG':
        if type(p[3]) is not ast.Constant:
            p[0] = ast.BinOp(left=ast.BinOp(left=p[3], op=ast.Sub(), right=ast.Constant(value=1)), op=ast.Div(),
                             right=ast.Constant(value=2.3))
        elif type(p[3]) is ast.Constant:
            p[0] = ast.Constant(value=int(math.log10(float(p[3].value))))  # no support for float
    elif p[1] == 'LN':
        if type(p[3]) is not ast.Constant:
            p[0] = ast.BinOp(left=p[3], op=ast.Sub(), right=ast.Constant(value=1))
        elif type(p[3]) is ast.Constant:
            p[0] = ast.Constant(value=int(math.log(float(p[3].value))))  # no support for float


def p_int_number(p):
    '''int_number : INT_NUMBER
                  | TRUE
                  | FALSE
                  | ID
    '''
    if len(p) == 2:
        if p[1] == 'TRUE':
            p[0] = ast.Constant(value=True)
        elif p[1] == 'FALSE':
            p[0] = ast.Constant(value=False)
        elif '#' in p[1]:
            number = re.findall(r'\d+', p[1])
            p[0] = ast.Constant(value=int(number[0]))
        else:
            p[0] = ast.Constant(value=int(p[1]))
    elif len(p) == 4:
        p[0] = ast.Constant(value=int(p[3]))


def p_float_number(p):
    '''float_number : FLOAT_NUMBER
                    | ID
    '''
    if len(p) == 2:
        if '#' in p[1]:
            number = re.findall(r'\d+', p[1])
            p[0] = ast.Constant(value=int(number[0]))
        else:
            p[0] = ast.Constant(value=int(float(p[1])))  # no support for float
    elif len(p) == 4:
        p[0] = ast.Constant(value=int(float(p[3])))  # no support for float


def p_string_number(p):
    '''string_number : STRING_NUMBER'''
    p[0] = ast.Constant(value=p[1])


# Error rule for syntax errors
def p_error(p):
    print("Syntax error in input!")