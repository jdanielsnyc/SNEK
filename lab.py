"""6.009 Lab 10: Snek Interpreter Part 2"""

# REPLACE THIS FILE WITH YOUR lab.py FROM THE PREVIOUS LAB, WHICH SHOULD BE THE
# STARTING POINT FOR THIS LAB.  YOU SHOULD ALSO ADD: import sys

import doctest
import sys
# NO ADDITIONAL IMPORTS!


###########################
# Snek-related Exceptions #
###########################

class SnekError(Exception):
    """
    A type of exception to be raised if there is an error with a Snek
    program.  Should never be raised directly; rather, subclasses should be
    raised.
    """
    pass


class SnekSyntaxError(SnekError):
    """
    Exception to be raised when trying to evaluate a malformed expression.
    """
    pass


class SnekNameError(SnekError):
    """
    Exception to be raised when looking up a name that has not been defined.
    """
    pass


class SnekEvaluationError(SnekError):
    """
    Exception to be raised if there is an error during evaluation other than a
    SnekNameError.
    """
    pass


###############################
# Snek Classes and Structures #
###############################

class Environment:

    def __init__(self, parent=None):
        self.parent = parent
        self.vars = {}

    def define(self, var, val):
        if not isinstance(var, str):
            raise SnekNameError("Invalid variable name")
        self.vars[var] = val
        return val

    def get(self, var):
        if var in self.vars:
            return self.vars[var]
        elif self.parent is not None:
            return self.parent.get(var)
        else:
            raise SnekNameError('Unresolved reference \'' + var + '\'')

    def set(self, var, val):
        if var in self.vars:
            self.vars[var] = val
            return self.vars[var]
        elif self.parent is not None:
            return self.parent.set(var, val)
        else:
            raise SnekNameError('Unresolved reference \'' + var + '\'')


class Function:
    def __init__(self, parent, params, expr):
        self.parent = parent
        self.params = params
        self.expr = expr

    def __call__(self, args=[]):
        if len(args) != len(self.params):
            raise SnekEvaluationError('Expected ' + str(len(self.params)) + ' arguments, got ' + str(len(args)))

        fcn_env = Environment(self.parent)  # Function environment
        for i, arg in enumerate(args):
            # Assign values in function environment
            param = self.params[i]
            fcn_env.vars[param] = arg

        return evaluate(self.expr, fcn_env)


class Pair:
    def __init__(self, car, cdr):
        self.car = car
        self.cdr = cdr

############################
# Tokenization and Parsing #
############################


def tokenize(source):
    """
    Splits an input string into meaningful tokens (left parens, right parens,
    other whitespace-separated values).  Returns a list of strings.

    Arguments:
        source (str): a string containing the source code of a Snek
                      expression
    """
    tokens = []
    current = ''
    comment = False

    def finish_token(new_current):
        nonlocal current
        if not current == '':
            tokens.append(current)
            current = new_current

    for c in source:
        if c == '\t' or comment:
            # Ignore tabs
            if c == '\n':
                # Comments end on line break
                comment = False
            continue
        elif c == ';':
            # Start comment on semi-colon
            comment = True
        else:
            if c == '(' or c == ')':
                finish_token('')  # If we had an expression before this, add it as a token
                tokens.append(c)
            elif c == ' ' or c == '\n':
                # Spaces mark completed tokens in non-comments
                finish_token('')
            else:
                current += c
    finish_token('')  # Add final token if necessary
    return tokens


keyword_tokens = {'define', 'lambda', 'if'}
def parse(tokens):
    """
    Parses a list of tokens, constructing a representation where:
        * symbols are represented as Python strings
        * numbers are represented as Python ints or floats
        * S-expressions are represented as Python lists

    Arguments:
        tokens (list): a list of strings representing tokens
    """

    def destringify(s):
        s_compare = s[1:] if s[0] == '-' else s
        if s_compare.isnumeric():
            # String is an integer
            return int(s)
        elif s_compare.replace('.', '', 1).isnumeric():
            # String is a float
            return float(s)
        else:
            # String is anything else
            return s

    if len(tokens) > 1:
        # ---S-EXPRESSION TOKENIZATION---
        if tokens[0] != '(' or match_parentheses(tokens, 0) != len(tokens) - 1:
            # If we did not parenthesize this S-expression correctly, return an error
            raise SnekSyntaxError('One or more unmatched parentheses')
        else:
            tokens = tokens[1:-1]  # Remove outer parentheses
            parsed = []
            i = 0
            while i < len(tokens):
                token = tokens[i]
                if token == '(':
                    right_bound = match_parentheses(tokens, i) + 1  # Right bound of this S-expression
                    parsed.append(parse(tokens[i:right_bound]))
                    i = right_bound
                else:
                    parsed.append(destringify(tokens[i]))
                    i += 1
            check_valid_definition(parsed)
        return parsed
    else:
        # ---NON S-EXPRESSION TOKENIZATION---
        token = tokens[0]
        if token == '(' or token == ')' or token in keyword_tokens:
            # Parenthesis mismatch
            raise SnekSyntaxError('One or more unmatched parentheses')
        return destringify(token)


def check_valid_definition(parsed):
    if len(parsed) == 0:
        return
    # ---DEFINE KEYWORD FORMAT CHECKS---
    if parsed[0] == 'define':
        if len(parsed) != 3:
            raise SnekSyntaxError('Expected 3 arguments, got ' + str(len(parsed)))
        elif isinstance(parsed[1], list):
            # For short function definitions using 'define'
            if len(parsed[1]) >= 1:
                for var in parsed[1]:
                    if not isinstance(var, str):
                        raise SnekSyntaxError('Function names and parameters must be strings')
                return  # All parameters (if they exist) are strings
            raise SnekSyntaxError('Unspecified function name')
        elif not isinstance(parsed[1], str):
            raise SnekSyntaxError('Object identifiers must be a string or S-expression')
    # ---LAMBDA KEYWORD FORMAT CHECKS---
    elif parsed[0] == 'lambda':
        if len(parsed) != 3:
            raise SnekSyntaxError('Expected 3 arguments, got ' + str(len(parsed)))
        elif isinstance(parsed[1], list):
            for var in parsed[1]:
                if not isinstance(var, str):
                    raise SnekSyntaxError('Function parameters must be strings')
            return  # All parameters (if they exist) are strings
        else:
            raise SnekSyntaxError('Expected parameter specification')


def match_parentheses(exp, start):
    # Given a tokenized expression and a starting index corresponding to an
    # opening parenthesis, find the corresponding closing parenthesis
    count = 0  # Add one for '(', subtract one for ')'
    for i, c in enumerate(exp[start:]):
        if c == '(':
            count += 1
        elif c == ')':
            count -= 1
        if count == 0:
            return i + start
    raise SnekSyntaxError('One or more unmatched parentheses')


######################
# Built-in Functions #
######################

def snek_mult(args=()):
    product = 1
    for i in args:
        product *= i
    return product


def snek_div(args=()):
    if not args:
        raise SnekEvaluationError('No arguments specified')
    quotient = 1/args[0] if len(args) == 1 else args[0]
    for i in args[1:]:
        quotient /= i
    return quotient


def snek_compare(comparator):
    # comparator is a lambda function that takes two arguments and returns
    # either true or false based on their relative values. This function defines
    # the comparative condition that the returned function checks for.
    def comp(args=()):
        if not args:
            raise SnekEvaluationError('No arguments given')
        elif len(args) == 1:
            return '#t'
        for i in range(len(args) - 1):
            # Check if each pair of adjacent elements fulfills comparison
            # condition, and we find one that doesn't, return False
            if not comparator(args[i], args[i + 1]):
                return '#f'
        return '#t'  # Everything checks out, return True
    return comp


def snek_not(args=()):
    if len(args) != 1:
        raise SnekEvaluationError('Expected 1 argument, got ' + str(len(args)))
    elif args[0] not in {'#t', '#f'}:
        raise SnekEvaluationError('Expected boolean, got, ' + str(type(args[0])))
    return '#t' if args[0] == '#f' else '#f'


def snek_cons(args=()):
    if len(args) != 2:
        raise SnekEvaluationError('Expected 2 arguments, got ' + str(len(args)))
    car = args[0]
    cdr = args[1]
    return Pair(car, cdr)


def snek_car(args=()):
    if len(args) != 1:
        raise SnekEvaluationError('Expected 1 argument, got ' + str(len(args)))
    elif not isinstance(args[0], Pair):
        raise SnekEvaluationError('Expected <class \'Pair\'>, got, ' + str(type(args[0])))
    return args[0].car


def snek_cdr(args=()):
    if len(args) != 1:
        raise SnekEvaluationError('Expected 1 argument, got ' + str(len(args)))
    elif not isinstance(args[0], Pair):
        raise SnekEvaluationError('Expected cons cell, got, ' + str(type(args[0])))
    return args[0].cdr


def snek_list(args=()):
    if len(args) == 0:
        return 'nil'
    car = args[0]
    cdr = snek_list(args[1:])
    return Pair(car, cdr)


def snek_length(args=()):
    if len(args) != 1:
        raise SnekEvaluationError('Expected 1 argument, got ' + str(len(args)))
    elif args[0] == 'nil':
        return 0
    elif not isinstance(args[0], Pair):
        raise SnekEvaluationError('Expected list, got, ' + str(type(args[0])))
    cell = args[0]
    length = 0
    while True:
        if cell.cdr == 'nil':
            # List ends
            return length + 1
        elif isinstance(cell.cdr, Pair):
            # List continues
            length += 1
            cell = cell.cdr
        else:
            # Cell is not a list
            raise SnekEvaluationError('Invalid list format')


def snek_get_elt(args=()):
    if len(args) != 2:
        raise SnekEvaluationError('Expected 2 arguments, got ' + str(len(args)))
    elif not isinstance(args[0], Pair):
        raise SnekEvaluationError('Expected list, got, ' + str(type(args[0])))
    elif not isinstance(args[1], int) or not args[1] >= 0:
        raise SnekEvaluationError('List indices must be non-negative integers')
    cell = args[0]
    query = args[1]
    elt = 0
    while True:
        if elt == query:
            # We reach the index we're looking for
            return cell.car
        elif cell.cdr == 'nil':
            # We reach the end of the list without finding the value
            raise SnekEvaluationError('Index out of bounds')
        elif isinstance(cell.cdr, Pair):
            # We haven't found the index we're looking for, but the list continues
            elt += 1
            cell = cell.cdr
        else:
            # Cell is not a list
            raise SnekEvaluationError('Invalid list format')


def validate(c):
    # Raises an error if the input cell is not a Pair, or is a Pair, but isn't a valid list cell
    if not isinstance(c, Pair) or not (c.cdr == 'nil' or isinstance(c.cdr, Pair)):
        raise SnekEvaluationError('Expected list, got, ' + str(type(c)))


def snek_concat(args=()):
    if len(args) == 0:
        # Return empty list if given no arguments
        return 'nil'

    end = Pair('nil', 'nil')
    source = end
    for l in args:
        if l == 'nil':
            continue
        cell = l
        while True:
            validate(cell)  # Make sure that the next cell to be appended is a list cell
            end.cdr = Pair(cell.car, cell.cdr)  # Append copy of cell to end of new list
            end = end.cdr  # Move pointer up one cell to the end of the list
            if cell.cdr != 'nil':
                cell = cell.cdr
            else:
                break
    return source.cdr  # We started off with an empty cell that pointed to the new concatenated list


def snek_map(args=()):
    if args[1] == 'nil':
        return 'nil'
    if len(args) != 2:
        raise SnekEvaluationError('Expected 2 arguments, got ' + str(len(args)))
    elif not isinstance(args[1], Pair):
        raise SnekEvaluationError('Expected list, got, ' + str(type(args[1])))

    func = args[0]
    cell = args[1]
    end = Pair('nil', 'nil')
    source = end
    while True:
        validate(cell)
        elt = func([cell.car])
        end.cdr = Pair(elt, 'nil')  # Append cell with mapped element to end of new list
        end = end.cdr  # Move pointer up one cell to the end of the list
        if cell.cdr != 'nil':
            cell = cell.cdr
        else:
            break
    return source.cdr  # We started off with an empty cell that pointed to the new mapped list


def snek_filter(args=()):
    if args[1] == 'nil':
        return 'nil'
    if len(args) != 2:
        raise SnekEvaluationError('Expected 2 arguments, got ' + str(len(args)))
    elif not isinstance(args[1], Pair):
        raise SnekEvaluationError('Expected list, got, ' + str(type(args[1])))

    func = args[0]
    cell = args[1]
    end = Pair('nil', 'nil')
    source = end
    while True:
        validate(cell)
        if func([cell.car]) == '#t':
            end.cdr = Pair(cell.car, 'nil')  # Append cell cell to end of new list
            end = end.cdr  # Move pointer up one cell to the end of the list

        if cell.cdr != 'nil':
            cell = cell.cdr
        else:
            break

    return source.cdr  # We started off with an empty cell that pointed to the new mapped list


def snek_reduce(args=()):
    if args[1] == 'nil':
        return args[2]
    if len(args) != 3:
        raise SnekEvaluationError('Expected 3 arguments, got ' + str(len(args)))
    elif not isinstance(args[1], Pair):
        raise SnekEvaluationError('Expected list, got, ' + str(type(args[1])))

    func = args[0]
    cell = args[1]
    val = args[2]
    while True:
        validate(cell)
        val = func([val, cell.car])
        if cell.cdr != 'nil':
            cell = cell.cdr
        else:
            break
    return val


def snek_print_list(l):
    output = []
    while True:
        if l.cdr == 'nil':
            break
        else:
            output.append(l.car)
            l = l.cdr
    print(output)


def snek_begin(args=()):
    if len(args) == 0:
        raise SnekEvaluationError('No arguments given')
    return args[-1]


snek_builtins = {
    '+': sum,
    '-': lambda args: -args[0] if len(args) == 1 else (args[0] - sum(args[1:])),
    '*': snek_mult,
    '/': snek_div,
    '=?': snek_compare(lambda a, b: a == b),
    '>': snek_compare(lambda a, b: a > b),
    '>=': snek_compare(lambda a, b: a >= b),
    '<': snek_compare(lambda a, b: a < b),
    '<=': snek_compare(lambda a, b: a <= b),

    'not': snek_not,

    'cons': snek_cons,
    'car': snek_car,
    'cdr': snek_cdr,
    'list': snek_list,
    'length': snek_length,
    'elt-at-index': snek_get_elt,
    'concat': snek_concat,
    'map': snek_map,
    'filter': snek_filter,
    'reduce': snek_reduce,

    'begin': snek_begin
}
root_env = Environment()
root_env.vars = snek_builtins


##############
# Evaluation #
##############


def evaluate(tree, env=None):
    """
    Evaluate the given syntax tree according to the rules of the Snek
    language.

    Arguments:
        tree (type varies): a fully parsed expression, as the output from the
                            parse function
        env: the environment in which this tree will be evaluated in
    """
    if env is None:
        env = Environment(root_env)
    typ = type(tree)
    if typ == list:
        if len(tree) == 0:
            raise SnekEvaluationError('Empty argument')
        head = tree[0]

        def call(fcn, args):
            # For a parsed expression (function, args), returns the output of function(args)
            args = [evaluate(arg, env) for arg in args]
            return fcn(args)
        if isinstance(head, str):
            special_forms = {'define': snek_define,
                             'lambda': snek_lambda,
                             'and': snek_and,
                             'or': snek_or,
                             'if': snek_if,
                             'let': snek_let,
                             'set!': snek_set
                             }
            if head in special_forms:
                special_func = special_forms[head]
                return special_func(tree, env)
            else:
                # Built-in and defined function handler
                return call(env.get(head), tree[1:])
        elif isinstance(evaluate(head, env), Function):
            # In-line lambda function handler
            result = call(evaluate(head, env), tree[1:])
            return result
        else:
            raise SnekEvaluationError('Invalid operator')
    else:
        sym = tree
        literals = {'#t', '#f', 'nil'}
        if typ == int or typ == float:
            # Number
            return sym
        elif typ == Function:
            # User-defined function with no parameters
            return sym()
        elif typ == str and sym not in literals:
            # User-defined object
            return env.get(sym)
        else:
            # Snek literal or non-evaluable
            return sym


def snek_define(tree, env):
    # Variable and short-def function handler
    if isinstance(tree[1], list):
        name = tree[1][0]
        params = tree[1][1:]
        expr = tree[2]
        return env.define(name, Function(env, params, expr))
    else:
        name = tree[1]
        expr = evaluate(tree[2], env)
        return env.define(name, expr)


def snek_lambda(tree, env):
    # Lambda function creation handler
    params = tree[1]
    expr = tree[2]
    return Function(env, params, expr)


def snek_and(tree, env):
    for cond in tree[1:]:
        val = evaluate(cond, env)
        if val not in {'#t', '#f'}:
            raise SnekEvaluationError('All arguments must be booleans')
        elif val == '#f':
            return '#f'
    return '#t'


def snek_or(tree, env):
    for cond in tree[1:]:
        val = evaluate(cond, env)
        if val not in {'#t', '#f'}:
            raise SnekEvaluationError('All arguments must be booleans')
        elif val == '#t':
            return '#t'
    return '#f'


def snek_if(tree, env):
    cond = evaluate(tree[1], env)
    if cond not in {'#t', '#f'}:
        raise SnekEvaluationError('Expected boolean, got, ' + str(type(cond)))
    true_exp = tree[2]
    false_exp = tree[3]
    if cond == '#t':
        return evaluate(true_exp, env)
    else:
        return evaluate(false_exp, env)


def snek_let(tree, env):
    sub_env = Environment(env)
    for pair in tree[1]:
        var = pair[0]
        val = evaluate(pair[1], sub_env)
        sub_env.define(var, val)
    body = tree[2]
    return evaluate(body, sub_env)


def snek_set(tree, env):
    name = tree[1]
    expr = evaluate(tree[2], env)
    return env.set(name, expr)


def result_and_env(tree, env=None):
    if env is None:
        env = Environment(root_env)
    return evaluate(tree, env), env


def evaluate_file(file, env=None):
    script = ''
    for i in open(file):
        script += i
    return evaluate(parse(tokenize(script)), env)


def REPL(starting_script=None):

    print()
    print('---Haha, Snek go brrr---')

    REPL_env = Environment(root_env)  # Create environment in which this REPL will run

    def recursive_REPL():
        inp = input('in> ')
        if not inp.upper() == 'QUIT':
            if inp == 'env':
                print(REPL_env.vars)
            else:
                try:
                    print('out> : ' + str(evaluate(parse(tokenize(inp)), REPL_env)))
                except SnekSyntaxError as e:
                    print('SnekSyntaxError: ' + str(e))
                except SnekEvaluationError as e:
                    print('SnekEvaluationError: ' + str(e))
                except SnekNameError as e:
                    print('SnekNameError: ' + str(e))
            print()
            recursive_REPL()
        else:
            print("So this is goodbye, I suppose...")

    recursive_REPL()


def unparse(parsed):
    s = str(parsed)
    original = ''
    for i in s:
        if i == '\'' or i == ',':
            continue
        elif i == '[':
            original += '('
        elif i == ']':
            original += ')'
        else:
            original += i
    return original


if __name__ == '__main__':
    # code in this block will only be executed if lab.py is the main file being
    # run (not when this module is imported)

    # uncommenting the following line will run doctests from above
    # doctest.testmod()

    for file in sys.argv[1:]:
        evaluate_file(file, root_env)
    REPL()

