import tokenize
from io import StringIO
from pygments import lex
from pygments.lexers import PythonLexer
from pygments.token import Token

# Sample Python code
code_sample = """def add(a, b):
    # This is a comment
    return a + b"""

# Using Python's tokenize module
print("Tokenizing with Python's tokenize module:")
token_stream = tokenize.generate_tokens(StringIO(code_sample).readline)
for token_type, token_string, start, end, line in token_stream:
    print(f"{tokenize.tok_name[token_type]}: {token_string}")

# A separator for clarity
print("\n" + "-"*50 + "\n")

# Using Pygments
print("Tokenizing with Pygments:")
lexer = PythonLexer()
for token_type, token_value in lex(code_sample, lexer):
    print(f"{token_type}: {token_value}")
