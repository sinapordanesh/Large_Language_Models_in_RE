import argparse
import math
from pygments import lex
from pygments.lexers import get_lexer_for_filename
from pygments.token import Token

def numLiteralsWithPygments(file_path):
    number_of_literals = 0
    number_of_characters = 0

    with open(file_path, 'r') as file:
        content = file.read()
        number_of_characters = len(content)
        print(f"Total characters in file: {number_of_characters}")

    try:
        lexer = get_lexer_for_filename(file_path)
        print(f"Using lexer: {lexer.name}")
    except ValueError:
        print("Error: No lexer found for the file extension. Ensure the file extension is correct.")
        return float('-inf')

    for token_type, token_value in lex(content, lexer):
        if token_type in (Token.Literal.String, Token.Literal.Number):
            number_of_literals += 1
            print(f"Found literal: {token_value} of type {token_type}")

    print(f"Total literals found: {number_of_literals}")

    if number_of_characters > 0 and number_of_literals > 0:
        ratio = number_of_literals / number_of_characters
        return math.log(ratio)
    else:
        print("Either no characters or no literals found in the file.")
        return float('-inf')

def main():
    parser = argparse.ArgumentParser(description="Extract the ln(numLiterals/length) lexical feature using Pygments.")
    parser.add_argument('file_path', type=str, help="Path to the file to analyze.")
    args = parser.parse_args()

    log_ratio = numLiteralsWithPygments(args.file_path)
    print(f"Logarithmic ratio of literals to file length: {log_ratio}")

if __name__ == "__main__":
    main()
