from collections import Counter
from typing import Dict, List

import numpy as np
from javalang.tokenizer import JavaToken
from javalang.tree import MethodDeclaration
from javalang.tree import Node
from javalang.tree import TernaryExpression

from .feature import Feature
from .utils import get_nodes
from .utils import get_nodes_count
from .utils import identifiers
from .utils import keywords
from .utils import literals
from .utils import non_empty_lines


class WordUnigramTF(Feature):
    @staticmethod
    def calculate(tokens: List[JavaToken]) -> Dict:
        values = map(lambda it: it.value, identifiers(tokens))
        count = Counter(values)

        features = {}
        total_count = sum(count.values())
        for key, value in count.items():
            features[f'WordUnigramTF_{key}'] = value / total_count

        return features


class NumKeyword(Feature):
    @staticmethod
    def calculate(tokens: List[JavaToken], file_length: int) -> Dict:
        values = map(lambda it: it.value, keywords(tokens))
        count = Counter(values)

        features = {}
        for key, value in count.items():
            features[f'ln(num_{key}/length)'] = np.log((value + 1e-10) / (file_length + 1e-10))

        return features


class NumTokens(Feature):
    @staticmethod
    def calculate(tokens: List[JavaToken], file_length: int) -> Dict:
        num_identifiers = len(identifiers(tokens))
        value = np.log((num_identifiers + 1e-10) / (file_length + 1e-10))
        return {'ln(numTokens/length)': value}


class NumComments(Feature):
    @staticmethod
    def calculate(code: str) -> Dict:
        lines = non_empty_lines(code)
        num_comments = sum(line.strip()[:2] == '//' for line in lines)
        value = np.log((num_comments + 1e-10) / (len(code) + 1e-10))
        return {'ln(numComments/length)': value}


class NumLiterals(Feature):
    @staticmethod
    def calculate(tokens: List[JavaToken], file_length: int) -> Dict:
        num_literals = len(literals(tokens))
        value = np.log((num_literals + 1e-10) / (file_length + 1e-10))
        return {'ln(numLiterals/length)': value}


class NumKeywords(Feature):
    @staticmethod
    def calculate(tokens: List[JavaToken], file_length: int) -> Dict:
        num_literals = len(keywords(tokens))
        value = np.log((num_literals + 1e-10) / (file_length + 1e-10))
        return {'ln(numKeywords/length)': value}


class NumFunctions(Feature):
    @staticmethod
    def calculate(tree: Node, file_length: int) -> Dict:
        num_functions = get_nodes_count(tree, MethodDeclaration)
        value = np.log((num_functions + 1e-10) / (file_length + 1e-10))
        return {'ln(numFunctions/length)': value}


class NumTernary(Feature):
    @staticmethod
    def calculate(tree: Node, file_length: int) -> Dict:
        num_ternary = get_nodes_count(tree, TernaryExpression)
        value = np.log((num_ternary + 1e-10) / (file_length + 1e-10))
        return {'ln(numTernary/length)': value}


class AvgLineLength(Feature):
    @staticmethod
    def calculate(code: str) -> Dict:
        lines = code.split('\n')
        value = np.mean([len(line) for line in lines])
        return {'avgLineLength': value}


class StdDevLineLength(Feature):
    @staticmethod
    def calculate(code: str) -> Dict:
        lines = code.split('\n')
        value = np.std([len(line) for line in lines])
        return {'stdDevLineLength': value}


class AvgParams(Feature):
    @staticmethod
    def calculate(tree: Node) -> Dict:
        nodes = get_nodes(tree, MethodDeclaration)
        num_params = [len(node.children[6]) for node in nodes]
        value = np.mean(num_params)
        return {'avgParams': value}


class StdDevNumParams(Feature):
    @staticmethod
    def calculate(tree: Node) -> Dict:
        nodes = get_nodes(tree, MethodDeclaration)
        num_params = [len(node.children[6]) for node in nodes]
        value = np.std(num_params)
        return {'stdDevNumParams': value}

# TODO: nestingDepth
# TODO: branchingFactor
