from typing import Dict
from typing import List

import javalang
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .layout import NewLineBeforeOpenBrace
from .layout import NumEmptyLines
from .layout import NumSpaces
from .layout import NumTabs
from .layout import TabsLeadLines
from .layout import WhiteSpaceRatio
from .lexical import AvgLineLength
from .lexical import AvgParams
from .lexical import NumFunctions
from .lexical import NumKeyword
from .lexical import NumKeywords
from .lexical import NumLiterals
from .lexical import NumTernary
from .lexical import NumTokens
from .lexical import StdDevLineLength
from .lexical import StdDevNumParams
from .lexical import WordUnigramTF
from .syntactic import ASTNodeBigramsTF
from .syntactic import ASTNodeTypesTF
from .syntactic import JavaKeywords
from .syntactic import MaxDepthASTNode
from .utils import build_mapping_to_ids

from tqdm.notebook import tqdm
from joblib import Parallel, delayed
from typing import List, Dict

def calculate_features(code: str) -> Dict:
    """
    Calculates a set of features for the given source file.

    :param code: code to be analyzed
    :return: dictionary with features
    """

    # with open(path, 'r', errors='ignore') as file:
    #     code = file.read()

    features = {}

    try:

        if not isinstance(code, str):
            # Return a marker or an empty dictionary to indicate this sample cannot be processed
            return {'error': 'Invalid code type'}
        
        file_length = len(code)
        tokens = list(javalang.tokenizer.tokenize(code))
        tree = javalang.parse.parse(code)

        # LEXICAL FEATURES
        features.update(WordUnigramTF.calculate(tokens))
        features.update(NumKeyword.calculate(tokens, file_length))
        features.update(NumTokens.calculate(tokens, file_length))
        features.update(NumLiterals.calculate(tokens, file_length))
        features.update(NumKeywords.calculate(tokens, file_length))
        features.update(NumFunctions.calculate(tree, file_length))
        features.update(NumFunctions.calculate(tree, file_length))
        features.update(NumTernary.calculate(tree, file_length))
        features.update(AvgLineLength.calculate(code))
        features.update(StdDevLineLength.calculate(code))
        features.update(AvgParams.calculate(tree))
        features.update(StdDevNumParams.calculate(tree))

        # LAYOUT FEATURES
        features.update(NumTabs.calculate(code))
        features.update(NumSpaces.calculate(code))
        features.update(NumEmptyLines.calculate(code))
        features.update(WhiteSpaceRatio.calculate(code))
        features.update(NewLineBeforeOpenBrace.calculate(code))
        features.update(TabsLeadLines.calculate(code))

        # SYNTACTIC FEATURES
        features.update(MaxDepthASTNode.calculate(tree))
        features.update(ASTNodeBigramsTF.calculate(tree))
        features.update(ASTNodeTypesTF.calculate(tree))
        features.update(JavaKeywords.calculate(tokens))

        return features

    except Exception as e:
        # Catch any kind of exception and return a generic error marker
        # You can optionally log the exception message or type for debugging purposes
        return {'error': 'General error'}


def calculate_features_for_files(codes: List[str], n_jobs: int = 4) -> List[Dict]:
    """
    Calculates sets of features for the given source files.

    :param files: list of files
    :param n_jobs: number of jobs
    :return: list with features for each source file
    """

    # with Parallel(n_jobs=n_jobs) as pool:
    #     features = pool(delayed(calculate_features)(path) for path in files)

    # return features
    
    # Initialize a tqdm notebook progress bar
    with tqdm(total=len(codes), desc='Calculating features') as progress_bar:
        with Parallel(n_jobs=n_jobs) as pool:
            features = pool(
                delayed(calculate_features)(code, progress_bar) for code in codes
            )

    # with Parallel(n_jobs=n_jobs) as pool:
    #     features = pool(delayed(calculate_features)(code) for code in codes)
            
    return features


def build_sample(sample: Dict, feature_to_id: Dict) -> np.array:
    features = np.empty(len(feature_to_id))
    features[:] = np.nan

    for key, value in sample.items():
        index = feature_to_id[key]
        features[index] = value

    return features


def build_dataset(samples: List[Dict], n_jobs: int = 4) -> pd.DataFrame:
    """
    Builds a pandas data frame from the given list of feature sets.

    :param samples: list of features
    :param n_jobs: number of jobs
    :return: data frame with all features
    """

    feature_names = set()

    for sample in samples:
        feature_names |= sample.keys()

    feature_names = sorted(feature_names)
    feature_to_id = build_mapping_to_ids(feature_names)

    with Parallel(n_jobs=n_jobs) as pool:
        features = pool(delayed(build_sample)(sample, feature_to_id) for sample in samples)

    features = pd.DataFrame(features)
    features.columns = feature_names

    return features
