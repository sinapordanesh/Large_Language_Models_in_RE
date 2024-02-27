import argparse
import os
import tokenize
import math
import statistics
import keyword
# import pymongo

    # ------------------------------
    # Function : numTokens
    # ------------------------------

def numTokens(file_path, number_of_characters):
    
    number_of_wordTokens = 0

    with tokenize.open(file_path) as file:
        tokens = tokenize.generate_tokens(file.readline)
        for token in tokens:
            number_of_wordTokens += 1
        # print(number_of_wordTokens)

    # number of word tokens divided by file length in characters
    numTokens = number_of_wordTokens/number_of_characters

    log_of_numTokens = math.log(numTokens)

    return (log_of_numTokens)

    # ------------------------------
    # Function : lineLength
    # ------------------------------  

def lineLength(file_path):

    file = open(file_path, "r")

    lines = file.readlines()

    avgLineLength = sum([len(line.strip('\n')) for line in lines])/len(lines)

    file.close()

    return avgLineLength

    # ------------------------------
    # Function : stdevLineLength
    # ------------------------------  

def stdevLineLength(file_path):

    stdDevList = []
    
    file = open(file_path, "r")

    lines = file.readlines()

    for line in lines:
        stdDevList.append(len(line.strip('\n')))
        
    stdDevLineLength = statistics.stdev(stdDevList)

    file.close()

    return(stdDevLineLength)

    # ------------------------------
    # Function : numComments
    # ------------------------------  

def numComments(file_path, number_of_characters):

    number_of_comments = 0

    with tokenize.open(file_path) as f:
        tokens = tokenize.generate_tokens(f.readline)
        for toknum, tokval, tok3 , tok4 , tok5 in tokens:
            if toknum == tokenize.COMMENT:
                number_of_comments +=1

    log_of_numComments = math.log(number_of_comments / number_of_characters)

    return(log_of_numComments)

    # ------------------------------
    # Function : numFunctions
    # ------------------------------  

def numFunctions(file_path, number_of_characters):
   
    number_of_functions = 0

    with tokenize.open(file_path) as f:
        tokens = tokenize.generate_tokens(f.readline)
        for toknum, tokval, tok3, tok4, tok5 in tokens:
            if tokval == 'def':
                number_of_functions += 1

    log_of_numFunctions = math.log(number_of_functions / number_of_characters)

    return(log_of_numFunctions)

    # ------------------------------
    # Function : numKeywords
    # ------------------------------  

def numKeywords(file_path, number_of_characters):

    keywordDict = {
    "elif":0.0,
    "if":0.0,
    "else":0.0,
    "for":0.0,
    "while":0.0
    }

    with tokenize.open(file_path) as f:
        tokens = tokenize.generate_tokens(f.readline)
        for toknum, tokval, tok3 , tok4 , tok5 in tokens:
            # print(tokval)
            
            if tokval == "if":
                keywordDict["if"] = keywordDict["if"] + 1

            if tokval == "elif":
                keywordDict["elif"] = keywordDict["elif"] + 1

            if tokval == "else":
                keywordDict["else"] = keywordDict["else"] + 1

            if tokval == "for":
                keywordDict["for"] = keywordDict["for"] + 1

            if tokval == "while":
                keywordDict["while"] = keywordDict["while"] + 1


    for key, value in keywordDict.items():
        keywordDict.update({key:(value/number_of_characters)})

                
    for key, value in keywordDict.items():
        if value != 0:
            keywordDict.update({key:math.log(value)})

    return(keywordDict)

    # ------------------------------
    # Function : tfwordUnigram
    # ------------------------------ 

def tfwordUnigram(file_path):

    term_frequency_dict = {}
    listoflength = []

    # with open (file_path, 'r') as fn:

    file = open(file_path, "r")

    for line in file:
        listoflength.append(len(line.split()))
        for word in line.split():
            if word in term_frequency_dict:
                term_frequency_dict[word] += 1
            else:
                term_frequency_dict[word] = 1
    

    # print(term_frequency_dict)

    # To sum the number of word unigrams in every line
    # listoflength = [len(line.split()) for line in file]

    # print(listoflength)


    sumOfAllWords = sum(listoflength)
    # print(sumOfAllWords)


    for word, freq in term_frequency_dict.items():
        # print(word, freq)
        term_frequncy = freq / sumOfAllWords
        term_frequency_dict[word] = term_frequncy

    file.close()

    return(term_frequency_dict)

    # ------------------------------
    # Function : avgParams
    # ------------------------------ 

def getParams(func_def):

    numParams = 0

    param_list = func_def[func_def.find('(')+1:func_def.find(')')]
    if param_list != '':
        param_count = param_list.split(',')
        numParams += len(param_count)
    else:
        numParams = numParams

    return(numParams)


def avgParams(file_path):

    listOfDef = []

    number_of_params = 0

    with tokenize.open(file_path) as f:
        tokens = tokenize.generate_tokens(f.readline)
        for tokname, tokval, startPos, endPos, lineString in tokens:
            if tokname == tokenize.NAME and tokval == 'def':
                # print(lineString)
                # print(type(lineString))
                listOfDef.append(str(lineString.strip('\n')))

    # print(listOfDef)

    for func_def in listOfDef:
        paramCount = getParams(func_def)
        number_of_params += paramCount
        # print(paramCount)

    # print(number_of_params)

    number_of_functions = len(listOfDef)

    avgParams = number_of_params / number_of_functions

    return(avgParams)

    # ------------------------------
    # Function : stdevNumParams
    # ------------------------------ 

def stdevNumParams(file_path):

    def getParamsList(func_def):

        param_list = func_def[func_def.find('(')+1:func_def.find(')')]
        if param_list != '':
            param_count = param_list.split(',')
            len_param_count = len(param_count)
        else:
            len_param_count = float(0.0)
        
        return(len_param_count)


    listOfDef = []

    list_of_params = []

    with tokenize.open(file_path) as f:
        tokens = tokenize.generate_tokens(f.readline)
        for tokname, tokval, startPos, endPos, lineString in tokens:
            if tokname == tokenize.NAME and tokval == 'def':
                # print(lineString)
                # print(type(lineString))
                listOfDef.append(str(lineString.strip('\n')))

    # print(listOfDef)

    for func_def in listOfDef:
        parameter_list = getParamsList(func_def)
        list_of_params.append(parameter_list)


    # print(list_of_params)
        
    if len(list_of_params) == 1:
        stdDevNumParams = float(0)
    else:
        stdDevNumParams = statistics.stdev(list_of_params)

    # print(stdDevNumParams)

    return(stdDevNumParams)
    # return()

    # ------------------------------
    # Function : pythonkeywords
    # ------------------------------ 

def pythonkeywords(file_path, number_of_characters):

    list_of_keyword = keyword.kwlist

    dict_of_keyword = dict.fromkeys(list_of_keyword, 0.0)

    with tokenize.open(file_path) as f:
        tokens = tokenize.generate_tokens(f.readline)
        for toknum, tokval, tokstart, tokend, tokline in tokens:
            for key, value in dict_of_keyword.items():
                if key == tokval:
                    dict_of_keyword[tokval] += 1.0
    
    # print(dict_of_keyword)

    counter_for_keywords = 0 

    for key, value in dict_of_keyword.items():
        if value > 0.0:
            # print(key)
            counter_for_keywords += 1

    # print(counter_for_keywords)

    division_of_keywords = counter_for_keywords / number_of_characters

    # print(division_of_keywords)

    log_of_pythonkeywords = math.log(division_of_keywords)

    return(log_of_pythonkeywords)

    # ------------------------------
    # Function : numLiterals
    # ------------------------------

def numLiterals(file_path, number_of_characters):

    # print("\n# of chars------->")
    # print(number_of_characters)
    number_of_stringLiteral = 0
    number_of_numberLiteral = 0

    with tokenize.open(file_path) as f:
        tokens = tokenize.generate_tokens(f.readline)
        # for toknum, tokval, tokstart, tokend, tokstring in tokens:
        #     if toknum == tokenize.STRING or toknum == tokenize.NUMBER:
        #         print(toknum, tokval, tokstring.strip('\n'))
        for token in tokens:
            if(token[0] == tokenize.STRING):
                # print(token)
                number_of_stringLiteral += 1
            if(token[0] == tokenize.NUMBER):
                # print(token)
                number_of_numberLiteral +=1
        
    # print(number_of_numberLiteral)
    # print(number_of_stringLiteral)


    number_of_literals = number_of_numberLiteral + number_of_stringLiteral

    avgLiteral = number_of_literals / number_of_characters

    lnAvgLiteral = math.log(avgLiteral)

    return(lnAvgLiteral)


    # ------------------------------
    # Function : saveOutputVector
    # ------------------------------

def saveOutputVector(file_name, file_folder, file_author, file_path, output_vector):

    # -------------------------------------------------------
    # Mongo db initialization commands starts here
    # -------------------------------------------------------
        
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")

    mydb = myclient["VersionPy"]

    mycol = mydb["VectorLexical"]

    insertDocument = {"File_Name" : file_name, "File_Folder" : file_folder, "Coded_By" : file_author, "File_Path" : file_path, "Output_Vector" : output_vector}

    x = mycol.insert_one(insertDocument)

    # print(x)
    pass


def main():
    
    
    # Parse the input commmand which contains the filename to extract the lexical features from
    
    cmd_parser = argparse.ArgumentParser()
    cmd_parser.add_argument('file', help='Add absolute path to the file that needs to be parsed')
    args = cmd_parser.parse_args()
    
    file_path = args.file
    
    splitted_file_path = file_path.split("/")
    
    # Extracting data about the file
        
    file_name = os.path.splitext(splitted_file_path[-1])[0]
    
    file_folder = splitted_file_path[-2]
    
    file_author = splitted_file_path[-3]

    # Extract the number of characters in the file for use of all feature extraction 

    number_of_characters = 0

    with open(file_path) as file:
        for line in file:
            number_of_characters += len(line.strip('\n')) # this is to remove new line at the end and start of the line

    # print(number_of_characters)

    # ------------------------------
    # LEXICAL FEATURE - ln(numTokens / length)
    #
    # log of the number of word tokens divided by file length in characters -- this should be number of word unigrams for me
    # ------------------------------

    log_of_numTokens = numTokens(file_path, number_of_characters)   
    # print(log_of_numTokens) 
    
    # ------------------------------
    # LEXICAL FEATURE - avgLineLength
    # 
    # The average length of each line 
    # ------------------------------  

    avg_line_length = lineLength(file_path)
    # print(avg_line_length)

    # ------------------------------
    # LEXICAL FEATURE - stdDevLineLength
    # 
    # The standard deviation of the character lengths of each line
    # ------------------------------

    stdDev_line_length = stdevLineLength(file_path)
    # print(stdDev_line_length) 

    # ------------------------------
    # LEXICAL FEATURE - ln(numComments / length)
    # 
    # Log of the number of comments divided by file lenght in characters
    # ------------------------------

    # log_of_numComments = numComments(file_path, number_of_characters)
    # print(log_of_numComments)

    # ------------------------------
    # LEXICAL FEATURE - ln(numFunctions / length)
    # 
    # Log of the number of functions divided by file length in characters
    # ------------------------------

    log_of_numFunctions = numFunctions(file_path, number_of_characters)
    # print(log_of_numFunctions)

    # ------------------------------
    # LEXICAL FEATURE - ln(numkeyword / length)
    # 
    # Log of the number of occurrences of keyword divided by file length in characters, where keyword is one of elif, if, else, for or while
    # ------------------------------

    dict_log_of_numKeywords = numKeywords(file_path, number_of_characters)
    # print(dict_log_of_numKeywords)

    # ------------------------------
    # LEXICAL FEATURE - WordUnigramTF
    # 
    # Term frequency of word unigrams in source code
    # ------------------------------

    dict_TF_wordUnigrams = tfwordUnigram(file_path)
    # print(dict_TF_wordUnigrams)

    # ------------------------------
    # LEXICAL FEATURE - avgParams
    # 
    # The average number of parameters among all functions
    # ------------------------------

    average_params = avgParams(file_path)
    # print(average_params)

    # ------------------------------
    # LEXICAL FEATURE - stdDevNumParams
    # 
    # The standard deviation of the number of parameters among all functions
    # ------------------------------

    stdDev_numParams = stdevNumParams(file_path)
    # print(stdDev_numParams)

    # ------------------------------
    # LEXICAL FEATURE - ln(numKeywords / length)
    # 
    # log of the number of unique keywords used divided by file length in characters
    # ------------------------------

    log_of_python_keywords = pythonkeywords(file_path, number_of_characters)
    # print(log_of_python_keywords)

    # ------------------------------
    # LEXICAL FEATURE - ln(numLiterals / length)
    # 
    # Log of the number of string and numeric literals divided by file length in characters
    # ------------------------------

    log_of_numLiterals = numLiterals(file_path, number_of_characters)
    # print(log_of_numLiterals)

    # ------------------------------
    # Create Output Vector
    #
    # missing dict_TF_wordUnigrams because how to incorporate dynamic number of points as a vector ? 
    # ------------------------------

    # Make dictionaries to list
    keywordlist = []
    for value in dict_log_of_numKeywords.values():
        keywordlist.append(value)


    output_vector = [log_of_numTokens, avg_line_length, stdDev_line_length, log_of_numFunctions, average_params,
                         stdDev_numParams, log_of_python_keywords, log_of_numLiterals] + keywordlist
    
    print(output_vector)

    # ------------------------------
    # Save Output Vector
    #
    # missing dict_TF_wordUnigrams because how to incorporate dynamic number of points as a vector ? 
    # ------------------------------

    # saveOutputVector(file_name, file_folder, file_author, file_path, output_vector)

if __name__ == '__main__':
    main()