import argparse
import os
import math
import pymongo

    # ------------------------------
    # Function : emptyLines
    # ------------------------------

def emptyLines(file_path, number_of_characters):

    list_of_lines = []
    dictionary_of_lines = {}

    with open(file_path, 'r') as file:
        for lines in file:
            list_of_lines.append(lines)
            # print(list_of_lines)

    reversed_list_of_lines = list(reversed(list_of_lines))

    # print(list_of_lines)

    # print(reversed_list_of_lines)

    dictionary_of_lines = dict(enumerate(list_of_lines))
    reversed_dictionary_of_lines = dict(enumerate(reversed_list_of_lines))

    starting_index = 0
    ending_index = 0

    for key, each_line in dictionary_of_lines.items():
        line = each_line.strip()
        # print(line)
        if line == '':
            # print("emptyline")
            starting_index = key+1

        elif not line == '':
            break


    for key, each_line in reversed_dictionary_of_lines.items():
        line = each_line.strip()
        # print(line)
        if line == '':
            # print("emptyline")
            ending_index = key+1

        elif not line == '':
            break

    # print(starting_index)
    # print(list_of_lines[starting_index])

    # print(ending_index)
    # print(reversed_list_of_lines[ending_index])

    last_index = len(list_of_lines)-ending_index
    # print(len(list_of_lines))
    # print(last_index)

    # print(list_of_lines[starting_index:last_index])

    # for line in list_of_lines[starting_index:last_index]:
    #     print(line)

    content_list_of_lines = list_of_lines[starting_index:last_index]


    # print(content_list_of_lines)

    empty_line_counter = 0

    for each_line in content_list_of_lines:
        stripped_line = each_line.strip()
        if each_line.isspace() or each_line == '':
            empty_line_counter += 1

    lines_divided_by_characters = empty_line_counter / number_of_characters

    log_of_emptyLines = math.log(lines_divided_by_characters)

    # print(empty_line_counter)
    # print(number_of_characters)
    # print(lines_divided_by_characters)
    # print(log_of_emptyLines)

    return(log_of_emptyLines)


    # ------------------------------
    # Function : whiteSpaceRatio
    # ------------------------------

def whiteSpaceRatio(file_path, number_of_characters):

    number_of_empty_spaces = 0

    with open(file_path, 'r') as file:
        for line in file:
            # if line.isspace() == True:
            #     number_of_empty_spaces += 1
            # print(line)
            for character in line:
                if(character.isspace()):
                    number_of_empty_spaces += 1
                # print(character)
                # print(number_of_empty_spaces)

    # print(number_of_empty_spaces)
    # print(number_of_characters)

    whiteSpaceRatio = number_of_empty_spaces / number_of_characters

    return(whiteSpaceRatio)

    # ------------------------------
    # Function : numTabs
    # ------------------------------

def numTabs(file_path, number_of_characters):

    number_of_tab_spaces = 0

    with open(file_path, 'r') as file:
        for line in file:
            # if line.isspace() == True:
            #     number_of_empty_spaces += 1
            # print(line)
            for character in line:
                if(character.isspace()):
                    # if character == ' ':
                    #     number_of_white_spaces += 1
                    if character == '\t':
                        number_of_tab_spaces += 1
                # print(character)
                # print(number_of_empty_spaces)

    # print(number_of_tab_spaces)
    # print(number_of_characters)

    divided_value = number_of_tab_spaces / number_of_characters

    return(divided_value)

    # log_of_numTabs = math.log(divided_value)

    # print(log_of_numTabs)

    # there is an issue.. since most python IDEs convert tabs to spaces.. if there are no tabs that becomes 0 tabs.. and log of 0 is undefined.. so if there were no tabs used then what is undefined value ?
    # maybe we can skip taking log in this case and just go with divided value ?

    # ------------------------------
    # Function : numSpaces
    # ------------------------------

def numSpaces(file_path, number_of_characters):

    number_of_white_spaces = 0

    with open(file_path, 'r') as file:
        for line in file:
            # if line.isspace() == True:
            #     number_of_empty_spaces += 1
            # print(line)
            for character in line:
                if(character.isspace()):
                    if character == ' ':
                        number_of_white_spaces += 1
                    # if character == '\t':
                    #     number_of_tab_spaces += 1
                # print(character)
                # print(number_of_empty_spaces)

    # print(number_of_tab_spaces)
    # print(number_of_white_spaces)
    # print(number_of_characters)


    divided_value = number_of_white_spaces / number_of_characters

    log_of_numSpaces = math.log(divided_value)

    return(log_of_numSpaces)


    # ------------------------------
    # Function : tabsLeadLines
    # ------------------------------

def tabsLeadLines(file_path):


    number_of_space_start = 0
    number_of_tab_start = 0

    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():
                if line.startswith(' '):
                    number_of_space_start += 1
                elif line.startswith('\t'):
                    number_of_tab_start += 1


    if number_of_tab_start >= number_of_space_start:
        return(1)
    else:
        return(0)

def saveOutputVector(file_name, file_folder, file_author, file_path, output_vector):

    # -------------------------------------------------------
    # Mongo db initialization commands starts here
    # -------------------------------------------------------
        
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")

    mydb = myclient["VersionPy"]

    mycol = mydb["VectorLayout"]

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
    # LAYOUT FEATURE - ln(EmptyLines/length)
    # 
    # Log of the number of empty lines divided by file length in characters, excluding leading and trailing lines between lines of text
    # ------------------------------

    log_of_empty_lines = emptyLines(file_path, number_of_characters)
    # print(log_of_empty_lines)

    # ------------------------------
    # LAYOUT FEATURE - whiteSpaceRatio
    # 
    # The ratio between the number of whitespace characters (spaces, tabs, and newlines) and non-whitespace characters
    # ------------------------------

    white_space_ratio = whiteSpaceRatio(file_path, number_of_characters)
    # print(white_space_ratio)

    # ------------------------------
    # LAYOUT FEATURE - (numTabs/length)
    # 
    # number of tab characters divided by file length in characters
    # ------------------------------

    numTabs_ratio = numTabs(file_path, number_of_characters)
    # print(numTabs_ratio)

    # ------------------------------
    # LAYOUT FEATURE - ln(numSpaces/length)
    # 
    # Log of the number of space characters divided by file length in characters
    # ------------------------------

    log_of_numSpaces = numSpaces(file_path, number_of_characters)
    # print(log_of_numSpaces)

    # ------------------------------
    # LAYOUT FEATURE - tabsLeadLines
    # 
    # A boolean representing whether the majority of indented lines begin with spaces or tabs
    # ------------------------------

    num_tabs_leads = tabsLeadLines(file_path)
    # print(num_tabs_leads)

    # ------------------------------
    #
    # Create Output Vector
    #
    # ------------------------------

    output_vector = [log_of_empty_lines, white_space_ratio, numTabs_ratio, log_of_numSpaces, num_tabs_leads]

    # ------------------------------
    #
    #  Save Output Vector
    #
    # ------------------------------

    saveOutputVector(file_name, file_folder, file_author, file_path, output_vector)



if __name__ == '__main__':
    main()