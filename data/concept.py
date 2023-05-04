import json
import csv
import nltk
import pickle
from ast import literal_eval
import numpy as np

def get_concept():
    concept_dict = json.load(open("ConceptNet.json", "r", encoding="utf-8"))

    rank_concept_file = json.load(open('ConceptNet_VAD_dict.json', 'r', encoding='utf-8'))
    # print(concept_dict)
    # print(type(concept_dict['afternoon']))
    # exit()
    return concept_dict,rank_concept_file
def match(path_to_txt,dict):
    mode='obj'
    dict2={}
    data = open(path_to_txt, 'rb')
    data = pickle.load(data)
    for item in data:
        id=item.key
        attr_word = []
        obj_word = []
        for word in item.value:
            word=word.split(' ')
            attr_word.append(word[0])
            obj_word.append(word[1])
            lists=[]
            if attr_word in dict.keys():
                list=dict[attr_word]
                dict2[attr_word]=list[0]
            if obj_word in dict.keys():
                list = dict[obj_word]
                dict2[obj_word] = list[0]
    print(len(dict2))
    exit()
    obj_concept_dict=open(mode+'_concept_dict.pkl', 'wb')
    pickle.dump(dict2,obj_concept_dict)

if __name__ == '__main__':
    mode='valid'
    path_to_obj=mode+'_obj.pkl'

    concept_dict,rank_concept_file=get_concept()
    match(path_to_obj,concept_dict)
    exit()