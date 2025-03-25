import json
import sys
from spacy.lang.en import English
import spacy
import random

random.seed(42)
#valid relations in the format of (subject_type, object_type, predicate from the table )
valid_relations = { #made this into an actual dictionary
    ("anatomical location", "human"): "Located in",
    ("anatomical location", "animal"): "Located in",
("bacteria", "bacteria"): "Interact",
    ("bacteria", "chemical"): "Interact",
    ("bacteria", "drug"): "Interact",
    ("bacteria", "DDF"): "Influence",
    ("bacteria", "gene"): "Change expression",
    ("bacteria", "human" ):"Located in",
    ("bacteria", "animal" ):"Located in",
    ("bacteria", "microbiome" ):"Part of",
    ("chemical", "anatomical location" ):"Located in",
    ("chemical", "human" ):"Located in",
    ("chemical", "animal" ):"Located in",
    ("chemical", "chemical"):"Interact",
    ("chemical", "microbiome" ):"Impact",
    ("chemical", "microbiome" ):"Produced by",
    ("chemical", "bacteria" ):"Impact",
    ("dietary supplement" ,"Bacteria"): "Impact",
    ("drug", "bacteria" ):"Impact",
    ("food", "bacteria" ):"Impact",
    ("chemical", "microbiome" ):"Impact",
    ("dietary supplement", "microbiome"): "Impact",
    ("drug", "microbiome" ):"Impact",
    ("food", "microbiome" ):"Impact",
    ("chemical", "DDF" ):"Influence",
    ("dietary supplement", "DDF" ):"Influence",
    ("drug", "DDF" ):"Influence",
    ("food", "DDF" ):"Influence",
    ("chemical", "gene" ):"Change expression",
    ("dietary supplement", "gene" ):"Change expression",
    ("drug", "gene" ):"Change expression",
    ("food", "gene" ):"Change expression",
    ("chemical", "human" ):"Administered",
    ("dietary supplement", "human" ):"Administered",
    ("drug", "human" ):"Administered",
    ("food", "human" ):"Administered",
    ("chemical", "animal" ):"Administered",
    ("dietary supplement", "animal" ):"Administered",
    ("drug", "animal" ):"Administered",
    ("food", "animal" ):"Administered",
    ("DDF", "anatomical location" ):"Strike",
    ("DDF", "bacteria" ):"Change abundance",
    ("DDF", "microbiome" ):"Change abundance",
    ("DDF", "chemical" ):"Interact",
    ("DDF", "DDF" ):"Affect",
    ("DDF", "DDF" ):"Is a",
    ("DDF", "human" ):"Target",
    ("DDF", "animal" ):"Target",
    ("drug", "chemical" ):"Interact",
    ("drug", "chemical" ):"Interact",
    ("drug", "DDF" ):"Change effect",
    ("human", "biomedical technique" ):"Used by",
    ("animal", "biomedical technique" ):"Used by",
    ("microbiome", "biomedical technique" ):"Used by",
    ("microbiome", "anatomical location" ):"Located in",
    ("microbiome", "human" ):"Located in",
    ("microbiome", "animal" ):"Located in",
    ("microbiome", "gene" ):"Change expression",
    ("microbiome", "DDF" ):"Is linked to",
    ("microbiome", "microbiome" ):"Compared to"
}

# Loading the dataSet
file_path = sys.argv[1]
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

def preprocess_train_data(data):
    processed_data = []
    counter = 0
    counter2 = 0
    nlp = spacy.load("en_core_sci_sm")
    nlp.add_pipe("sentencizer")
    for doc_id, doc_data in data.items():
        metadata = doc_data.get("metadata", {})
        abstract = metadata.get("abstract", "")
        title = metadata.get("title", "")
        relations = doc_data.get("relations", [])
        entities = doc_data.get("entities", [])
        used = []
        title_doc = nlp(title)
        abstract_doc = nlp(abstract)
        """for span in doc.sents:
            print(span.text)
            print(abstract[span.start_char:span.end_char])
            assert span.text == abstract[span.start_char:span.end_char]
        exit()"""#just a test

        # Iteration over relations
        for relation in relations:
            subject_start = relation.get("subject_start_idx")
            subject_end = relation.get("subject_end_idx")
            object_start = relation.get("object_start_idx")
            object_end = relation.get("object_end_idx")
            predicate = relation.get("predicate")
            subject_text = relation.get("subject_text_span")
            object_text = relation.get("object_text_span")
            subject_location = relation.get("subject_location")
            object_location = relation.get("object_location")
            sample = ""
            relative_subject_start= -1
            relative_object_start = -1
            relative_subject_end = -1
            relative_object_end = -1
            if subject_location!=object_location:
                counter2+=1

            if subject_location == "abstract":
                for span in abstract_doc.sents:
                    if span.start_char<= subject_start and span.end_char>=subject_end+1:
                        relative_subject_start = len(sample)+(subject_start-span.start_char)
                        relative_subject_end = len(sample)+(subject_end-span.start_char)
                        sample+=span.text
            elif subject_location == "title":
                for span in title_doc.sents:
                    if span.start_char<=subject_start and span.end_char>=subject_end+1:
                        relative_subject_start = len(sample)+(subject_start-span.start_char)
                        relative_subject_end = len(sample)+(subject_end-span.start_char)
                        sample+=span.text
            if object_location == "abstract":
                for span in abstract_doc.sents:
                    if span.start_char<= object_start and span.end_char>=object_end+1:
                        if sample!=span.text:
                            relative_object_start = len(sample)+(object_start-span.start_char)
                            relative_object_end = len(sample)+(object_end-span.start_char)
                            sample+=span.text
                            if subject_location=="abstract":
                                counter+=1
                        elif sample == span.text:
                            relative_object_start = (object_start-span.start_char)
                            relative_object_end = (object_end-span.start_char)

            elif object_location=="title":
                for span in title_doc.sents:
                    if span.start_char<=object_start and span.end_char>=object_end+1:
                        if sample!=span.text:
                            relative_object_start = len(sample)+(object_start-span.start_char)
                            relative_object_end = len(sample)+(object_end-span.start_char)
                            sample+=span.text
                        elif sample == span.text:
                            relative_object_start = (object_start-span.start_char)
                            relative_object_end = (object_end-span.start_char)
            #print((title+abstract)[relative_subject_start:relative_subject_end])
            # print(subject_text)
            assert (sample)[relative_subject_start:relative_subject_end + 1] == subject_text
            assert (sample)[relative_object_start:relative_object_end + 1] == object_text
        
            processed_data.append({
                "sample": sample,
                "subject": subject_text,
                "subject_label": relation.get("subject_label"),
                "object": object_text,
                "object_label": relation.get("object_label"),
                "relation": predicate,
                "relative_subject_start": relative_subject_start,
                "relative_subject_end": relative_subject_end,
                "relative_object_start": relative_object_start,
                "relative_object_end": relative_object_end,
                "doc_id": doc_id
            })
            used.append((sample,(relative_subject_start, relative_subject_end), (relative_object_start, relative_object_end)))
        for e in entities: #need this for preprocess_test_data
            for e1 in entities:
                if e != e1:
                    subject_start = e.get("start_idx")
                    subject_end = e.get("end_idx")
                    object_start = e1.get("start_idx")
                    object_end = e1.get("end_idx")
                    subject_location = e.get("location")
                    object_location = e1.get("location")
                    sample = ""
                    relative_subject_start = -1
                    relative_subject_end = -1
                    relative_object_start = -1
                    relative_object_end = -1
                    if subject_location == "abstract":
                        for span in abstract_doc.sents:
                            if span.start_char<= subject_start and span.end_char>=subject_end+1:
                                relative_subject_start = len(sample)+(subject_start-span.start_char)
                                relative_subject_end = len(sample)+(subject_end-span.start_char)
                                sample+=span.text
                    elif subject_location == "title":
                        for span in title_doc.sents:
                            if span.start_char<=subject_start and span.end_char>=subject_end+1:
                                relative_subject_start = len(sample)+(subject_start-span.start_char)
                                relative_subject_end = len(sample)+(subject_end-span.start_char)
                                sample+=span.text
                    if object_location == "abstract":
                        for span in abstract_doc.sents:
                            if span.start_char<= object_start and span.end_char>=object_end+1 and sample!=span.text:
                                relative_object_start = len(sample)+(object_start-span.start_char)
                                relative_object_end = len(sample)+(object_end-span.start_char)
                                sample+=span.text
                            elif sample == span.text:
                                relative_object_start = (object_start-span.start_char)
                                relative_object_end = (object_end-span.start_char)

                    elif object_location=="title":
                        for span in title_doc.sents:
                            if span.start_char<=object_start and span.end_char>=object_end+1 and sample!=span.text:
                                relative_object_start = len(sample)+(object_start-span.start_char)
                                relative_object_end = len(sample)+(object_end-span.start_char)
                                sample+=span.text
                            elif sample == span.text:
                                relative_object_start = (object_start-span.start_char)
                                relative_object_end = (object_end-span.start_char)

                    # Check if the relation is NONE and if the subject and object types are valid
                    if (sample, (relative_subject_start, relative_subject_end), (relative_object_start, relative_object_end)) not in used:
                        subject_type = e.get("label", "") #FIXED had wrong attribute
                        object_type = e1.get("label", "")

                        # Only include "NONE" relations if they match a valid subject-object pair
                        if (subject_type, object_type) in valid_relations:
                           #  predicate = valid_relations[(subject_type, object_type)] we don't want to overwrite the no relatons
                            processed_data.append({
                                "sample": sample,
                                "subject": e.get("text_span", ""),
                                "subject_label": subject_type,
                                "object": e1.get("text_span", ""),
                                "object_label": object_type,
                                "relation": "NONE", #there's still no relation FOR TEST WE DON'T KNOW THIS
                                "relative_subject_start": relative_subject_start,
                                "relative_subject_end": relative_subject_end,
                                "relative_object_start": relative_object_start,
                                "relative_object_end": relative_object_end,
                                "doc_id":doc_id
                            })

                        else:
                            used.append((sample,(relative_subject_start, relative_subject_end), (relative_object_start, relative_object_end)))
    print("Number of cross sentence (within abstract) entities in relations "+str(counter))
    print("Number of cross sentence (accross title and abstract) entities in relations "+str(counter2))
    return processed_data

def downsample_none(data, fraction=.5): #might want to make this generalized later, but who cares
    tot_rel = 0
    #first we need to figure out how much in total
    dist_frac = {"Target": .152,
            "Influence":.146,
            "Located in":.137,
            "Is linked to":.137,
            "Affect":.121,
            "Impact":.059,
            "Used by":0.052,
            "Interact":0.039,
            "Change abundance":.035,
            "Part of":.029,
            "Is a": 0.025,
            "Change effect":.024,
            "Administered":.014,
            "Change expression":.012,
            "Strike":0.011,
            "Produced by":0.005,
            "Compared to":0.003
    }

    relations = {key:[] for key in dist_frac}
    new_data = []
    for relation in data:
        if relation["relation"]!="NONE":
            new_data.append(relation)
        else:
            tot_rel+=1
            potential_relation=valid_relations[relation["subject_label"], relation["object_label"]] #need to fix
            relations[potential_relation].append(relation)
    for relation in relations:
        random.shuffle(relations[relation])
    dist_num = {key:int(dist_frac[key]*tot_rel) for key in dist_frac}
    sample_no_rel = [relations[key][:dist_num[key]] for key in dist_num]
    for sample in sample_no_rel:
        if sample!=[]:
            new_data.extend(sample)
    return new_data



def preprocess_test_data(data):
    processed_data = []
    nlp = spacy.load("en_core_sci_sm")
    nlp.add_pipe("sentencizer")
    
    for doc_id, doc_data in data.items():
        metadata = doc_data.get("metadata", {})
        abstract = metadata.get("abstract", "")
        title = metadata.get("title", "")
        entities = doc_data.get("entities", [])
        title_doc = nlp(title)
        abstract_doc = nlp(abstract)
        
        
        for e in entities: #need this for preprocess_test_data
            for e1 in entities:
                if e != e1:
                    subject_start = e.get("start_idx")
                    subject_end = e.get("end_idx")
                    object_start = e1.get("start_idx")
                    object_end = e1.get("end_idx")
                    subject_location = e.get("location")
                    object_location = e1.get("location")
                    sample = ""
                    relative_subject_start = -1
                    relative_subject_end = -1
                    relative_object_start = -1
                    relative_object_end = -1
                    subject_label = e.get("label", "")
                    object_label = e1.get("label", "")
                    if(subject_label, object_label) not in valid_relations:
                        print(f"{subject_label}, {object_label}")
                        continue
                    if subject_location == "abstract":
                        for span in abstract_doc.sents:
                            if span.start_char<= subject_start and span.end_char>=subject_end+1:
                                relative_subject_start = len(sample)+(subject_start-span.start_char)
                                relative_subject_end = len(sample)+(subject_end-span.start_char)
                                sample+=span.text
                    elif subject_location == "title":
                        for span in title_doc.sents:
                            if span.start_char<=subject_start and span.end_char>=subject_end+1:
                                relative_subject_start = len(sample)+(subject_start-span.start_char)
                                relative_subject_end = len(sample)+(subject_end-span.start_char)
                                sample+=span.text
                    if object_location == "abstract":
                        for span in abstract_doc.sents:
                            if span.start_char<= object_start and span.end_char>=object_end+1 and sample!=span.text:
                                relative_object_start = len(sample)+(object_start-span.start_char)
                                relative_object_end = len(sample)+(object_end-span.start_char)
                                sample+=span.text
                            elif sample == span.text:
                                relative_object_start = (object_start-span.start_char)
                                relative_object_end = (object_end-span.start_char)

                    elif object_location=="title":
                        for span in title_doc.sents:
                            if span.start_char<=object_start and span.end_char>=object_end+1 and sample!=span.text:
                                relative_object_start = len(sample)+(object_start-span.start_char)
                                relative_object_end = len(sample)+(object_end-span.start_char)
                                sample+=span.text
                            elif sample == span.text:
                                relative_object_start = (object_start-span.start_char)
                                relative_object_end = (object_end-span.start_char)

                    processed_data.append({
                        "sample": sample,
                        "subject": e.get("text_span", ""),
                        "subject_label": e.get("label", ""),
                        "object": e1.get("text_span", ""),
                        "object_label": e1.get("label", ""),
                        # "relation": "NONE", #there's still no relation FOR TEST WE DON'T KNOW THIS
                        "relative_subject_start": relative_subject_start,
                        "relative_subject_end": relative_subject_end,
                        "relative_object_start": relative_object_start,
                        "relative_object_end": relative_object_end,
                        "doc_id":doc_id
                    })                       
                    print(f"({subject_label}, {object_label})")
    return processed_data

    

    

# Save processed data using the 3rd arg 
output_file = sys.argv[2]
train_or_test = sys.argv[3]
with open(output_file, "w", encoding="utf-8") as file:
    
    if train_or_test == 'train':        
        processed_test_output = preprocess_train_data(data)
        downsampled_processed_output = downsample_none(processed_test_output, .5)
        json.dump(downsampled_processed_output, file, indent=4)
        
    elif train_or_test == 'test':
        
        processed_test_output = preprocess_test_data(data)
        json.dump(processed_test_output, file, indent=4)
    else:
        print("Invalid argument for train_or_test. Please use 'train' or 'test'.")
        sys.exit(1)

print(f"Processed data saved to {output_file}")



    
    
