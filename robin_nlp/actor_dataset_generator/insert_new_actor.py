import nltk
import numpy as np
import re


def generate_all_windows(lst_length, target_idx, window_size):
    """
    Generates all possible windows of fixed length that contain the target index.
    
    Args:
        lst_length: Length of the original list
        target_idx: Index that must be included in each window
        window_size: Total size of the window
        
    Returns:
        List of lists, where each inner list contains indices forming a valid window
    """
    
    # Calculate the earliest and latest possible start positions for windows
    earliest_start = max(0, target_idx - (window_size - 1))
    latest_start = min(lst_length - window_size, target_idx)
    
    # Generate all possible windows
    all_windows = []
    for start in range(earliest_start, latest_start + 1):
        end = start + window_size
        if end <= lst_length:
            window = list(range(start, end))
            # Verify target_idx is in this window
            if target_idx in window:
                all_windows.append(window)
    
    return all_windows


# Pre-compiled patterns for efficiency
MALE_PATTERN = re.compile(r'\{actor_\d+_full\}')
FEMALE_PATTERN = re.compile(r'\{actress_\d+_full\}')
BOTH_PATTERN = re.compile(r'\{(actor|actress)_\d+_full\}')

def find_valid_templates(sentence, gender):
    """Find templates that contain gender_NUMBER_full within a single template"""
    pattern = BOTH_PATTERN if gender == "both" else \
              MALE_PATTERN if gender == "male" else \
              FEMALE_PATTERN
    return bool(pattern.search(sentence))


def insert_replacement_actor(templated_review: str, 
                           name_mapping_dict: dict, 
                           replacement_actor_name: str = False, 
                           replacement_actor_gender: str = False, 
                           context_window_size: int = 1,
                           shortcut_only_full : bool=False) -> tuple[str, str]:
    """
    Inserts a replacement actor name into a templated review within a specified context window.
    
    Args:
        templated_review: Review text with name templates
        name_mapping_dict: Dictionary mapping template keys to actor names
        replacement_actor_name: New actor name to insert (optional)
        replacement_actor_gender: Gender of new actor (optional)
        context_window_size: Size of context window around name mention
        
    Returns:
        tuple: (populated_review, populated_review_og, used_template_key)
    """
    assert context_window_size > 0, "Window size must be a positive integer - it is now the total length of the window"
    
    sentences = nltk.sent_tokenize(templated_review)
    total_sentences = len(sentences)
    
    # Flatten the nested dictionary of name templates
    flattened_name_templates = {k:v for item_dic in name_mapping_dict.values() for k,v in item_dic.items()}
    
    if replacement_actor_gender == False:
        # If we are not inserting a new actor but the old ones, take gender as random
        replacement_actor_gender = np.random.choice(["male", "female"])
    
    # Get the substring how we can find templates in the review    
    gender_template_prefix = "{actor_" if replacement_actor_gender == "male" else "{actress_"
    
    ## Step 1 : Find sentence range
    if len(sentences) < (context_window_size):
        # If the review is too short, use all sentences
        window_indices = range(0, total_sentences)
    else:
        # 1.1. Find all sentences with full_names  
        candidate_sentence_indices = [
            idx for idx, sent in enumerate(sentences) 
            if find_valid_templates(sent, replacement_actor_gender)
        ]
        
        # 1.2. Choose a random sentence
        selected_sentence_index = np.random.choice(candidate_sentence_indices) if candidate_sentence_indices else None
        
        if selected_sentence_index is None:
            # Could not find a name
            selected_sentence_index = np.random.randint(0, total_sentences)
           
        # 1.3. Choose a random window of context_window_size around it
        all_windows = generate_all_windows(total_sentences, selected_sentence_index, context_window_size)
        rnd_idx = np.random.choice(len(all_windows))
        window_indices = all_windows[rnd_idx]
    
    # Stitch the sentences together that are in the selected window
    windowed_review_text = " ".join([sentences[i] for i in window_indices])
       
    ## Step 2: Find the actor name variants we are inserted in    
    template_prefix = gender_template_prefix[1:]
    
    # 2.1 find the name_templates that are in the review
    matched_templates = [k for k in flattened_name_templates.keys() if k in windowed_review_text]
    active_templates = {k:v for k,v in flattened_name_templates.items() if k in matched_templates}
    full_name_templates = [k for k in matched_templates 
                         if (template_prefix in k) and ("_full" in k)]
    
    selected_full_template = np.random.choice(full_name_templates) if full_name_templates else False
    
    populated_review_og = windowed_review_text.format(**active_templates)
    if not replacement_actor_name or not selected_full_template:
        return populated_review_og, populated_review_og, False
   
    # 2.2 find the name_templates that we are inserting
    # find all items with same start of string as selected_full_template
    template_base_key = "_".join(selected_full_template.split("_")[:2]) # e.g. actor_0
    related_templates = [k for k in matched_templates if k.startswith(template_base_key)]

    # If we only want to insert full names (easier for shortcut analysis)
    if shortcut_only_full:
        related_templates = [k for k in related_templates if "_full" in k]

    # replace the right actor names in the matched_templates
    parts = replacement_actor_name.split()
    name_components = {
        "_full": replacement_actor_name,
        "_first": parts[0],
        "_last": parts[-1]
    }
    
    for template_name in related_templates:
        possible_suffix = ["_full", "_first", "_last"]
        for suffix in possible_suffix:
            if template_name.endswith(suffix):
                active_templates[template_name] = name_components[suffix]
                
    ## Step 3: Replace the actor names in the selected review
    populated_review = windowed_review_text.format(**active_templates)
    return populated_review, populated_review_og, template_base_key

