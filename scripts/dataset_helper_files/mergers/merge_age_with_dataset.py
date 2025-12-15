import json

# Load age extractions into a dictionary
age_data = {}
with open('Metadata Extraction Files/age_extractions_JSNOL_results/age_extractions.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        age_case = json.loads(line)
        # Use both case_id and case_no as keys
        case_id = age_case.get('case_id')
        case_no = age_case.get('case_no')
        
        # Store age_info
        if case_id:
            age_data[case_id] = age_case.get('age_info', {})
        if case_no:
            age_data[case_no] = age_case.get('age_info', {})

print(f"Loaded {len(age_data)} age records")

# Merge with train_with_gender.jsonl
merged_count = 0
with open('dataset/train_with_gender.jsonl', 'r', encoding='utf-8') as f_in, \
     open('dataset/train_with_gender_and_age.jsonl', 'w', encoding='utf-8') as f_out:
    
    for line in f_in:
        case = json.loads(line)
        
        # Try to find age_info by case_id or case_no
        case_id = case.get('case_id')
        case_no = case.get('case_no')
        
        age_info = None
        if case_id in age_data:
            age_info = age_data[case_id]
            merged_count += 1
        elif case_no in age_data:
            age_info = age_data[case_no]
            merged_count += 1
        
        # Add age_info to case
        if age_info:
            case['age_info'] = age_info
        else:
            case['age_info'] = {}  # Empty dict if no age data
        
        # Write merged case
        f_out.write(json.dumps(case) + '\n')

print(f"\nMerged {merged_count} cases with age data")
print(f"Created: dataset/train_with_gender_and_age.jsonl")