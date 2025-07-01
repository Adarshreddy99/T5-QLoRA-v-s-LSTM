#!/usr/bin/env python3
"""
Multi-Domain Dataset Generator - Real Dataset Sources Only
Creates a combined JSON dataset from actual datasets:
- Technical: Stack Overflow data from Hugging Face (expanded and verified sources)
- Medical: PubMed QA data from Hugging Face  
- Financial: Financial datasets from Hugging Face
- General: SQuAD dataset from Hugging Face

Total: ~12,000 examples (3000 per domain, targeted from a shuffled stream for main dataset)
Evaluation: ~100 examples (25 per domain, targeted from a shuffled stream, separate from main)
"""

import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import random
from datetime import datetime
import os
import time
import warnings

# Suppress specific warnings from Hugging Face datasets library
warnings.filterwarnings('ignore', module='datasets.load')
warnings.filterwarnings('ignore', category=FutureWarning)

try:
    from datasets import load_dataset
    from tqdm import tqdm # For local execution, use standard tqdm
except ImportError:
    print("Required packages not found. Installing now...")
    # This block is for convenience; normally you'd run `pip install -r requirements.txt`
    # or list these directly in your terminal before running the script.
    os.system("pip install datasets transformers pandas numpy tqdm")
    from datasets import load_dataset
    from tqdm import tqdm

class MultiDomainDatasetGenerator:
    def __init__(self):
        self.combined_data = [] # For main dataset
        self.evaluation_data = [] # For evaluation dataset
        self.domain_counts = { # For main dataset counts
            'technical': 0,
            'medical': 0, 
            'financial': 0,
            'general': 0
        }
        self.eval_domain_counts = { # For evaluation dataset counts
            'technical': 0,
            'medical': 0,
            'financial': 0,
            'general': 0
        }
        self.seed = 42 # For reproducible shuffling
        self.main_per_domain = 3000
        self.eval_per_domain = 25
        self.total_fetch_per_domain = self.main_per_domain + self.eval_per_domain

    def fetch_stackoverflow_data(self) -> List[Dict]:
        """Fetch technical Q&A data from high-quality, verified technical and instruction-tuned datasets."""
        print("\nFetching Technical data (OpenHermes, Dolly, Stack Exchange, Code Alpaca, Stack Overflow)...")
        
        # Prioritize high-quality, diverse instruction-tuned datasets with significant technical content
        dataset_options = [   
            
            ##("teknium/OpenHermes-2.5", "train"), # Very large, diverse, high-quality. Should cover technical broadly.
            ("databricks/databricks-dolly-15k", "train"), # High-quality, human-generated instructions, includes technical topics.
            ("WizardLMTeam/WizardLM_evol_instruct_V2_196k", "train"), # Explicitly coding instructions, high quality, identified as the underlying data for WizardCoder.
            ("iamtarun/code_instructions_120k_alpaca", "train"), # Code instructions in Alpaca format.
            ("HuggingFaceH4/CodeAlpaca_20k", "instruction"), # Code-related instructions        # Very large, diverse, high-quality instruction dataset
            ("databricks/databricks-dolly-15k", "train"),    # High-quality, human-generated instructions, includes technical topics
            ("TokenBender/code_instructions_120k", "train"),
            # Specific Stack Exchange / Code Q&A
            ("ArmelR/stack-exchange-instructions", "train"), # Broad Stack Exchange Q&A, instruction-tuned format
            ("HuggingFaceH4/CodeAlpaca_20k", "train"),       # Code-related instructions
            
            # Original Stack Overflow extracts (as fallback if others don't yield enough)
            ("pacovaldez/stackoverflow-questions", "train"), # General Stack Overflow Q&A
            ("koutch/stackoverflow_python", "train"),        # Focused on Python Stack Overflow
        ]
        
        all_fetched_data = []
        
        for dataset_name, split in dataset_options:
            try:
                print(f"Trying dataset: {dataset_name}")
                # Load in streaming mode, shuffle, and then take only the total required count
                dataset = load_dataset(dataset_name, split=split, streaming=True)
                limited_dataset_stream = dataset.shuffle(seed=self.seed).take(self.total_fetch_per_domain)
                
                technical_data = []
                
                for item in tqdm(limited_dataset_stream, desc=f"Processing {dataset_name}", total=self.total_fetch_per_domain):
                    question = None
                    answer = None
                    context = ""
                    tags = []
                    
                    # Extract question (prioritize 'instruction', 'question', 'title', then 'body')
                    if 'instruction' in item and item['instruction']: # For instruction-tuned datasets
                        question = str(item['instruction']).strip()
                    elif 'question' in item and item['question']:
                        question = str(item['question']).strip()
                    elif 'title' in item and item['title']:
                        question = str(item['title']).strip()
                    elif 'body' in item and item['body']:
                        # Use first sentence or summary of body if question is not explicit
                        question = str(item['body']).split('.')[0].strip() + ("?" if not str(item['body']).split('.')[0].strip().endswith('?') else '')
                        if len(question) > 200: # Truncate if too long
                            question = question[:197] + "..."
                    
                    # Extract answer (prioritize 'output', 'answer', 'accepted_answer')
                    if 'output' in item and item['output']: # For instruction-tuned datasets
                        answer = str(item['output']).strip()
                    elif 'answer' in item and item['answer']:
                        answer = str(item['answer']).strip()
                    elif 'accepted_answer' in item and item['accepted_answer']:
                        answer = str(item['accepted_answer']).strip()
                    elif 'answers' in item and isinstance(item['answers'], list) and len(item['answers']) > 0:
                        # Take the first answer if it's a list
                        answer = str(item['answers'][0]).strip()
                    
                    # Extract context/body
                    if 'question_body' in item and item['question_body']:
                        context = str(item['question_body']).strip()[:500]
                    elif 'body' in item and item['body'] != question: # Ensure not duplicating 'question' if it came from 'body'
                        context = str(item['body']).strip()[:500]
                    elif 'input' in item and item['input'] != question: # If instruction uses 'input' as context
                        context = str(item['input']).strip()[:500]
                    elif 'context' in item and item['context'] != question: # General 'context' field
                        context = str(item['context']).strip()[:500]

                    # Extract tags (if available)
                    if 'tags' in item:
                        # Tags can be string (e.g., "<python><django>") or list
                        if isinstance(item['tags'], str):
                            # Convert "<tag1><tag2>" to list ["tag1", "tag2"]
                            tags = [t.strip('<>').lower() for t in item['tags'].split('><') if t]
                        elif isinstance(item['tags'], list):
                            tags = [str(t).lower() for t in item['tags']]
                        tags = tags[:5] # Limit tags for conciseness
                    
                    # Basic validation and length limits
                    if (question and answer and 
                        len(question) >= 10 and len(question) <= 500 and
                        len(answer) >= 20 and len(answer) <= 2000):
                        
                        entry = {
                            'id': f'tech_{len(technical_data):06d}', # ID will be re-assigned later
                            'domain': 'technical',
                            'question': question,
                            'answer': answer, # Keep for analysis, will be removed for final output
                            'context': context,
                            'tags': tags,
                            'source': dataset_name,
                            'timestamp': datetime.now().isoformat()
                        }
                        technical_data.append(entry)
                    
                all_fetched_data.extend(technical_data)
                
                # If we've collected enough unique examples across all attempts for this domain, break early
                if len(all_fetched_data) >= self.total_fetch_per_domain:
                    print(f"Successfully collected {len(all_fetched_data)} raw technical examples from {dataset_name} and previous sources.")
                    return all_fetched_data[:self.total_fetch_per_domain]
                    
            except Exception as e:
                print(f"Failed to load or process {dataset_name}: {e}")
                print(f"Moving to next dataset option for technical domain...")
                continue
        
        # Fallback if not enough real data could be loaded or processed
        print(f"Could not load enough technical data. Generating {len(all_fetched_data)} real + synthetic fallback.")
        synthetic_count = self.total_fetch_per_domain - len(all_fetched_data)
        if synthetic_count > 0:
            synthetic_data = [
                {"id": "tech_synth_000", "domain": "technical", "question": "What is object-oriented programming?", "answer": "Object-oriented programming (OOP) is a programming paradigm based on the concept of \"objects\", which can contain data and code.", "context": "", "tags": ["oop", "programming"], "source": "synthetic", "timestamp": datetime.now().isoformat()},
                {"id": "tech_synth_001", "domain": "technical", "question": "How to resolve 'ModuleNotFoundError' in Python?", "answer": "This error typically means Python can't find a module you're trying to import. Ensure the module is installed (e.g., `pip install module_name`) and available in your Python environment's path.", "context": "", "tags": ["python", "error"], "source": "synthetic", "timestamp": datetime.now().isoformat()}
            ] * ((synthetic_count // 2) + 1) # Ensure we have enough synthetic
            all_fetched_data.extend(random.sample(synthetic_data, synthetic_count)) # Take exact synthetic count
        return all_fetched_data[:self.total_fetch_per_domain]

    def fetch_pubmed_qa_data(self) -> List[Dict]:
        """Fetch medical Q&A data from PubMed QA and related datasets."""
        print("\nFetching PubMed QA data...")
        
        dataset_options = [
            ("pubmed_qa", "pqa_artificial", "train"), # Largest, most synthetic
            ("pubmed_qa", "pqa_labeled", "train"),    # Smaller, human-labeled
            ("medalpaca/medical_meadow_pubmed_causal", None, "train"), # Instruction-following medical data
            ("lavita/medical-qa-datasets", None, "train") # Various medical QA
        ]
        
        all_fetched_data = []
        
        for dataset_info in dataset_options:
            try:
                dataset_name = dataset_info[0]
                config = dataset_info[1]
                split = dataset_info[2]
                
                print(f"Trying dataset: {dataset_name}" + (f" config: {config}" if config else ""))
                
                if config:
                    dataset = load_dataset(dataset_name, config, split=split, streaming=True)
                else:
                    dataset = load_dataset(dataset_name, split=split, streaming=True)
                
                # Shuffle and take only the total required count
                limited_dataset_stream = dataset.shuffle(seed=self.seed).take(self.total_fetch_per_domain)
                
                medical_data = []
                
                for item in tqdm(limited_dataset_stream, desc=f"Processing {dataset_name}", total=self.total_fetch_per_domain):
                    question = None
                    answer = None
                    context = ""
                    
                    # Extract question
                    if 'question' in item and item['question']:
                        question = str(item['question']).strip()
                    elif 'input' in item and item['input']: # For instruction-tuned formats
                        question = str(item['input']).strip()
                    elif 'instruction' in item and item['instruction']:
                        question = str(item['instruction']).strip()
                    
                    # Extract answer
                    if 'long_answer' in item and item['long_answer']:
                        answer = str(item['long_answer']).strip()
                    elif 'answer' in item and item['answer']:
                        answer = str(item['answer']).strip()
                    elif 'output' in item and item['output']: # For instruction-tuned formats
                        answer = str(item['output']).strip()
                    elif 'response' in item and item['response']:
                        answer = str(item['response']).strip()
                    
                    # Extract context
                    if 'context' in item:
                        if isinstance(item['context'], dict) and 'contexts' in item['context']:
                            context = ' '.join(item['context']['contexts'][:3]).strip() # Limit context parts
                        elif isinstance(item['context'], str):
                            context = item['context'].strip()
                        elif isinstance(item['context'], list):
                            context = ' '.join(item['context'][:3]).strip()
                        context = context[:800] # Truncate context string
                    
                    # Basic validation
                    if (question and answer and 
                        len(question) >= 10 and len(question) <= 500 and 
                        len(answer) >= 20 and len(answer) <= 2000):
                        
                        entry = {
                            'id': f'med_{len(medical_data):06d}', # ID will be re-assigned later
                            'domain': 'medical',
                            'question': question,
                            'answer': answer, # Keep for analysis
                            'context': context,
                            'final_decision': str(item.get('final_decision', '')).strip(), # Specific to PubMedQA
                            'source': dataset_name + (f"_{config}" if config else ""),
                            'timestamp': datetime.now().isoformat()
                        }
                        medical_data.append(entry)
                    
                all_fetched_data.extend(medical_data)
                
                if len(all_fetched_data) >= self.total_fetch_per_domain:
                    print(f"Successfully collected {len(all_fetched_data)} raw medical examples from {dataset_name} and previous sources.")
                    return all_fetched_data[:self.total_fetch_per_domain]
                    
            except Exception as e:
                print(f"Failed to load or process {dataset_name}: {e}")
                print(f"Moving to next dataset option for medical domain...")
                continue
        
        print(f"Could not load enough medical data. Generating {len(all_fetched_data)} real + synthetic fallback.")
        synthetic_count = self.total_fetch_per_domain - len(all_fetched_data)
        if synthetic_count > 0:
            synthetic_data = [
                {"id": "med_synth_000", "domain": "medical", "question": "What are the common symptoms of the flu?", "answer": "Common flu symptoms include fever, cough, sore throat, muscle aches, and fatigue. It's important to differentiate from a common cold.", "context": "", "final_decision": "yes", "source": "synthetic", "timestamp": datetime.now().isoformat()},
                {"id": "med_synth_001", "domain": "medical", "question": "How does vaccination work?", "answer": "Vaccination introduces a weakened or inactive form of a pathogen (or its components) into the body, stimulating the immune system to produce antibodies and memory cells without causing the disease. This prepares the body to fight off future infections more effectively.", "context": "", "final_decision": "yes", "source": "synthetic", "timestamp": datetime.now().isoformat()}
            ] * ((synthetic_count // 2) + 1)
            all_fetched_data.extend(random.sample(synthetic_data, synthetic_count))
        return all_fetched_data[:self.total_fetch_per_domain]


    def fetch_financial_data(self) -> List[Dict]:
        """Fetch financial Q&A data from financial datasets from Hugging Face."""
        print("\nFetching Financial data...")
        
        dataset_options = [
            ("ChanceFocus/flare-finqa", None, "train"),            # Financial QA with table data
            ("sujet-ai/Sujet-Finance-Instruct-177k", None, "train"), # Instruction-tuned finance
            ("gbharti/finance-alpaca", None, "train"),              # Alpaca-style finance instructions
            ("FinGPT/fingpt-finqa", None, "train"),                # Financial QA
            ("pauri32/fiqa-2018", None, "train")                    # Financial opinion mining
        ]
        
        all_fetched_data = []
        
        for dataset_info in dataset_options:
            try:
                dataset_name = dataset_info[0]
                config = dataset_info[1] 
                split = dataset_info[2]
                
                print(f"Trying dataset: {dataset_name}" + (f" config: {config}" if config else ""))
                
                if config:
                    dataset = load_dataset(dataset_name, config, split=split, streaming=True)
                else:
                    dataset = load_dataset(dataset_name, split=split, streaming=True)

                # Shuffle and take only the total required count
                limited_dataset_stream = dataset.shuffle(seed=self.seed).take(self.total_fetch_per_domain)
                
                financial_data = []
                
                for item in tqdm(limited_dataset_stream, desc=f"Processing {dataset_name}", total=self.total_fetch_per_domain):
                    question = None
                    answer = None
                    context = ""
                    
                    # Extract question
                    if 'question' in item and item['question']:
                        question = str(item['question']).strip()
                    elif 'input' in item and item['input']: # For instruction-tuned formats
                        question = str(item['input']).strip()
                    elif 'instruction' in item and item['instruction']:
                        question = str(item['instruction']).strip()
                    elif 'query' in item and item['query']: # Specific to fiqa-2018
                        question = str(item['query']).strip()
                    
                    # Extract answer
                    if 'answer' in item and item['answer']:
                        answer = str(item['answer']).strip()
                    elif 'output' in item and item['output']: # For instruction-tuned formats
                        answer = str(item['output']).strip()
                    elif 'response' in item and item['response']:
                        answer = str(item['response']).strip()
                    elif 'explanation' in item and item['explanation']:
                        answer = str(item['explanation']).strip()
                    elif 'text' in item and 'sentiment' in item and item['text']: # For sentiment datasets like fiqa-2018 headlines
                        question = str(item['text']).strip()
                        answer = f"The sentiment is: {str(item['sentiment']).strip()}"
                    
                    # Extract context
                    context_parts = []
                    if 'pre_text' in item and item['pre_text']:
                        context_parts.append(str(item['pre_text']))
                    if 'post_text' in item and item['post_text']:
                        context_parts.append(str(item['post_text']))
                    if 'context' in item and item['context']:
                        context_parts.append(str(item['context']))
                    if 'passage' in item and item['passage']: # For FinQA table data
                        context_parts.append(str(item['passage']))
                    context = ' '.join(context_parts).strip()[:800] # Truncate context string
                    
                    # Basic validation
                    if (question and answer and 
                        len(question) >= 10 and len(question) <= 500 and 
                        len(answer) >= 10 and len(answer) <= 2000):
                        
                        entry = {
                            'id': f'fin_{len(financial_data):06d}', # ID will be re-assigned later
                            'domain': 'financial',
                            'question': question,
                            'answer': answer, # Keep for analysis
                            'context': context,
                            'table_data': str(item.get('table', '')).strip()[:500], # Specific to FinQA
                            'source': dataset_name,
                            'timestamp': datetime.now().isoformat()
                        }
                        financial_data.append(entry)
                    
                all_fetched_data.extend(financial_data)

                if len(all_fetched_data) >= self.total_fetch_per_domain:
                    print(f"Successfully collected {len(all_fetched_data)} raw financial examples from {dataset_name} and previous sources.")
                    return all_fetched_data[:self.total_fetch_per_domain]
                    
            except Exception as e:
                print(f"Failed to load or process {dataset_name}: {e}")
                print(f"Moving to next dataset option for financial domain...")
                continue
        
        print(f"Could not load enough financial data. Generating {len(all_fetched_data)} real + synthetic fallback.")
        synthetic_count = self.total_fetch_per_domain - len(all_fetched_data)
        if synthetic_count > 0:
            synthetic_data = [
                {"id": "fin_synth_000", "domain": "financial", "question": "What is a bond in finance?", "answer": "A bond is a fixed-income instrument that represents a loan made by an investor to a borrower (typically corporate or governmental).", "context": "", "table_data": "", "source": "synthetic", "timestamp": datetime.now().isoformat()},
                {"id": "fin_synth_001", "domain": "financial", "question": "Explain cryptocurrency volatility.", "answer": "Cryptocurrency volatility refers to how rapidly the price of a cryptocurrency can change. It's influenced by factors like market sentiment, regulation, and adoption, making it highly unpredictable.", "context": "", "table_data": "", "source": "synthetic", "timestamp": datetime.now().isoformat()}
            ] * ((synthetic_count // 2) + 1)
            all_fetched_data.extend(random.sample(synthetic_data, synthetic_count))
        return all_fetched_data[:self.total_fetch_per_domain]

    def fetch_squad_data(self) -> List[Dict]:
        """Fetch general Q&A data from SQuAD dataset and similar."""
        print("\nFetching SQuAD data...")
        
        dataset_options = [
            ("squad", None, "train"),
            ("squad_v2", None, "train"), # Includes unanswerable questions, might need filtering if you only want answerable ones
            ("rajpurkar/squad_v1", None, "train"),
            ("microsoft/ms_marco", "v1.1", "train"), # Microsoft Machine Reading Comprehension
            ("natural_questions", None, "train") # Google Natural Questions
        ]
        
        all_fetched_data = []
        
        for dataset_info in dataset_options:
            try:
                dataset_name = dataset_info[0]
                config = dataset_info[1]
                split = dataset_info[2]
                
                print(f"Trying dataset: {dataset_name}" + (f" config: {config}" if config else ""))
                
                if config:
                    dataset = load_dataset(dataset_name, config, split=split, streaming=True)
                else:
                    dataset = load_dataset(dataset_name, split=split, streaming=True)

                # Shuffle and take only the total required count
                limited_dataset_stream = dataset.shuffle(seed=self.seed).take(self.total_fetch_per_domain)
                
                general_data = []
                
                for item in tqdm(limited_dataset_stream, desc=f"Processing {dataset_name}", total=self.total_fetch_per_domain):
                    question = None
                    answer = None
                    context = ""
                    title = ""
                    all_answers = []
                    
                    # Extract question
                    if 'question' in item and item['question']:
                        question = str(item['question']).strip()
                    elif 'query' in item and item['query']: # Specific to MS MARCO/Natural Questions
                        question = str(item['query']).strip()
                    
                    # Extract answer
                    if 'answers' in item:
                        if isinstance(item['answers'], dict) and 'text' in item['answers']:
                            answers_list = item['answers']['text']
                            if answers_list and len(answers_list) > 0:
                                answer = str(answers_list[0]).strip()
                                all_answers = [str(a).strip() for a in answers_list[:5]] # Limit to 5 answers
                        elif isinstance(item['answers'], list) and len(item['answers']) > 0:
                            answer = str(item['answers'][0]).strip()
                            all_answers = [str(a).strip() for a in item['answers'][:5]]
                    elif 'answer' in item and item['answer']: # For Natural Questions
                        answer = str(item['answer']).strip()
                        all_answers = [answer] # Assume single answer if not a list
                    elif 'response' in item and item['response']: # Catch-all for response field
                        answer = str(item['response']).strip()
                        all_answers = [answer]

                    # Extract context and title
                    if 'context' in item and item['context']:
                        context = str(item['context']).strip()[:1000]
                    if 'title' in item and item['title']:
                        title = str(item['title']).strip()
                    
                    # Filter questions that might overlap too much with other domains (basic heuristic)
                    # and ensure non-empty Q&A
                    if (question and answer and
                        len(question) >= 10 and len(question) <= 500 and
                        len(answer) >= 3 and len(answer) <= 1000 and
                        not any(keyword in question.lower() for keyword in ['code', 'program', 'software', 'hardware', 'stock', 'invest', 'financial', 'market', 'symptom', 'medical', 'disease', 'health'])):
                        
                        entry = {
                            'id': f'gen_{len(general_data):06d}', # ID will be re-assigned later
                            'domain': 'general',
                            'question': question,
                            'answer': answer, # Keep for analysis
                            'context': context,
                            'title': title,
                            'all_answers': all_answers,
                            'source': dataset_name + (f"_{config}" if config else ""),
                            'timestamp': datetime.now().isoformat()
                        }
                        general_data.append(entry)
                    
                all_fetched_data.extend(general_data)

                if len(all_fetched_data) >= self.total_fetch_per_domain:
                    print(f"Successfully collected {len(all_fetched_data)} raw general examples from {dataset_name} and previous sources.")
                    return all_fetched_data[:self.total_fetch_per_domain]
                    
            except Exception as e:
                print(f"Failed to load or process {dataset_name}: {e}")
                print(f"Moving to next dataset option for general domain...")
                continue
        
        print(f"Could not load enough general data. Generating {len(all_fetched_data)} real + synthetic fallback.")
        synthetic_count = self.total_fetch_per_domain - len(all_fetched_data)
        if synthetic_count > 0:
            synthetic_data = [
                {"id": "gen_synth_000", "domain": "general", "question": "What is the capital of Japan?", "answer": "The capital of Japan is Tokyo.", "context": "", "title": "", "all_answers": ["Tokyo"], "source": "synthetic", "timestamp": datetime.now().isoformat()},
                {"id": "gen_synth_001", "domain": "general", "question": "Tell me about the history of the internet.", "answer": "The internet originated from the Advanced Research Projects Agency Network (ARPANNET) developed by the US Department of Defense in the late 1960s.", "context": "", "title": "", "all_answers": [], "source": "synthetic", "timestamp": datetime.now().isoformat()}
            ] * ((synthetic_count // 2) + 1)
            all_fetched_data.extend(random.sample(synthetic_data, synthetic_count))
        return all_fetched_data[:self.total_fetch_per_domain]

    def combine_datasets(self) -> Dict[str, Any]:
        """Combine all datasets into a single structure"""
        print("\nCombining all datasets...")
        print("This may take several minutes depending on network speed and dataset sizes...")
        
        all_original_data = [] # Stores data for the main dataset
        evaluation_data_raw = [] # Stores data for the evaluation dataset
        
        # Using a list of (function, domain_name) pairs to iterate
        fetch_tasks = [
            (self.fetch_stackoverflow_data, 'technical'),
            (self.fetch_pubmed_qa_data, 'medical'),
            (self.fetch_financial_data, 'financial'),
            (self.fetch_squad_data, 'general')
        ]

        for fetch_func, domain_name in fetch_tasks:
            try:
                # Fetch a combined set of data for main and evaluation
                fetched_domain_data = fetch_func() # This now fetches total_fetch_per_domain examples
                
                # Ensure we have enough data to split
                if len(fetched_domain_data) < (self.main_per_domain + self.eval_per_domain):
                    print(f"Warning: Only collected {len(fetched_domain_data)} for {domain_name}. Expected {self.total_fetch_per_domain}. Filling with duplicates or missing some data.")
                
                # Shuffle the fetched data for this domain to randomize main/eval split
                random.shuffle(fetched_domain_data)

                # Split into main and evaluation sets
                main_set_for_domain = fetched_domain_data[:self.main_per_domain]
                eval_set_for_domain = fetched_domain_data[self.main_per_domain : self.main_per_domain + self.eval_per_domain]
                
                # Assign unique IDs and extend lists
                for i, item in enumerate(main_set_for_domain):
                    item['id'] = f"{domain_name[:3]}_main_{i:06d}"
                all_original_data.extend(main_set_for_domain)
                self.domain_counts[domain_name] = len(main_set_for_domain)

                for i, item in enumerate(eval_set_for_domain):
                    item['id'] = f"{domain_name[:3]}_eval_{i:03d}" # Shorter ID for eval
                evaluation_data_raw.extend(eval_set_for_domain)
                self.eval_domain_counts[domain_name] = len(eval_set_for_domain)
                
                print(f"✓ {domain_name.capitalize()} data: Main: {len(main_set_for_domain)} examples, Eval: {len(eval_set_for_domain)} examples")

            except Exception as e:
                print(f"✗ Failed to fetch {domain_name} data: {e}. Using fallback/synthetic data if any.")
                self.domain_counts[domain_name] = 0 # Mark as 0 if initial fetch failed
                self.eval_domain_counts[domain_name] = 0
                import traceback
                traceback.print_exc() # Print full traceback for debugging

        if not all_original_data and not evaluation_data_raw:
            raise Exception("No data could be fetched from any dataset!")
        
        # Shuffle the combined main dataset and evaluation dataset independently
        random.shuffle(all_original_data)
        random.shuffle(evaluation_data_raw)
        
        # Create final dataset structure for router LLM fine-tuning (main dataset)
        formatted_for_router_llm = []
        for item in all_original_data:
            formatted_for_router_llm.append({
                "text": f"User query: {item['question'].strip()}\nDomain: {item['domain']}"
            })

        main_dataset = {
            'metadata': {
                'total_examples': len(formatted_for_router_llm),
                'domain_distribution': self.domain_counts,
                'creation_date': datetime.now().isoformat(),
                'description': f'Multi-domain Q&A dataset combining technical, medical, financial, and general knowledge from real datasets, formatted for router LLM fine-tuning. Each domain targets {self.main_per_domain} examples. Each example contains "User query: [question]\\nDomain: [domain]".',
                'domains': {
                    'technical': 'Programming and software development questions from OpenHermes, Dolly, Stack Exchange, Code Alpaca, and Stack Overflow datasets',
                    'medical': 'Medical and health-related questions from PubMed QA and medical datasets',
                    'financial': 'Financial and investment questions from financial QA datasets',
                    'general': 'General knowledge questions from SQuAD and similar QA datasets'
                },
                'sources_attempted': {
                    'technical': ["teknium/OpenHermes-2.5", "databricks/databricks-dolly-15k", "ArmelR/stack-exchange-instructions", "HuggingFaceH4/CodeAlpaca_20k", "pacovaldez/stackoverflow-questions", "koutch/stackoverflow_python"],
                    'medical': ["pubmed_qa", "medalpaca/medical_meadow_pubmed_causal", "lavita/medical-qa-datasets"],
                    'financial': ["ChanceFocus/flare-finqa", "sujet-ai/Sujet-Finance-Instruct-177k", "gbharti/finance-alpaca", "FinGPT/fingpt-finqa", "pauri32/fiqa-2018"],
                    'general': ["squad", "squad_v2", "rajpurkar/squad_v1", "microsoft/ms_marco", "natural_questions"]
                }
            },
            'data': formatted_for_router_llm
        }
        
        # Create final dataset structure for router LLM fine-tuning (evaluation dataset)
        formatted_for_eval_llm = []
        for item in evaluation_data_raw:
            formatted_for_eval_llm.append({
                "text": f"User query: {item['question'].strip()}\nDomain: {item['domain']}"
            })

        evaluation_dataset = {
            'metadata': {
                'total_examples': len(formatted_for_eval_llm),
                'domain_distribution': self.eval_domain_counts,
                'creation_date': datetime.now().isoformat(),
                'description': f'Evaluation set for multi-domain Q&A, separate from main dataset. Each domain targets {self.eval_per_domain} examples. Each example contains "User query: [question]\\nDomain: [domain]".',
                'domains': {
                    'technical': 'Programming and software development questions from OpenHermes, Dolly, Stack Exchange, Code Alpaca, and Stack Overflow datasets',
                    'medical': 'Medical and health-related questions from PubMed QA and medical datasets',
                    'financial': 'Financial and investment questions from financial QA datasets',
                    'general': 'General knowledge questions from SQuAD and similar QA datasets'
                },
                'sources_attempted': {
                    'technical': ["teknium/OpenHermes-2.5", "databricks/databricks-dolly-15k", "ArmelR/stack-exchange-instructions", "HuggingFaceH4/CodeAlpaca_20k", "pacovaldez/stackoverflow-questions", "koutch/stackoverflow_python"],
                    'medical': ["pubmed_qa", "medalpaca/medical_meadow_pubmed_causal", "lavita/medical-qa-datasets"],
                    'financial': ["ChanceFocus/flare-finqa", "sujet-ai/Sujet-Finance-Instruct-177k", "gbharti/finance-alpaca", "FinGPT/fingpt-finqa", "pauri32/fiqa-2018"],
                    'general': ["squad", "squad_v2", "rajpurkar/squad_v1", "microsoft/ms_marco", "natural_questions"]
                }
            },
            'data': formatted_for_eval_llm
        }

        # Store original data for the main dataset separately for analysis
        self._original_combined_data_for_analysis = all_original_data
        
        return main_dataset, evaluation_dataset
    
    def save_dataset(self, dataset: Dict[str, Any], filename: str):
        """Save the combined dataset to JSON file"""
        print(f"\nSaving dataset to {filename}...")
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        print(f"Dataset saved successfully to {filename}!")
        print(f"Total examples: {dataset['metadata']['total_examples']}")
        print(f"Domain distribution: {dataset['metadata']['domain_distribution']}")
    
    def generate_sample_analysis(self, dataset: Dict[str, Any]):
        """Generate comprehensive analysis of the dataset"""
        print("\n" + "="*60)
        print("COMPREHENSIVE DATASET ANALYSIS (Main Dataset)")
        print("="*60)
        
        # Use the _original_combined_data_for_analysis for detailed stats
        # as the 'data' field in `dataset` is now just the 'text' string
        data_for_analysis = self._original_combined_data_for_analysis if hasattr(self, '_original_combined_data_for_analysis') else []

        if not data_for_analysis:
            print("\n⚠️ Cannot perform detailed analysis: Original data (before formatting) not available.")
            print("The final dataset 'data' field only contains the formatted 'text' strings.")
            return

        # Domain distribution (using original data's domain field)
        domain_counts = {}
        for item in data_for_analysis:
            domain = item['domain']
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        print(f"\n📊 DOMAIN DISTRIBUTION:")
        total = sum(domain_counts.values())
        for domain, count in domain_counts.items():
            percentage = (count / total) * 100
            print(f"  {domain.capitalize():<12}: {count:>5} examples ({percentage:>5.1f}%)")
        
        # Text statistics
        print(f"\n📝 TEXT STATISTICS:")
        for domain in domain_counts.keys():
            domain_data = [item for item in data_for_analysis if item['domain'] == domain]
            if domain_data:
                q_lengths = [len(item['question']) for item in domain_data]
                a_lengths = [len(item['answer']) for item in domain_data]
                
                print(f"\n  {domain.upper()}:")
                print(f"    Questions - Avg: {np.mean(q_lengths):.1f}, Min: {min(q_lengths)}, Max: {max(q_lengths)}")
                print(f"    Answers   - Avg: {np.mean(a_lengths):.1f}, Min: {min(a_lengths)}, Max: {max(a_lengths)}")
        
        # Data sources
        print(f"\n🔗 DATA SOURCES:")
        source_counts = {}
        for item in data_for_analysis:
            source = item.get('source', 'unknown')
            source_counts[source] = source_counts.get(source, 0) + 1
        
        for source, count in sorted(source_counts.items()):
            print(f"  {source:<35}: {count:>5} examples")
        
        # Sample examples
        print(f"\n💡 SAMPLE EXAMPLES (from original data format):")
        for domain in domain_counts.keys():
            domain_examples = [item for item in data_for_analysis if item['domain'] == domain]
            if domain_examples:
                sample = random.choice(domain_examples)
                print(f"\n  📌 {domain.upper()} EXAMPLE (ID: {sample['id']}):")
                print(f"     Q: {sample['question'][:150]}{'...' if len(sample['question']) > 150 else ''}")
                print(f"     A: {sample['answer'][:150]}{'...' if len(sample['answer']) > 150 else ''}")
                if sample.get('source'):
                    print(f"     Source: {sample['source']}")
        
        # Quality metrics
        print(f"\n✅ QUALITY METRICS:")
        avg_q_len = np.mean([len(item['question']) for item in data_for_analysis])
        avg_a_len = np.mean([len(item['answer']) for item in data_for_analysis])
        
        # Count items with context
        with_context = sum(1 for item in data_for_analysis if item.get('context', '').strip())
        
        print(f"  Average question length: {avg_q_len:.1f} characters")
        print(f"  Average answer length: {avg_a_len:.1f} characters")
        print(f"  Items with context: {with_context}/{len(data_for_analysis)} ({(with_context/len(data_for_analysis)*100):.1f}%)")
        print(f"  Unique domains: {len(domain_counts)}")
        print(f"  Unique sources: {len(source_counts)}")

def main():
    """Main function to generate the multi-domain dataset"""
    print("🚀 Multi-Domain Dataset Generator (Real Datasets Only)")
    print("="*60)
    print(f"Targeting {MultiDomainDatasetGenerator().main_per_domain * 4} examples ({MultiDomainDatasetGenerator().main_per_domain} per domain) for main dataset.")
    print(f"Targeting {MultiDomainDatasetGenerator().eval_per_domain * 4} examples ({MultiDomainDatasetGenerator().eval_per_domain} per domain) for evaluation dataset.")
    print("This process may take some time depending on network speed and dataset sizes...")
    print("="*60)
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Initialize generator
    generator = MultiDomainDatasetGenerator()
    
    try:
        # Generate combined dataset
        main_dataset, evaluation_dataset = generator.combine_datasets()
        
        # Save main dataset to JSON file
        generator.save_dataset(main_dataset, 'multi_domain_qa_dataset_3.json')
        
        # Save evaluation dataset to JSON file
        generator.save_dataset(evaluation_dataset, 'evaluation_multi_domain_qa_dataset.json')
        
        # Generate comprehensive analysis for the main dataset
        generator.generate_sample_analysis(main_dataset)
        
        # Save a smaller sample for quick testing from the main dataset
        if len(main_dataset['data']) > 200:
            sample_data = {
                'metadata': main_dataset['metadata'].copy(),
                'data': main_dataset['data'][:200]  # First 200 examples
            }
            sample_data['metadata']['total_examples'] = 200
            sample_data['metadata']['description'] += ' (Sample version with 200 examples)'
            
            generator.save_dataset(sample_data, 'sample_multi_domain_dataset.json')
            print(f"\n📄 Files created:")
            print(f"  ✓ multi_domain_qa_dataset.json (Main dataset: {len(main_dataset['data'])} examples)")
            print(f"  ✓ evaluation_multi_domain_qa_dataset.json (Evaluation dataset: {len(evaluation_dataset['data'])} examples)")
            print(f"  ✓ sample_multi_domain_dataset.json (Sample: 200 examples from main dataset)")
        else:
            print(f"\n📄 Files created:")
            print(f"  ✓ multi_domain_qa_dataset.json (Main dataset: {len(main_dataset['data'])} examples)")
            print(f"  ✓ evaluation_multi_domain_qa_dataset.json (Evaluation dataset: {len(evaluation_dataset['data'])} examples)")
        
        return main_dataset, evaluation_dataset
        
    except Exception as e:
        print(f"\n❌ Error generating dataset: {e}")
        print("Please check your internet connection and ensure Hugging Face datasets are accessible.")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        raise

# Run the generator
if __name__ == "__main__":
    main_dataset, evaluation_dataset = main()