"""
Coding-specific training datasets and utilities
"""
import json
import os
from datetime import datetime
from app import db
from models import CodingDataset

class CodingTrainingService:
    def __init__(self):
        self.data_dir = "data/coding"
        os.makedirs(self.data_dir, exist_ok=True)
    
    def create_python_dataset(self):
        """Create a Python coding dataset"""
        python_examples = [
            {
                "instruction": "Write a function to calculate the factorial of a number",
                "input": "def factorial(n):",
                "output": """def factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n - 1)"""
            },
            {
                "instruction": "Create a function to find the maximum element in a list",
                "input": "def find_max(lst):",
                "output": """def find_max(lst):
    if not lst:
        return None
    max_val = lst[0]
    for item in lst[1:]:
        if item > max_val:
            max_val = item
    return max_val"""
            },
            {
                "instruction": "Write a function to check if a string is a palindrome",
                "input": "def is_palindrome(s):",
                "output": """def is_palindrome(s):
    s = s.lower().replace(' ', '')
    return s == s[::-1]"""
            },
            {
                "instruction": "Create a class for a simple bank account",
                "input": "class BankAccount:",
                "output": """class BankAccount:
    def __init__(self, initial_balance=0):
        self.balance = initial_balance
    
    def deposit(self, amount):
        if amount > 0:
            self.balance += amount
            return True
        return False
    
    def withdraw(self, amount):
        if 0 < amount <= self.balance:
            self.balance -= amount
            return True
        return False
    
    def get_balance(self):
        return self.balance"""
            },
            {
                "instruction": "Write a function to merge two sorted lists",
                "input": "def merge_sorted_lists(list1, list2):",
                "output": """def merge_sorted_lists(list1, list2):
    merged = []
    i, j = 0, 0
    
    while i < len(list1) and j < len(list2):
        if list1[i] <= list2[j]:
            merged.append(list1[i])
            i += 1
        else:
            merged.append(list2[j])
            j += 1
    
    merged.extend(list1[i:])
    merged.extend(list2[j:])
    
    return merged"""
            }
        ]
        
        dataset_path = os.path.join(self.data_dir, "python_basics.json")
        with open(dataset_path, 'w') as f:
            json.dump(python_examples, f, indent=2)
        
        return self.register_dataset(
            name="Python Basics",
            description="Basic Python programming examples for code completion",
            language="Python",
            dataset_type="code_completion",
            file_path=dataset_path
        )
    
    def create_javascript_dataset(self):
        """Create a JavaScript coding dataset"""
        js_examples = [
            {
                "instruction": "Write a function to reverse a string",
                "input": "function reverseString(str) {",
                "output": """function reverseString(str) {
    return str.split('').reverse().join('');
}"""
            },
            {
                "instruction": "Create a function to check if a number is prime",
                "input": "function isPrime(num) {",
                "output": """function isPrime(num) {
    if (num <= 1) return false;
    if (num <= 3) return true;
    if (num % 2 === 0 || num % 3 === 0) return false;
    
    for (let i = 5; i * i <= num; i += 6) {
        if (num % i === 0 || num % (i + 2) === 0) {
            return false;
        }
    }
    return true;
}"""
            },
            {
                "instruction": "Write a function to find duplicate elements in an array",
                "input": "function findDuplicates(arr) {",
                "output": """function findDuplicates(arr) {
    const seen = new Set();
    const duplicates = new Set();
    
    for (const item of arr) {
        if (seen.has(item)) {
            duplicates.add(item);
        } else {
            seen.add(item);
        }
    }
    
    return Array.from(duplicates);
}"""
            },
            {
                "instruction": "Create a debounce function",
                "input": "function debounce(func, delay) {",
                "output": """function debounce(func, delay) {
    let timeoutId;
    return function(...args) {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => func.apply(this, args), delay);
    };
}"""
            }
        ]
        
        dataset_path = os.path.join(self.data_dir, "javascript_basics.json")
        with open(dataset_path, 'w') as f:
            json.dump(js_examples, f, indent=2)
        
        return self.register_dataset(
            name="JavaScript Basics",
            description="Basic JavaScript programming examples",
            language="JavaScript",
            dataset_type="code_completion",
            file_path=dataset_path
        )
    
    def create_bug_fixing_dataset(self):
        """Create a dataset for bug fixing training"""
        bug_examples = [
            {
                "instruction": "Fix the bug in this Python function",
                "input": """def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)  # Bug: division by zero""",
                "output": """def calculate_average(numbers):
    if not numbers:
        return 0
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)"""
            },
            {
                "instruction": "Fix the infinite loop in this code",
                "input": """def countdown(n):
    while n > 0:
        print(n)
        # Bug: n is never decremented""",
                "output": """def countdown(n):
    while n > 0:
        print(n)
        n -= 1"""
            },
            {
                "instruction": "Fix the index error in this function",
                "input": """def get_last_element(lst):
    return lst[len(lst)]  # Bug: index out of range""",
                "output": """def get_last_element(lst):
    if not lst:
        return None
    return lst[len(lst) - 1]"""
            }
        ]
        
        dataset_path = os.path.join(self.data_dir, "bug_fixing.json")
        with open(dataset_path, 'w') as f:
            json.dump(bug_examples, f, indent=2)
        
        return self.register_dataset(
            name="Bug Fixing",
            description="Examples of common programming bugs and their fixes",
            language="Python",
            dataset_type="bug_fixing",
            file_path=dataset_path
        )
    
    def register_dataset(self, name, description, language, dataset_type, file_path):
        """Register a dataset in the database"""
        try:
            dataset = CodingDataset(
                name=name,
                description=description,
                language=language,
                dataset_type=dataset_type,
                file_path=file_path
            )
            
            db.session.add(dataset)
            db.session.commit()
            
            return dataset
            
        except Exception as e:
            db.session.rollback()
            return None
    
    def get_datasets(self):
        """Get all available coding datasets"""
        return CodingDataset.query.all()
    
    def initialize_default_datasets(self):
        """Initialize default coding datasets if they don't exist"""
        if CodingDataset.query.count() == 0:
            self.create_python_dataset()
            self.create_javascript_dataset()
            self.create_bug_fixing_dataset()
            print("Created default coding datasets")

# Initialize coding training service
coding_service = CodingTrainingService()