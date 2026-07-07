# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 06:12:12 2025

@author: leona
"""

import fitz  # PyMuPDF
import re

def get_full_text(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = doc.get_text()
    return full_text
    
def extract_abstract_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""

    # Combine text from the first few pages (abstract is usually at the beginning)
    for page_num in range(min(5, len(doc))):
        full_text += doc[page_num].get_text()

    # Use regex or simple keyword-based slicing
    match = re.search(r'(?i)(abstract)\s*[:\n]?\s*(.*?)\n(?=\w{2,20}\s*[:\n])', full_text, re.DOTALL)
    
    if match:
        abstract_text = match.group(2).strip()
        return full_text, abstract_text
    else:
        return "Abstract not found."

# Example usage
pdf_file = "../TDA RESOURCES - Literature/Using Topological Data Analysis and Machine Learning to Predict Customer Churn.pdf"
doc = get_full_text(pdf_file)
abstract = extract_abstract_from_pdf(pdf_file)
