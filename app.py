# app.py
# Streamlit app for Legal Document Analysis and Contract Intelligence

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import io
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import PyPDF2
import difflib
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Set page configuration
st.set_page_config(
    page_title="Legal Document Analyzer",
    page_icon="⚖️",
    layout="wide"
)

# Function to convert PDF to text
def convert_pdf_to_text(pdf_file):
    """Extract text from uploaded PDF file"""
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return None

# Function to segment legal document into logical sections
def segment_legal_document(text):
    """Segment legal document into sections based on headings"""
    # Common section titles in legal documents
    section_patterns = [
        r"(?i)^(.*?agreement)$",
        r"(?i)^(.*?definitions)$",
        r"(?i)^(.*?term)$",
        r"(?i)^(.*?termination)$",
        r"(?i)^(.*?compensation)$",
        r"(?i)^(.*?payment)$",
        r"(?i)^(.*?obligations)$",
        r"(?i)^(.*?warranties)$",
        r"(?i)^(.*?confidentiality)$",
        r"(?i)^(.*?governing law)$",
        r"(?i)^(.*?signatures)$",
        r"(?i)^([IVX]+\.\s+.*?)$",  # Roman numeral sections
        r"(?i)^(\d+\.\s+.*?)$",     # Numbered sections
        r"(?i)^(section\s+\d+\.\s+.*?)$"  # Explicit section headers
    ]
    
    # Split text into lines
    lines = text.split('\n')
    
    # Initialize variables
    sections = {}
    current_section = "Preamble"
    current_content = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check if the line matches any section pattern
        is_section_header = False
        for pattern in section_patterns:
            match = re.match(pattern, line)
            if match:
                # Save the current section if it exists
                if current_content:
                    sections[current_section] = '\n'.join(current_content)
                
                # Start a new section
                current_section = match.group(1)
                current_content = []
                is_section_header = True
                break
        
        if not is_section_header:
            current_content.append(line)
    
    # Save the last section
    if current_content:
        sections[current_section] = '\n'.join(current_content)
    
    return sections

# Function to extract legal entities
def extract_legal_entities(text):
    """Extract entities like parties, dates, monetary values from legal text"""
    entities = {
        'parties': [],
        'dates': [],
        'monetary_values': [],
    }
    
    # Pattern-based extraction
    # Parties (usually at the beginning of contracts)
    party_patterns = [
        r"(?i)(?:between|among)\s+(.*?)(?:and|,)\s+(.*?)(?:,|\.|;)",
        r"(?i)THIS AGREEMENT (?:made|entered into) by and between\s+(.*?)(?:and|,)\s+(.*?)(?:,|\.|;)"
    ]
    
    for pattern in party_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if isinstance(match, tuple):
                for party in match:
                    clean_party = re.sub(r"\s+", " ", party).strip()
                    if clean_party and clean_party not in entities['parties']:
                        entities['parties'].append(clean_party)
            else:
                clean_party = re.sub(r"\s+", " ", match).strip()
                if clean_party and clean_party not in entities['parties']:
                    entities['parties'].append(clean_party)
    
    # Dates
    date_patterns = [
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",  # MM/DD/YYYY or DD/MM/YYYY
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,\s+\d{4}\b"  # Month DD, YYYY
    ]
    
    for pattern in date_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            clean_date = match.strip()
            if clean_date not in entities['dates']:
                entities['dates'].append(clean_date)
    
    # Monetary values
    money_patterns = [
        r"\$\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?",  # $X,XXX.XX
        r"\b\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s+(?:dollars|USD)\b"  # X,XXX.XX dollars/USD
    ]
    
    for pattern in money_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            clean_money = match.strip()
            if clean_money not in entities['monetary_values']:
                entities['monetary_values'].append(clean_money)
    
    return entities

# Function to extract obligations and rights
def extract_obligations_and_rights(text):
    """Extract obligations and rights from legal text"""
    extractions = {
        'obligations': [],
        'rights': []
    }
    
    # Split text into sentences
    sentences = sent_tokenize(text)
    
    # Obligation patterns (usually contain modal verbs like shall, must, will)
    obligation_patterns = [
        r"(?i)(?:shall|must|will|is required to|is obligated to|agrees to)\s+([^\.;:]+)",
        r"(?i)(?:is|are) (?:obliged|obligated|required)\s+to\s+([^\.;:]+)"
    ]
    
    # Rights patterns (usually contain may, is entitled to, has the right to)
    rights_patterns = [
        r"(?i)(?:may|can|is entitled to|has the right to)\s+([^\.;:]+)",
        r"(?i)(?:is|are) (?:allowed|permitted)\s+to\s+([^\.;:]+)"
    ]
    
    # Extract obligations
    for sentence in sentences:
        for pattern in obligation_patterns:
            matches = re.findall(pattern, sentence)
            for match in matches:
                clean_obligation = re.sub(r"\s+", " ", match).strip()
                if clean_obligation and clean_obligation not in extractions['obligations']:
                    extractions['obligations'].append(clean_obligation)
    
    # Extract rights
    for sentence in sentences:
        for pattern in rights_patterns:
            matches = re.findall(pattern, sentence)
            for match in matches:
                clean_right = re.sub(r"\s+", " ", match).strip()
                if clean_right and clean_right not in extractions['rights']:
                    extractions['rights'].append(clean_right)
    
    return extractions

# Function to identify potential risks in legal text
def assess_legal_risks(text):
    """Identify potential risks, ambiguities, and inconsistencies in legal text"""
    risks = {
        'vague_terms': [],
        'missing_information': [],
        'unfavorable_clauses': []
    }
    
    # Split text into sentences
    sentences = sent_tokenize(text)
    
    # Pattern-based risk identification
    
    # Vague terms patterns
    vague_patterns = [
        r"(?i)\b(?:reasonable|promptly|timely|adequate|satisfactory|appropriate|customary|industry standard|good faith|best efforts|commercially reasonable efforts)\b",
        r"(?i)\b(?:to be determined|to be agreed|to be defined|to be specified|as necessary)\b"
    ]
    
    # Missing information patterns
    missing_patterns = [
        r"(?i)\b(?:TBD|TBA|to be announced|to be determined|to be provided|to be supplied|to be confirmed)\b",
        r"(?i)\b(?:left blank|intentionally omitted)\b"
    ]
    
    # Unfavorable clause indicators
    unfavorable_patterns = [
        r"(?i)\b(?:indemnify|hold harmless|waive|disclaim|release|no liability|not be liable|sole discretion)\b",
        r"(?i)\b(?:perpetual|irrevocable|unlimited|unrestricted|non-cancelable|non-terminable)\b"
    ]
    
    # Check each sentence for risks
    for sentence in sentences:
        # Check for vague terms
        for pattern in vague_patterns:
            if re.search(pattern, sentence):
                risks['vague_terms'].append(sentence)
                break  # Only add the sentence once
        
        # Check for missing information
        for pattern in missing_patterns:
            if re.search(pattern, sentence):
                risks['missing_information'].append(sentence)
                break
        
        # Check for unfavorable clauses
        for pattern in unfavorable_patterns:
            if re.search(pattern, sentence):
                risks['unfavorable_clauses'].append(sentence)
                break
    
    return risks

# Function for extractive summarization
def extractive_summarization(text, ratio=0.3):
    """Generate extractive summary of legal document"""
    # Split text into sentences
    sentences = sent_tokenize(text)
    
    # Return full text if it's too short
    if len(sentences) <= 3:
        return text
    
    # Create TF-IDF matrix
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentences)
    
    # Calculate sentence similarity scores
    sentence_similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Calculate sentence scores
    sentence_scores = np.sum(sentence_similarity, axis=1)
    
    # Determine number of sentences for summary
    n_sentences = max(int(len(sentences) * ratio), 3)  # At least 3 sentences
    
    # Get top sentences
    top_indices = sentence_scores.argsort()[-n_sentences:]
    top_indices = sorted(top_indices)
    
    # Create summary
    summary = ' '.join([sentences[i] for i in top_indices])
    
    return summary

# Function to compare two legal documents and identify differences
def compare_legal_documents(doc1, doc2):
    """Compare two legal documents and identify differences"""
    # Split documents into sections
    sections1 = segment_legal_document(doc1)
    sections2 = segment_legal_document(doc2)
    
    comparison_results = {
        'matching_sections': [],
        'unique_to_doc1': [],
        'unique_to_doc2': [],
        'modified_sections': []
    }
    
    # Find sections unique to each document
    for section in sections1:
        if section not in sections2:
            comparison_results['unique_to_doc1'].append(section)
    
    for section in sections2:
        if section not in sections1:
            comparison_results['unique_to_doc2'].append(section)
    
    # Find matching and modified sections
    for section in sections1:
        if section in sections2:
            # Compare section content
            text1 = sections1[section]
            text2 = sections2[section]
            
            if text1 == text2:
                comparison_results['matching_sections'].append(section)
            else:
                # Generate diff
                diff = difflib.ndiff(text1.splitlines(), text2.splitlines())
                comparison_results['modified_sections'].append({
                    'section': section,
                    'diff': list(diff)
                })
    
    return comparison_results

# Main Streamlit app
def main():
    # App title and description
    st.title("⚖️ Legal Document Analysis and Contract Intelligence")
    st.markdown("""
    This app analyzes legal documents and contracts to extract key information, 
    identify potential risks, generate summaries, and compare documents.
    """)
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select a page:", [
        "Single Document Analysis", 
        "Document Comparison", 
        "About"
    ])
    
    # Single Document Analysis page
    if page == "Single Document Analysis":
        st.header("Document Analysis")
        st.markdown("Upload a legal document (PDF) to analyze its content.")
        
        uploaded_file = st.file_uploader("Choose a file", type=["pdf"])
        
        if uploaded_file is not None:
            # Display document processing
            with st.spinner("Processing document..."):
                # Extract text from PDF
                document_text = convert_pdf_to_text(uploaded_file)
                
                if document_text:
                    st.success("Document processed successfully!")
                    
                    # Create tabs for different analysis results
                    tab1, tab2, tab3, tab4, tab5 = st.tabs([
                        "Document Overview", 
                        "Entity Extraction", 
                        "Obligations & Rights", 
                        "Risk Assessment",
                        "Summary"
                    ])
                    
                    with tab1:
                        st.subheader("Document Overview")
                        
                        # Display document statistics
                        st.markdown(f"**Document length:** {len(document_text.split())} words")
                        
                        # Display document sections
                        sections = segment_legal_document(document_text)
                        st.markdown(f"**Document sections:** {len(sections)}")
                        
                        for section, content in sections.items():
                            with st.expander(f"Section: {section}"):
                                st.text_area("Content", value=content, height=150, disabled=True)
                    
                    with tab2:
                        st.subheader("Entity Extraction")
                        
                        # Extract and display entities
                        entities = extract_legal_entities(document_text)
                        
                        # Parties
                        if entities['parties']:
                            st.markdown("**Parties Identified:**")
                            for party in entities['parties']:
                                st.markdown(f"- {party}")
                        else:
                            st.markdown("No parties identified.")
                        
                        # Dates
                        if entities['dates']:
                            st.markdown("**Dates Identified:**")
                            for date in entities['dates']:
                                st.markdown(f"- {date}")
                        else:
                            st.markdown("No dates identified.")
                        
                        # Monetary values
                        if entities['monetary_values']:
                            st.markdown("**Monetary Values Identified:**")
                            for value in entities['monetary_values']:
                                st.markdown(f"- {value}")
                        else:
                            st.markdown("No monetary values identified.")
                    
                    with tab3:
                        st.subheader("Obligations & Rights")
                        
                        # Extract and display obligations and rights
                        obligations_rights = extract_obligations_and_rights(document_text)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Obligations:**")
                            if obligations_rights['obligations']:
                                for obligation in obligations_rights['obligations']:
                                    st.markdown(f"- {obligation}")
                            else:
                                st.markdown("No obligations identified.")
                        
                        with col2:
                            st.markdown("**Rights:**")
                            if obligations_rights['rights']:
                                for right in obligations_rights['rights']:
                                    st.markdown(f"- {right}")
                            else:
                                st.markdown("No rights identified.")
                    
                    with tab4:
                        st.subheader("Risk Assessment")
                        
                        # Identify and display risks
                        risks = assess_legal_risks(document_text)
                        
                        # Vague terms
                        if risks['vague_terms']:
                            st.markdown("**Vague Terms:**")
                            for term in risks['vague_terms']:
                                st.markdown(f"- {term}")
                        else:
                            st.markdown("No vague terms identified.")
                        
                        # Missing information
                        if risks['missing_information']:
                            st.markdown("**Missing Information:**")
                            for info in risks['missing_information']:
                                st.markdown(f"- {info}")
                        else:
                            st.markdown("No missing information identified.")
                        
                        # Unfavorable clauses
                        if risks['unfavorable_clauses']:
                            st.markdown("**Potentially Unfavorable Clauses:**")
                            for clause in risks['unfavorable_clauses']:
                                st.markdown(f"- {clause}")
                        else:
                            st.markdown("No potentially unfavorable clauses identified.")
                    
                    with tab5:
                        st.subheader("Document Summary")
                        
                        # Generate and display summary
                        summary = extractive_summarization(document_text)
                        st.markdown("**Extractive Summary:**")
                        st.markdown(summary)
                        
                        # Display summary statistics
                        st.markdown(f"**Original document:** {len(document_text.split())} words")
                        st.markdown(f"**Summary:** {len(summary.split())} words")
                        st.markdown(f"**Compression ratio:** {len(summary.split()) / len(document_text.split()):.2f}")
                
                else:
                    st.error("Failed to extract text from the document. Please try another file.")
    
    # Document Comparison page
    elif page == "Document Comparison":
        st.header("Document Comparison")
        st.markdown("Upload two legal documents (PDF) to compare them.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Document 1:**")
            doc1_file = st.file_uploader("Choose first document", type=["pdf"], key="doc1")
        
        with col2:
            st.markdown("**Document 2:**")
            doc2_file = st.file_uploader("Choose second document", type=["pdf"], key="doc2")
        
        if doc1_file and doc2_file:
            # Display document processing
            with st.spinner("Comparing documents..."):
                # Extract text from PDFs
                doc1_text = convert_pdf_to_text(doc1_file)
                doc2_text = convert_pdf_to_text(doc2_file)
                
                if doc1_text and doc2_text:
                    st.success("Documents compared successfully!")
                    
                    # Compare documents
                    comparison_results = compare_legal_documents(doc1_text, doc2_text)
                    
                    # Display comparison results
                    st.subheader("Comparison Results")
                    
                    # Summary
                    st.markdown("**Summary:**")
                    st.markdown(f"- Matching sections: {len(comparison_results['matching_sections'])}")
                    st.markdown(f"- Sections only in Document 1: {len(comparison_results['unique_to_doc1'])}")
                    st.markdown(f"- Sections only in Document 2: {len(comparison_results['unique_to_doc2'])}")
                    st.markdown(f"- Modified sections: {len(comparison_results['modified_sections'])}")
                    
                    # Sections unique to Document 1
                    if comparison_results['unique_to_doc1']:
                        with st.expander("Sections only in Document 1"):
                            for section in comparison_results['unique_to_doc1']:
                                st.markdown(f"- {section}")
                    
                    # Sections unique to Document 2
                    if comparison_results['unique_to_doc2']:
                        with st.expander("Sections only in Document 2"):
                            for section in comparison_results['unique_to_doc2']:
                                st.markdown(f"- {section}")
                    
                    # Modified sections
                    if comparison_results['modified_sections']:
                        st.markdown("**Modified Sections:**")
                        for mod_section in comparison_results['modified_sections']:
                            section_name = mod_section['section']
                            with st.expander(f"Changes in: {section_name}"):
                                # Display diff
                                for line in mod_section['diff']:
                                    if line.startswith('+ '):
                                        st.markdown(f"<span style='color:green'>Added: {line[2:]}</span>", unsafe_allow_html=True)
                                    elif line.startswith('- '):
                                        st.markdown(f"<span style='color:red'>Removed: {line[2:]}</span>", unsafe_allow_html=True)
                                    elif not line.startswith('? '):  # Skip diff control lines
                                        st.text(line)
                
                else:
                    st.error("Failed to extract text from one or both documents. Please try other files.")
    
    # About page
    elif page == "About":
        st.header("About")
        st.markdown("""
        ## Legal Document Analysis and Contract Intelligence
        
        This application was developed as part of the Natural Language Processing (IN321) final project
        at the College of Artificial Intelligence (El Alamein).
        
        ### Project Description
        This project aims to develop an advanced NLP system for legal document analysis and
        contract intelligence. Our system automatically extracts key information from legal
        documents, identifies potential risks, generates summaries, and compares similar contracts to
        highlight differences.
        
        ### Features
        - Document preprocessing and segmentation
        - Information extraction (entities, obligations, rights)
        - Risk assessment
        - Document summarization
        - Document comparison
        
        ### Team Members
        - Mohamed Sallam (20100406)
        - Omar Ahmed (20103728)
        
        """)

if __name__ == "__main__":
    main()