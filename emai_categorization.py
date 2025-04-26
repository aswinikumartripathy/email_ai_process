import os
from dotenv import load_dotenv

from llama_index.core import Document
from llama_index.core import DocumentSummaryIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import PromptHelper
from llama_index.core import ServiceContext
from llama_index.core import StorageContext
from llama_index.core import Settings
from llama_index.core import GPTVectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core.prompts import PromptTemplate

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import bert_score

# 1. Load environment variables
def load_api_key():
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = "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"


# 2. Set global LlamaIndex settings
def configure_llama_index():
    Settings.llm = OpenAI(model= "gpt-4", temperature=0.0)
    Settings.embed_model = OpenAIEmbedding()
    Settings.chunk_size_limit = 128
    Settings.chunk_overlap = 50
    Settings.num_output = 2048
    prompt_helper = PromptHelper(
        context_window = 4096,
        num_output = 3000,
        chunk_overlap_ratio = 0.1,
        chunk_size_limit = 512,
    )
    Settings.prompt_helper = prompt_helper

# 3. Create index using GPTVectorStoreIndex (which supports chunk retrieval)
def build_vector_index(email_text: str) -> GPTVectorStoreIndex:
    document  = Document(text=email_text)
    index = GPTVectorStoreIndex.from_documents([document])
    return index

# categorize emails into different categories
def categorize_email(email_text: str) -> str:
    # Build the vector index (make sure this function is defined elsewhere)
    index = build_vector_index(email_text)

    categories = ["payment", "billing", "enrollment", "uncategorized"]
    category_query = f"""
    Based on the relevant parts of this email, categorize it into one of the following categories:
    {', '.join(categories)}. If it does not fit any of these categories, please categorize it as 'uncategorized'.
    Respond only with the category name.
    """

    # Retrieve relevant chunks from the index
    retriever = VectorIndexRetriever(index=index, similarity_top_k=3)
    nodes = retriever.retrieve(category_query)

    # Get total number of chunks
    total_chunks = len(index.docstore.docs)
    print(f"Total chunks in the index: {total_chunks}")
    print(f"Top {len(nodes)} relevant chunks retrieved:\n")

    for i, node in enumerate(nodes, 1):
        print(f"Chunk {i}:\n{'-'*40}\n{node.text.strip()}\n")

    # Combine retrieved chunks
    combined_text = "\n".join([node.text for node in nodes])

    # prompts for LLM
    full_prompt = f"""
    {combined_text}

    Categorize this email based on the above content into one of: {', '.join(categories)}.

    ---
    1. Enrollment  
    Categorize the email as Enrollment if any of the below keywords are found:  
    a) AWD  
    b) Enrollment  
    c) Autopay  

    2. Payment  
    Categorize the email as Payment if any of the below keywords are found:  
    a) Payment  
    b) Cheque  
    c) Remittance  
    d) Invoice  
    e) Coupon  
    f) Credit  
    g) Refund  

    3. Billing  
    Categorize the email as Billing if any of the below keywords are found:  
    a) Billing  
    b) Premium  
    c) Invoice  
    d) Incorrect Invoice  

    Post-Processing Steps:

    1. Autonomic Analysis Protocol  
    - Automatically identify patterns and context beyond just keywords (e.g., "I was charged wrongly" implies billing).  
    - Detect sentence structure and tone to infer category when explicit keywords are missing.

    2. Primary Intent Detection  
    - If multiple categories are detected, determine which intent is dominant based on keyword frequency, placement, and context.  
    - Prioritize the category mentioned in the subject or first few lines.

    3. Contradicting Evidence Check  
    - Look for conflicting phrases (e.g., “Refund not received” implies Payment, not Billing).  
    - Remove false positives caused by ambiguous keyword overlap (e.g., “Autopay invoice” likely relates to Enrollment, not Billing).

    4. Priority Rules  
    - If both Enrollment and Payment are detected, prioritize Enrollment.  
    - If both Billing and Payment are detected, prioritize Payment.  
    - If all three are mentioned, prioritize based on order: Enrollment > Payment > Billing.

    5. Confidence Assessment  
    - Assign a confidence score (0–100%) based on keyword density and clarity.  
    - If confidence is below 60%, flag the result for manual review.

    6. Integration of Atomic Signal  
    - Include other metadata if available (e.g., subject line, tags, sender type) to refine prediction.  
    - Example: If the sender is a known billing department, weight Billing higher.

    7. Output Validation  
    - Ensure the final category logically matches the context.  
    - If mismatch found, re-apply rules from step 1 to 6.  
    - Log output decision along with justification for traceability.

    If none of the keywords match, categorize the email as 'uncategorized'.
    """

    response = Settings.llm(full_prompt)
    return response.strip().lower()

def get_existing_chunks(index: GPTVectorStoreIndex):
    # return list of node object already used in the index
    return list(index.docstore.docs.values())

def recursive_summerize(index: GPTVectorStoreIndex, summary_window_words: int = 100) -> str:
    nodes = get_existing_chunks(index)

    summary_template = PromptTemplate(
        f"Summerize the following text into approximately {summary_window_words} words:\n\n"
        "{{context_str}}\n\nSummary:"
    )

    tree_summerizer = TreeSummarize(summary_template=summary_template)
    summary_index = DocumentSummaryIndex(nodes)

    query_engine = summary_index.as_query_engine(response_synthesizer=tree_summerizer)
    response = query_engine.query("Please summerize the entire content of this email.")

    return response.response.strip()

def cosine_similarity_score(original_text, generated_summary):
    """
    Evaluate summary using cosine similarity between the original text and generated summary.
    
    Parameters:
        original_text (str): The original text.
        generated_summary (str): The generated summary.
    
    Returns:
        float: Cosine similarity score.
    """
    # Initialize the vectorizer
    vectorizer = TfidfVectorizer().fit_transform([original_text, generated_summary])
    # Compute cosine similarity between the original and generated text
    cosine_sim = cosine_similarity(vectorizer[0:1], vectorizer[1:2])
    
    return cosine_sim[0][0]

def bert_score_evaluation(original_text, generated_summary):
    """
    Evaluate the generated summary using BERTScore, which compares embeddings.
    
    Parameters:
        original_text (str): The original text.
        generated_summary (str): The generated summary.
    
    Returns:
        dict: Precision, Recall, and F1 score from BERTScore.
    """
    # Compute BERTScore
    P, R, F1 = bert_score.score([generated_summary], [original_text], lang="en")
    
    return {"precision": P.item(), "recall": R.item(), "f1": F1.item()}

def load_email_from_file(file_path: str) -> str:
    with open(file_path, 'r', encoding="utf-8") as file:
        email_text = file.read()
    return email_text


if __name__ == "__main__":
    load_api_key()
    configure_llama_index()

    # Load email text from a file
    email_text = load_email_from_file("emails/email4.txt")
    index = build_vector_index(email_text)

    category = categorize_email(email_text)
    print(f"Predicted Category: {category}")

    summary = recursive_summerize(index, summary_window_words=50)
    print(f"\nEmail Summary: {summary}")


similarity_score = cosine_similarity_score(email_text, summary)
print(f"Cosine Similarity: {similarity_score}")
bert_scores = bert_score_evaluation(email_text, summary)
print(f"BERTScore - Precision: {bert_scores['precision']}, Recall: {bert_scores['recall']}, F1: {bert_scores['f1']}")
