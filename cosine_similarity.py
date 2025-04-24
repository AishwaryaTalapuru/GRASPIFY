import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

def cosine_similarity(vec1, vec2):
    """
    Calculate the cosine similarity between two vectors.

    Parameters:
        vec1 (list or np.ndarray): First vector.
        vec2 (list or np.ndarray): Second vector.

    Returns:
        float: Cosine similarity between vec1 and vec2.
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0  # Avoid division by zero
    return dot_product / (norm_vec1 * norm_vec2)

# Example usage:
a = [1, 2, 3]
b = [4, 5, 6]
#print(cosine_similarity(a, b))  # Output: 0.9746318461970762



# Initialize model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()  # Set model to evaluation mode

def get_bert_embeddings(sentences):
    # Tokenize and prepare input
    encoded_input = tokenizer.batch_encode_plus(
        sentences,
        padding=True,
        truncation=True,
        return_tensors='pt',
        add_special_tokens=True
    )
    
    # Forward pass through BERT
    with torch.no_grad():
        outputs = model(**encoded_input)
    
    # Extract last hidden states and compute mean pooling
    last_hidden_states = outputs.last_hidden_state
    attention_mask = encoded_input['attention_mask']
    
    # Expand attention mask for dimension matching
    expanded_mask = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
    
    # Sum hidden states while ignoring padding tokens, then divide by mask sum
    sum_embeddings = torch.sum(last_hidden_states * expanded_mask, 1)
    sum_mask = torch.clamp(expanded_mask.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

# Example usage
sentence1 = "The quick brown fox jumps over the lazy dog"
sentence2 = "A fast auburn canine leaps above a sleeping hound"

embeddings = get_bert_embeddings([sentence1, sentence2])
print(embeddings)
#similarity = cosine_similarity(embeddings[0].numpy().reshape(1,-1), 
                              #embeddings[1].numpy().reshape(1,-1))

#print(f"Cosine similarity: {similarity[0][0]:.4f}")

