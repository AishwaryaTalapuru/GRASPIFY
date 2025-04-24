import json
import numpy as np
from collections import Counter
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from rerank import SemanticAxisReranker

class AnalyseSimilarity:
    def __init__(self, json_path="transcription/data.json"):
        self.json_path = json_path
        self.open_json_file()
    def open_json_file(self):
        #Opening data.json file
        with open(self.json_path, "r") as f:
            self.data = json.load(f)
        self.create_embeddings()
    def create_embeddings(self):
        print(f"##### Creating Embeddings for the DATA #####")
        self.embeddings = []
        self.metadata = []
        for i, chunk in enumerate(self.data):
            emb = chunk.get("embedding")
            
            #Checking if an embedding is not a list
            if not isinstance(emb, list):
                #print(f"Index {i}: Not a list → {emb}")
                continue
            #Checking if an embedding is a nested list
            if any(isinstance(x, list) for x in emb):
                #print(f"Index {i}: Nested list → {emb}")
                continue
            
            #Checking if an embedding contains non-float elements
            if not all(isinstance(x, (float, int)) for x in emb):
                #print(f"Index {i}: Contains non-float elements → {emb}")
                continue

            self.embeddings.append(emb)
            metadata_data = {}
            if chunk["type"]=="text":
                metadata_data = {
                    "index" : i, 
                    "type" : chunk["type"],
                    "subtype" : chunk["subtype"],
                    "page" : chunk["page"],
                    "line_no" : chunk["line_no"],
                    "content" : chunk["content"]
                }
            elif chunk["type"]=="audio":
                metadata_data = {
                    "index" : i, 
                    "type" : chunk["type"], 
                    "subtype" : chunk["subtype"],
                    "start" : chunk["start"],
                    "end" : chunk["end"],
                    "content" : chunk["content"]
                }
            elif chunk["type"] == "caption_of_image":
                metadata_data = {
                    "index" : i, 
                    "type" : chunk["type"],
                    "subtype" : chunk["subtype"],
                    "filename" : chunk["filename"], 
                    "content" : chunk["content"], 
                    "text_on_image" : chunk["text_on_image"]
                }
            self.metadata.append(metadata_data)
            #print(f"Index {i}: OK → len = {len(emb)}")

        #Pruning out valid embeddings
        self.valid_embeddings = []
        for emb in self.embeddings:
            if isinstance(emb, list) and all(isinstance(x, (float, int)) for x in emb):
                if not any(isinstance(x, list) for x in emb):
                    self.valid_embeddings.append(emb)
                else:
                    print(f"Skipping nested embedding: {emb}")
            else:
                print(f"Skipping malformed embedding: {emb}")
        


        self.lens = [len(e) for e in self.valid_embeddings]
        self.lens_counter = Counter(self.lens)
        self.most_common_len = self.lens_counter.most_common(1)[0][0]

        self.cleaned_embeddings = [e for e in self.valid_embeddings if len(e) == self.most_common_len]

        #print(f"Using embeddings of length {self.most_common_len} (kept {len(self.cleaned_embeddings)} items)")
        self.embeddings = np.array(self.cleaned_embeddings, dtype=np.float32)
        self.calclate_cosine_similarity()
    
    def calclate_cosine_similarity(self):
        print(f"##### Calculating COSINE SIMILARITY and CLUSTERING #####")
        # Use cosine distance (not similarity) for clustering
        self.cosine_dists = cosine_distances(self.embeddings)

        # Apply DBSCAN
        self.clustering = DBSCAN(eps=0.05, min_samples=2, metric="precomputed")  # you can tune eps
        self.labels = self.clustering.fit_predict(self.cosine_dists)
        #print(f"labels: {self.labels}")

        #print(f"Number of clusters found: {len(set(self.labels)) - (1 if -1 in self.labels else 0)}")

        self.clusters = defaultdict(list)

        for label, meta in zip(self.labels, self.metadata):
            if label == -1:
                continue  # Skip noise points (optional)
            self.clusters[label].append(meta)
        
        print(f"Number of clusters found: {len(set(self.labels)) - (1 if -1 in self.labels else 0)}")
        self.create_centroids_for_clusters()
    
    def create_centroids_for_clusters(self):
        # Compute centroids for each cluster
        self.cluster_output = []
        for cluster_id, items in self.clusters.items():
            self.emb_list = [self.embeddings[item["index"]] for item in items]
            centroid = np.mean(self.emb_list, axis=0).tolist()

            self.cluster_output.append({
                "cluster_id": f"cluster_{cluster_id}",
                "segments": items,
                "centroid": centroid
            })
        #print("cluster output: ", self.cluster_output)

        self.write_to_json_file()
    
    def write_to_json_file(self):
        # Clear the transcription/data.json file and initialize it with an empty list
        if os.path.exists("semantic_clusters.json"):
            os.remove("semantic_clusters.json")
            print("JSON file deleted.")
        
        with open("semantic_clusters.json", "w") as f:
            json.dump(self.cluster_output, f, indent=2)

        print("Clusters saved to semantic_clusters.json")


        while True:
            print("\n" + "="*50)
            question = input("Please enter your QUERY (or type 'quit' to exit):\n> ")
            
            if question.strip().lower() == "quit":
                print("Exiting... Have a great day!")
                break

            self.semantic_axis = input("\n What is the semantic AXIS you want to focus on? (e.g., explanation, definition, example):\n> ")
            print("\nProcessing your query...\n")
            self.query_processing_and_embedding(question)


    def query_processing_and_embedding(self, question):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.query = question 
        self.query_embedding = self.model.encode(self.query, convert_to_tensor=False)
        # Assume: clusters is a list of cluster dictionaries
        self.centroids = [c["centroid"] for c in self.cluster_output]
        self.similarities = cosine_similarity([self.query_embedding], self.centroids)[0]

        self.top_k = 10 # you can change this
        self.top_indices = np.argsort(self.similarities)[::-1][:self.top_k]

        self.top_clusters = [self.clusters[i] for i in self.top_indices]
        print("Top clusters : ", self.top_clusters)

        #print(self.top_clusters)

        self.segment_contents = []

        for i, idx in enumerate(self.top_indices):
            cluster = self.cluster_output[idx]  # Get the cluster dict
            #print(f"Top {i+1}: Cluster {cluster['segments']} with score {self.similarities[idx]:.4f}")
            for ind in range(0, len(cluster['segments'])):
                meta_dict = {}
                meta_dict['content'] = cluster['segments'][ind]['content']
                meta_dict['source'] = cluster['segments'][ind]['subtype']
                if meta_dict['source'] == "pdf":
                    meta_dict['page'] = cluster['segments'][ind]['page']
                    meta_dict['line_no'] = cluster['segments'][ind]['line_no']
                if meta_dict['source'] == "mp3":
                    meta_dict['start'] = cluster['segments'][ind]['start']
                    meta_dict['end'] = cluster['segments'][ind]['end']
                #self.segment_contents.append({'content': cluster['segments'][ind]['content'], 'source': cluster['segments'][ind]['subtype']})
                self.segment_contents.append(meta_dict)
        
        
        #print(self.segment_contents)
        
        #self.rerank(question, "explanation", self.segment_contents)
        
        #reranker = SemanticAxisReranker()
        #ranked = reranker.answer_generator(query=question, axis="explanation", segments = self.segment_contents)
        #print("Answer is : ", ranked)
        reranker = SemanticAxisReranker(model_name="llama3:latest")
        final_answer = reranker.generate_answer(question, self.semantic_axis, self.segment_contents)
        print("\n" + "="*60)
        print(" FINAL ANSWER")
        print("="*60)
        print(final_answer)
        print("="*60 + "\n")
        
        




obj = AnalyseSimilarity()
