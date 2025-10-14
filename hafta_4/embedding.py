import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE

print("\n Load Senteces transform Model")

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = ["The weather is sunny and warm today.",
"I forgot my keys on the kitchen table.",
"Artificial intelligence is changing the world rapidly.",
"Can you recommend me a good Italian restaurant nearby?",
"The cat is sleeping on the sofa.",
"I need to prepare a presentation for tomorrow’s meeting.",
"She bought a new laptop because her old one stopped working.",
"Reading books before bed helps me relax.",
"He runs five kilometers every morning.",
"The train was delayed due to heavy rain.",
"I enjoy listening to jazz music in the evenings.",
"Our team won the final match by two goals.",
"Traveling to Japan has always been my dream.",
"This smartphone has an excellent camera quality.",
"Please send me the updated version of the report.",
"The movie was too long but the ending was great.",
"Learning a new language requires patience and practice.",
"I ordered a coffee with milk and sugar.",
"The doctor advised him to exercise regularly.",
"She was late to work because of the traffic jam."]
print(f"\n working with {len(sentences)}s entences")


for i, sentence in enumerate(sentences,1):
    print(f"{i}.{sentence}")

embeddings = model.encode(sentences)
print(f"\n embedding lenght : {embeddings.shape}")
print(f"\n embedding shaped vector : {embeddings.shape[1]}")

print("\n Cosine-Similarity is calculating...")
similarity_matrix = cosine_similarity(embeddings)

# Finding the most similar sentences pair:
print("\n The most similar sentces pair:")
max_similarity = 0
best_pair = (0,0)

for i in range(len(sentences)):
    for j in range(i+1, len(sentences)):
        similarity = similarity_matrix[i][j]
        if similarity > max_similarity:
            max_similarity = similarity
            best_pair = (i, j)
        print(f"sentence {i+1} - Sentence {j+1} : {similarity:.3f}")


print(f"\n🏆 the Best Pair:")
print(f"Sentence {best_pair[0]+1}: '{sentences[best_pair[0]]}'")
print(f"Sentence {best_pair[1]+1}: '{sentences[best_pair[1]]}'")
print(f"Similarity Score: {max_similarity:.3f}")

# Multi-dim to 2D with TSNE:

print("\n Dimensionality reduction with TSNE")
tsne = TSNE(n_components = 2, random_state = 42, perplexity = 5)
embeddings_2d = tsne.fit_transform(embeddings)

# Visualization

plt.figure(figsize=(12, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
           c=range(len(sentences)), cmap='tab10', s=100, alpha=0.7)

# Add sentence number as label for each point

for i, (x, y) in enumerate(embeddings_2d):
    plt.annotate(f'{i+1}', (x, y), xytext=(5, 5), 
                textcoords='offset points', fontsize=12, fontweight='bold')

plt.title('2D TSNE Visıalization for snetences Embedding', fontsize=16, fontweight='bold')
plt.xlabel('TSNE Dim 1', fontsize=12)
plt.ylabel('TSNE Dim 2', fontsize=12)
plt.grid(True, alpha=0.3)

# Renk barı ekle
cbar = plt.colorbar()
cbar.set_label('Sentence Number', fontsize=12)

plt.tight_layout()
plt.savefig('/Users/etmco/buildwith_LLM/hafta_4/images_w4/embedding_visualization.png', 
            dpi=300, bbox_inches='tight')
plt.show()

# Adım 6: Özet bilgiler
print("\n📋 SUMMARY:")
print("="*50)
print(f"• Total Sentences Count : {len(sentences)}")
print(f"• Embedding dimesion: {embeddings.shape[1]}")
print(f"• Maximum similar Pair : Sentence {best_pair[0]+1} and {best_pair[1]+1}")
print(f"• Maximu similarity: {max_similarity:.3f}")
print(f"• Visualization saved : embedding_visualization.png")

print("\n🎯 About Embeddings':")
print("• Embeddings are numerical vector representations of texts.")
print("• Semantically similar texts have similar embedding vectors.")
print("• Cosine similarity, scales similarity in between two vectors.")
print("• TSNE, is used for visualize the multi-dimension data as 2Ddatayüksek boyutlu veriyi 2D'de görselleştirmeye yarar")