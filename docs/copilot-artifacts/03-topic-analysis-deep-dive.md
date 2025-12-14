# Topic Analysis Deep Dive: Semantic Clustering Explained

## Understanding AI-Powered Theme Extraction and Topic Discovery

---

## What Is Topic Analysis?

**Definition:**
Topic analysis is the computational task of automatically identifying, extracting, and organizing the main themes discussed in a text document or collection of documents.

**The Goal:**
Transform unstructured text into structured insights:

- **What** are the main topics?
- **How** are topics related to each other?
- **Where** in the text do topics appear?
- **Why** are these topics important?

**Beyond Word Frequency:**
Your system doesn't just count words—it uses AI to:

- Group related concepts semantically ("economy" + "jobs" → "Economic Policy")
- Generate meaningful topic labels using LLMs
- Show contextual examples of topics in use
- Provide interpretive summaries of main themes

---

## The Problem with Simple Word Frequency

### Traditional Approach: Word Counting

**Method:**

```python
# Count words
words = text.split()
freq = Counter(words)
top_10 = freq.most_common(10)

# Output: ["the", "and", "to", "of", "we", "our", "that", "is", "in", "this"]
```

**Problems:**

1. **Stopwords dominate** ("the", "and", "to")
2. **No semantic grouping** ("economy" and "jobs" counted separately)
3. **No context** (can't tell *how* word is used)
4. **No meaning** (just a list of words, not themes)

### Your Solution: Semantic Clustering

**Method:**

1. Extract important keywords (filtered, no stopwords)
2. Convert keywords to semantic embeddings
3. Cluster similar keywords together using AI
4. Generate meaningful labels for each cluster
5. Show contextual examples with highlighting
6. Provide AI-generated summary

**Result:**

```text
Cluster 1: "Economic Policy"
- Keywords: economy (40), jobs (35), market (28), growth (22)
- Snippet: "The **economy** is booming with **job** creation..."

Cluster 2: "Border Security"  
- Keywords: immigration (52), border (45), wall (38), security (30)
- Snippet: "We're securing our **border** with a **wall**..."
```

---

## How Semantic Clustering Works

### Step 1: Keyword Extraction

**Goal:** Identify important words that represent topics

**Process:**

```python

def extract_keywords(text: str, top_n: int = 20) -> List[Dict]:
    # 1. Clean text
    cleaned = clean_text(text, remove_stopwords=True)
    
    # 2. Tokenize
    tokens = tokenize_text(cleaned)
    
    # 3. Filter
    tokens = [
        t for t in tokens 
        if t.isalpha()  # Only alphabetic
        and len(t) > 2  # At least 3 characters
        and t.lower() not in excluded_verbs  # Not common verbs
    ]
    
    # 4. Count frequencies
    freq = Counter(tokens)
    top_words = freq.most_common(top_n)
    
    # 5. Calculate relevance scores
    max_count = top_words[0][1]
    keywords = []
    for word, count in top_words:
        keywords.append({
            "word": word,
            "count": count,
            "relevance": count / max_count  # Normalize 0-1
        })
    
    return keywords
```

**Example:**

```text
Input: Long political speech

Output:
[
  {"word": "economy", "count": 40, "relevance": 1.0},
  {"word": "jobs", "count": 35, "relevance": 0.875},
  {"word": "border", "count": 30, "relevance": 0.75},
  {"word": "immigration", "count": 28, "relevance": 0.70},
  ...
]
```

**Excluded Verbs:**
Common verbs like "said", "going", "know", "think" are filtered out because they don't represent topics.

### Step 2: Generate Embeddings

**Goal:** Convert each keyword into a 768-dimensional vector

**Why Embeddings?**
Words with similar meanings should cluster together in vector space.

**Model:** MPNet (`all-mpnet-base-v2`)

**Process:**

```python

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-mpnet-base-v2")

keywords = ["economy", "jobs", "market", "growth", "border", "immigration"]
embeddings = model.encode(keywords)

print(embeddings.shape)  # (6, 768)
```

**Visualization (Simplified to 2D):**

```text
        economy •
           jobs •  ← Close together (economic cluster)
         market •

                     border •
                immigration •  ← Close together (immigration cluster)
```

### Step 3: KMeans Clustering

**Goal:** Group keywords into semantic clusters

**Algorithm:** KMeans (unsupervised learning)

**How KMeans Works:**

1. **Initialize:** Randomly place K cluster centers in 768-dimensional space
2. **Assign:** Assign each keyword to nearest cluster center
3. **Update:** Move cluster centers to mean of assigned keywords
4. **Repeat:** Until cluster assignments stabilize

**Your Implementation:**

```python
from sklearn.cluster import KMeans

# Auto-determine number of clusters
num_keywords = len(keywords)
if num_keywords < 10:
    num_clusters = 3
elif num_keywords < 20:
    num_clusters = 4
elif num_keywords < 30:
    num_clusters = 5
else:
    num_clusters = 6

# Cluster
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(embeddings)

# Group keywords by cluster
clusters = defaultdict(list)
for keyword, label in zip(keywords, cluster_labels):
    clusters[label].append(keyword)
```

**Example Result:**

```text
Cluster 0: ["economy", "jobs", "market", "growth", "prosperity"]
Cluster 1: ["border", "immigration", "wall", "security", "illegal"]
Cluster 2: ["media", "fake", "news", "press", "cnn"]
```

### Step 4: Generate Cluster Labels

**Goal:** Create meaningful names for each cluster

**Problem:** Cluster 0 isn't a good label—need something like "Economic Policy"

**Solution:** Use Gemini LLM to generate labels

**Prompt:**

```text
You are a topic labeling expert. Given a cluster of related keywords from a 
political speech, generate a concise, descriptive label (2-4 words) that 
captures the main theme.

KEYWORDS: economy, jobs, market, growth, prosperity

Generate a label that represents this topic cluster.
```

**Gemini Response:**

```text
"Economic Policy"
```

**Fallback:** If LLM unavailable, use most frequent keyword as label:

```python

label = max(cluster_keywords, key=lambda k: k["count"])["word"].title()
```

### Step 5: Extract Contextual Snippets

**Goal:** Show keywords in actual context with highlighting

**Process:**

```python

def extract_snippets(
    text: str,
    cluster_keywords: List[str],
    snippets_per_topic: int = 3
) -> List[str]:
    # 1. Find all positions where keywords appear
    positions = []
    text_lower = text.lower()
    for keyword in cluster_keywords:
        for match in re.finditer(r'\b' + re.escape(keyword) + r'\b', text_lower):
            positions.append(match.start())
    
    # 2. Deduplicate (min 200 chars apart)
    positions.sort()
    unique_positions = [positions[0]]
    for pos in positions[1:]:
        if pos - unique_positions[-1] >= 200:
            unique_positions.append(pos)
    
    # 3. Extract context around each position (±100 characters)
    snippets = []
    for pos in unique_positions[:snippets_per_topic]:
        start = max(0, pos - 100)
        end = min(len(text), pos + 100)
        snippet = text[start:end]
        
        # 4. Highlight keywords with **bold** markdown
        for keyword in cluster_keywords:
            snippet = re.sub(
                r'\b(' + re.escape(keyword) + r')\b',
                r'**\1**',
                snippet,
                flags=re.IGNORECASE
            )
        
        snippets.append(snippet.strip())
    
    return snippets
```

**Example:**

```text
Cluster: "Economic Policy" (keywords: economy, jobs, market)

Snippets:
1. "The **economy** is booming like never before. We've created millions of **jobs**..."
2. "Manufacturing **jobs** are returning to America. The stock **market** is at record highs..."
3. "Our economic policies have unleashed the **market** and created the best **economy**..."
```

### Step 6: Generate AI Summary

**Goal:** Provide interpretive analysis of main themes

**Gemini Prompt:**

```text
You are analyzing the main topics in a political speech. Here are the topic 
clusters identified:

TOPIC CLUSTERS:
1. Economic Policy (75 mentions): economy, jobs, market, growth
2. Border Security (60 mentions): immigration, border, wall, security
3. Media Criticism (45 mentions): media, fake, news, press

TEXT SAMPLE:
[First 2000 characters of text]

Write a 2-3 sentence analytical summary identifying the dominant topics and 
their significance in the speech. Be objective and concise.
```

**Gemini Response:**

```text
"The speech emphasizes economic achievements as a central theme, with frequent 
references to job creation and market performance. Border security emerges as 
a secondary but substantial focus, highlighting immigration enforcement. Media 
criticism serves as a recurring rhetorical device throughout the discourse."
```

---

## Technical Deep Dive: The Math Behind Clustering

### Embeddings and Vector Space

**What Is Vector Space?**
A mathematical space where each dimension represents a learned feature.

**Example (Simplified to 3D):**

```text
"economy" = [0.8, 0.2, -0.1]
           ↑    ↑     ↑
         dim1 dim2  dim3

"jobs"    = [0.75, 0.25, -0.05]  ← Close to "economy"
"border"  = [-0.3, 0.9, 0.6]     ← Far from "economy"
```

**Distance Calculation:**

```text
distance(economy, jobs) = sqrt((0.8-0.75)² + (0.2-0.25)² + (-0.1-(-0.05))²)
                        = sqrt(0.0025 + 0.0025 + 0.0025)
                        = 0.0866  (very close)

distance(economy, border) = sqrt((0.8-(-0.3))² + (0.2-0.9)² + (-0.1-0.6)²)
                          = sqrt(1.21 + 0.49 + 0.49)
                          = 1.48  (far apart)
```

**In 768 Dimensions:**
Same principle, just with 768 numbers instead of 3.

### KMeans Convergence

**Iteration 1:**

```text
Initial centers (random):
C1 = [0.5, 0.5, ...]
C2 = [-0.5, 0.8, ...]
C3 = [0.2, -0.6, ...]

Assign keywords to nearest center:
"economy" → C1
"jobs" → C1
"market" → C1
"border" → C2
...
```

**Iteration 2:**

```text
Update centers to mean of assigned keywords:
C1 = mean([economy, jobs, market]) = [0.76, 0.23, ...]
C2 = mean([border, immigration, wall]) = [-0.25, 0.85, ...]

Re-assign based on new centers:
(some keywords may switch clusters)
```

**Iteration N:**

```text
Centers stabilize (no keywords switching):
Convergence achieved!
```

**Why KMeans?**

- **Fast** — Converges in few iterations
- **Deterministic** — With fixed random seed, always same result
- **Scalable** — Works well with high-dimensional data
- **Interpretable** — Clear cluster assignments

---

## Configuration and Tuning

### Auto-Cluster Determination

**Your Logic:**

```python
num_keywords = len(keywords)

if num_keywords < 10:
    num_clusters = 3
elif num_keywords < 20:
    num_clusters = 4
elif num_keywords < 30:
    num_clusters = 5
else:
    num_clusters = 6
```

**Reasoning:**

- **Too few clusters** → Overly broad topics (everything is "Politics")
- **Too many clusters** → Overly specific topics ("Jobs in Ohio", "Jobs in Michigan")
- **Sweet spot** → 3-6 clusters for typical speeches

**Can Override:**

```python
# API call with custom cluster count
POST /analyze/topics?num_clusters=5
```

### Relevance Threshold Filtering

**Purpose:** Exclude weak clusters with low relevance

**Configuration (YAML):**

```yaml

topic:
  topic_relevance_threshold: 0.3
  topic_min_clusters: 3
```

**Logic:**

```python
# Filter clusters by relevance
filtered = [c for c in clusters if c["avg_relevance"] >= 0.3]

# Keep minimum 3 clusters even if below threshold
if len(filtered) < 3 and len(clusters) >= 3:
    filtered = clusters[:3]
```

**Why Filter?**
Some clusters have low-frequency keywords that aren't truly important topics.

### Excluded Verbs

**Configuration:**

```python

excluded_verbs = [
    "said", "going", "know", "think", "get", "got", "want",
    "make", "take", "come", "see", "look", "tell", "give"
]
```

**Why Exclude?**
These verbs appear frequently but don't represent topics.

**Example:**

```text
Without filtering: "said" appears 100 times
With filtering: "economy" (40), "jobs" (35), "border" (30) become top topics
```

---

## Real-World Applications

### 1. Document Summarization

**Use Case:** Quickly understand what a long speech is about

**Workflow:**

```python

result = analyze_topics(speech_text, top_n=5)

print("Main Topics:")
for cluster in result["clustered_topics"]:
    print(f"- {cluster['label']} ({cluster['total_mentions']} mentions)")

print(f"\nSummary: {result['summary']}")
```

**Output:**

```text
Main Topics:
- Economic Policy (120 mentions)
- Border Security (85 mentions)
- Media Criticism (60 mentions)

Summary: The speech centers on economic achievements and border security,
with recurring criticism of media coverage.
```

### 2. Comparative Analysis

**Use Case:** Compare topics across multiple speeches

**Workflow:**

```python

speeches = load_all_speeches()
topic_evolution = []

for speech in speeches:
    result = analyze_topics(speech["text"])
    topic_evolution.append({
        "date": speech["date"],
        "topics": [c["label"] for c in result["clustered_topics"][:3]]
    })

# Plot topic frequency over time
plot_topic_timeline(topic_evolution)
```

### 3. Content Classification

**Use Case:** Categorize documents by topic

**Workflow:**

```python

def categorize_document(text):
    result = analyze_topics(text, top_n=1)
    primary_topic = result["clustered_topics"][0]["label"]
    return primary_topic

# Categorize corpus
for doc in documents:
    category = categorize_document(doc)
    categories[category].append(doc)
```

### 4. Research Discovery

**Use Case:** Find speeches discussing specific topics

**Workflow:**

```python

def find_speeches_about(topic_keyword):
    matching_speeches = []
    
    for speech in all_speeches:
        result = analyze_topics(speech["text"])
        
        # Check if any cluster contains keyword
        for cluster in result["clustered_topics"]:
            if any(topic_keyword in kw["word"].lower() 
                   for kw in cluster["keywords"]):
                matching_speeches.append(speech)
                break
    
    return matching_speeches

# Find all speeches mentioning "immigration"
immigration_speeches = find_speeches_about("immigration")
```

---

## Performance Considerations

### Latency Breakdown

**Typical Request (1000 words):**

```text
Keyword extraction: 100-200ms
Embedding generation: 200-500ms (depends on keyword count)
KMeans clustering: 50-100ms
Snippet extraction: 200-400ms
LLM label generation: 500-1000ms (per cluster)
LLM summary: 1000-1500ms
Total: ~3-5 seconds
```

**Long Text (5000 words):**

```text
Keyword extraction: 300-500ms
Embedding generation: 500-1000ms (more keywords)
KMeans clustering: 100-200ms
Snippet extraction: 500-800ms
LLM calls: 2000-3000ms
Total: ~5-8 seconds
```

### Optimization Opportunities

**Batch Embedding Generation:**

```python

# Instead of one at a time
embeddings = [model.encode(kw) for kw in keywords]  # Slow

# Batch process
embeddings = model.encode(keywords)  # Faster
```

**Cache Topic Results:**

```python

@lru_cache(maxsize=100)
def analyze_topics_cached(text_hash):
    return analyze_topics(text_hash)
```

**Parallel LLM Calls:**

```python

async def generate_labels(clusters):
    tasks = [generate_label_async(cluster) for cluster in clusters]
    labels = await asyncio.gather(*tasks)
    return labels
```

### Memory Usage

**Models:**

- Sentence-transformers (MPNet): ~500 MB
- Scikit-learn (KMeans): Minimal (~10 MB)

**Runtime:**

- Embeddings: ~50-100 MB per request (temporary)
- Cluster assignments: <1 MB

**Total:** ~1.5 GB RAM recommended

---

## Testing Strategy

### Unit Tests

**Keyword Extraction:**

```python

def test_keyword_extraction():
    text = "economy jobs economy market jobs economy"
    keywords = extract_keywords(text, top_n=3)
    
    assert keywords[0]["word"] == "economy"
    assert keywords[0]["count"] == 3
    assert keywords[0]["relevance"] == 1.0
```

**Clustering:**

```python

def test_clustering():
    keywords = [
        {"word": "economy", "count": 10},
        {"word": "jobs", "count": 8},
        {"word": "border", "count": 7},
    ]
    clusters = cluster_keywords(keywords, num_clusters=2)
    
    assert len(clusters) == 2
    # economy and jobs should cluster together
```

**Snippet Extraction:**

```python

def test_snippet_extraction():
    text = "The economy is strong. Later, the border needs security."
    keywords = ["economy", "border"]
    snippets = extract_snippets(text, keywords, snippets_per_topic=2)
    
    assert len(snippets) == 2
    assert "**economy**" in snippets[0]
    assert "**border**" in snippets[1]
```

### Integration Tests

```python

def test_full_topic_analysis():
    text = load_sample_speech()
    result = analyze_topics(text, top_n=5)
    
    assert "clustered_topics" in result
    assert "snippets" in result
    assert "summary" in result
    assert "metadata" in result
    
    assert len(result["clustered_topics"]) <= 5
    assert all("label" in c for c in result["clustered_topics"])
```

---

## Common Issues & Solutions

### Issue: No Clusters Generated

**Symptoms:** Empty `clustered_topics` array

**Diagnosis:**

- Text too short (<50 words)
- All words are stopwords
- Filtered too aggressively

**Solutions:**

```python

# Check keyword count
keywords = extract_keywords(text)
print(f"Keywords found: {len(keywords)}")

# Lower filtering threshold
# Remove fewer excluded_verbs
# Increase top_n parameter
```

### Issue: Generic Cluster Labels

**Symptoms:** Labels like "Great" or "Country" instead of "Economic Policy"

**Diagnosis:**

- LLM not available
- Fallback to most frequent keyword
- Keyword isn't descriptive

**Solutions:**

```python

# Check LLM configuration
print(f"LLM available: {llm_service is not None}")

# Set LLM_API_KEY in .env
# Use better keywords (longer text)
```

### Issue: Too Many/Too Few Clusters

**Symptoms:** Results feel over-/under-granular

**Diagnosis:**

- Auto-determined cluster count doesn't fit use case

**Solutions:**

```python

# Manually specify cluster count
POST /analyze/topics?num_clusters=4

# Tune auto-determination logic in config
```

---

## Advanced Concepts

### Alternative Clustering Algorithms

**DBSCAN (Density-Based):**

- Doesn't require specifying K
- Can find arbitrarily shaped clusters
- More expensive computationally

**Hierarchical Clustering:**

- Creates dendrogram of cluster relationships
- Good for understanding topic hierarchies
- Slower than KMeans

**Why You Use KMeans:**

- Fast and deterministic
- Works well with fixed K
- Good enough for most use cases
- Easy to tune and understand

### Topic Modeling Alternatives

**LDA (Latent Dirichlet Allocation):**

- Classical topic modeling algorithm
- Assumes documents are mixtures of topics
- Probabilistic approach

**BERTopic:**

- Modern transformer-based topic modeling
- Uses embeddings + clustering (similar to your approach)
- More complex, better for large corpora

**Why Your Approach:**

- Simpler and more transparent
- Semantic clustering with embeddings
- LLM-generated labels for interpretability
- Good balance of simplicity and quality

---

## Next Steps

**Continue Learning:**

- **`04-technical-architecture.md`** — Overall system design and patterns
- **`05-llm-integration.md`** — Deep dive on LLM usage
- **`06-concepts-glossary.md`** — Quick reference for all terms

**Practice Explaining:**

- What's the difference between word frequency and semantic clustering?
- Why use KMeans instead of DBSCAN?
- How do embeddings enable semantic grouping?
- What value does the LLM add?

**Interview Questions:**

- Why not just use LDA or BERTopic?
- How would you handle multilingual text?
- What's the computational complexity of your approach?
- How would you evaluate clustering quality?

---

You now understand semantic topic analysis at a deep level! 🔍
