## 🧠 CodeRacer



<img width="983" height="825" alt="image" src="https://github.com/user-attachments/assets/dfc5d62a-ed50-4cca-88f5-e0233031c623" />
<img width="912" height="601" alt="image" src="https://github.com/user-attachments/assets/189c5fc9-63c8-4662-bb3b-406f02f72afb" />

<img width="961" height="616" alt="image" src="https://github.com/user-attachments/assets/f2d8f208-3bca-4339-8537-a39aed2b13a1" />


## Interactive Gesture-Based Coding Assistant using RAG & LLMs

---

## 📌 Project Overview

**CodeRacer** is an AI-powered, gesture-controlled coding assistant designed to help **beginner and intermediate programmers** understand code and debug errors efficiently.

Unlike traditional IDEs that only provide syntax highlighting and basic debugging, CodeRacer enables **real-time, context-aware explanations and bug detection** using **hand gestures**, **Retrieval-Augmented Generation (RAG)**, and **Large Language Models (LLMs)**.

Users interact with the system **without using keyboard or mouse**, making learning more intuitive and interactive.

---

## 🎯 Problem Statement

Novice programmers often face difficulties in:

* Understanding what a piece of code does
* Interpreting complex logic
* Debugging errors efficiently

Current IDEs lack:

* Line-level natural language explanations
* Context-aware bug detection
* Interactive learning mechanisms

As a result, developers waste time searching documentation, tutorials, or Stack Overflow, breaking their coding flow.

---

## ✅ Proposed Solution

CodeRacer introduces a **gesture-based AI coding assistant** where developers can simply use **hand gestures** to:

* Explain the **entire code**
* Explain a **specific line of code**
* Detect **bugs and suggest fixes**

The system uses **RAG** to retrieve relevant documentation and tutorials, ensuring that AI responses are **accurate, grounded, and non-hallucinated**.

---

## ✋ Gesture Control Logic (Core Feature)

The system uses **MediaPipe + OpenCV** to detect hand gestures in real time.
Each gesture maps to a **specific AI action**.

### 🖐️ Palm Gesture — *Explain Entire Code*

**Gesture:** Open palm (all fingers visible)
**Action:**

* Explains the **entire code file or code block**
* Covers:

  * Overall purpose of the program
  * High-level logic and flow
  * Key functions and their roles
  * How components interact

**Use Case:**
✔ When the user wants a **complete understanding** of the program.

---

### ☝️ One Finger Gesture — *Explain Pointed Line*

**Gesture:** Single finger pointing
**Action:**

* Explains **only the selected line of code**
* Includes:

  * What the line does
  * Why it is used
  * Syntax and logic explanation

**Use Case:**
✔ When the user is confused about **one specific line**.

---

### ✌️ Two Fingers Gesture — *Detect Bugs & Suggest Fixes*

**Gesture:** Two fingers raised
**Action:**

* Analyzes the code to:

  * Detect logical errors
  * Identify inefficiencies
  * Highlight bad practices
* Suggests:

  * Bug fixes
  * Optimized logic
  * Better algorithms (if applicable)

**Use Case:**
✔ When the user wants **debugging help or improvement suggestions**.

---

### 🧠 Gesture Summary Table

| Gesture        | Meaning                | AI Action                     |
| -------------- | ---------------------- | ----------------------------- |
| 🖐️ Palm       | Full code explanation  | Explains entire program       |
| ☝️ One finger  | Line-level explanation | Explains pointed line         |
| ✌️ Two fingers | Debug mode             | Detects bugs & suggests fixes |

---

## 🏗️ System Architecture

```
Camera Feed
   ↓
Gesture Detection (MediaPipe + OpenCV)
   ↓
Gesture Classification
   ↓
Query Formation
   ↓
RAG Retrieval (FAISS / Chroma)
   ↓
Prompt Augmentation
   ↓
LLM (Local / Cloud)
   ↓
Explanation / Fix Output
```

---

## 🛠️ Technology Stack

### Gesture Detection

* MediaPipe Hands
* OpenCV

### RAG Pipeline

* LangChain / Custom RAG
* FAISS / Chroma Vector Databases

### Embeddings

* Sentence-Transformers (all-MiniLM-L6-v2)
* (Optional) OpenAI Embeddings

### LLMs

* Local: LLaMA / Mistral (GGUF)
* Cloud: OpenAI GPT models (optional)

### UI

* Terminal-based output
* Popup / overlay display (extendable)

---

## 📂 Project Structure

```
coderacer-backend/
│
├── ingest_codes.py        # Code & PDF ingestion + chunking
├── extract_text_pdf.py    # PDF text extraction
├── script_and_embed.py    # Embedding & vector store creation
├── retrieval.py           # Semantic retrieval
├── rag.py                 # Core RAG pipeline
├── llm_client.py          # LLM interface
├── main.py                # Main execution
│
├── extracted/             # Chunked text (JSON)
├── vectorstores/
│   ├── faiss/
│   └── chroma/
│
├── data/                  # PDFs & resources
├── requirements.txt
├── README.md
```

---

## 🔄 Implementation Status (Verified)

### ✅ Gesture Detection

* Real-time hand tracking
* Palm, one-finger, two-finger detection working

### ✅ Data Ingestion & Chunking

* PDFs and code files ingested
* Chunking with overlap
* Metadata preserved
* **549 chunks successfully upserted**

### ✅ Embeddings & Vector Store

* Embeddings generated
* Stored in FAISS / Chroma

### ✅ Local LLM Setup

* LLaMA installed and verified
* CLI tools working

### ✅ RAG Query Flow

* Gesture → Query
* Retrieval → Context
* LLM → Explanation / Fix

---

## 🚀 How to Run the Project

### 1️⃣ Clone Repository

```bash
git clone https://github.com/gitsish/ragprojectCodeRacer.git
cd coderacer-backend
```

---

### 2️⃣ Create Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Configure Environment

Create `.env` file:

```env
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
VECTOR_DB=faiss
LLM_MODE=local
```

---

### 5️⃣ Ingest Data (Chunking)

```bash
python ingest_codes.py
```

Expected output:

```
Ingestion complete: XXX chunks upserted
```

---

### 6️⃣ Create Embeddings

```bash
python script_and_embed.py
```

---

### 7️⃣ Run Gesture Detection

```bash
python gestures.py
```

---

### 8️⃣ Run CodeRacer

```bash
python main.py
```

Now use:

* 🖐️ Palm → full code explanation
* ☝️ One finger → line explanation
* ✌️ Two fingers → bug detection & fixes

---

## 📊 Accuracy & Evaluation

### Retrieval Accuracy

* Precision@K
* Recall@K

### Generation Quality

* BLEU / ROUGE Score
* Human evaluation (correctness & relevance)

📌 — **pre-trained models + RAG** used.

---
Live Execution and Screenshots
<img width="1484" height="980" alt="image" src="https://github.com/user-attachments/assets/80aef0cf-d5f6-4035-a2ad-aee850cf062a" />
<img width="1210" height="439" alt="Screenshot 2025-09-07 111948" src="https://github.com/user-attachments/assets/7a3b78e6-8e29-4df9-8b96-03bf77278adc" />
<img width="1176" height="448" alt="Screenshot 2025-09-07 093935" src="https://github.com/user-attachments/assets/f54bf1a0-1dc7-4dc3-9d64-1ed12715aa70" />
<img width="1916" height="1077" alt="Screenshot 2025-09-07 154708" src="https://github.com/user-attachments/assets/4046b6de-7194-474b-bd82-6e8e8737d25a" />
<img width="1336" height="243" alt="Screenshot 2025-09-07 090919" src="https://github.com/user-attachments/assets/47c768d3-813d-4b39-83f6-d25889deb390" />

<img width="773" height="637" alt="Screenshot 2025-09-07 163510" src="https://github.com/user-attachments/assets/cd2c4019-be3c-4444-a889-89d99666567e" />
<img width="1328" height="519" alt="image" src="https://github.com/user-attachments/assets/b16b0e3f-ec8c-4523-87d9-34cb9f5232ec" />
<img width="1912" height="1014" alt="image" src="https://github.com/user-attachments/assets/1dbc802e-ba38-4e8f-b406-d2c7c7f14b51" />

<img width="1919" height="1006" alt="image" src="https://github.com/user-attachments/assets/f1963f19-e78b-43ea-9420-08358f074fc3" />
<img width="862" height="753" alt="image" src="https://github.com/user-attachments/assets/1f27dff5-cde5-42c5-847d-49c59c12e3ee" />
<img width="942" height="935" alt="Screenshot 2025-12-25 180005" src="https://github.com/user-attachments/assets/9a373e4e-5690-45d8-806d-ae2d0f695036" />
<img width="915" height="975" alt="Screenshot 2025-12-25 180112" src="https://github.com/user-attachments/assets/04f7bca3-b097-46d0-b13c-bde642f49356" />
<img width="862" height="753" alt="Screenshot 2025-12-25 180233" src="https://github.com/user-attachments/assets/f6cdc09a-965d-414a-804f-a3a3f0bafcf7" />
<img width="910" height="800" alt="Screenshot 2025-12-25 180428" src="https://github.com/user-attachments/assets/8569cd29-20e8-4326-94fc-b20a4129a1ab" />
<img width="953" height="969" alt="Screenshot 2025-12-25 180446" src="https://github.com/user-attachments/assets/e31bbf4e-02b8-42ed-bae4-9f455369383d" />
<img width="914" height="763" alt="Screenshot 2025-12-25 180122" src="https://github.com/user-attachments/assets/aff422b1-ac1d-409d-83b2-18b48f15d350" />

<img width="986" height="983" alt="Screenshot 2025-12-25 180433" src="https://github.com/user-attachments/assets/5bdfff0e-b88d-4534-ae5c-26072e2feb1c" />
<img width="538" height="388" alt="image" src="https://github.com/user-attachments/assets/6f3af86e-08cc-4a69-9d35-50693a342d87" />
<img width="509" height="272" alt="image" src="https://github.com/user-attachments/assets/1c602091-87f8-47cf-9647-be6fef8db70b" />


## 🌍 Real-World Impact

* Acts as a **personal coding tutor**
* Reduces context switching
* Improves learning efficiency
* Demonstrates real-world AI integration (CV + NLP + RAG)

---

## 🔮 Future Enhancements

* VS Code extension
* On-screen code highlighting
* Multimodal RAG (code + images)
* Better gesture robustness
* Cloud–local hybrid LLMs

 “CodeRacer is a gesture-controlled AI coding assistant that uses Retrieval-Augmented Generation to explain entire code, individual lines, and detect bugs in real time.”




