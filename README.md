# rag_ex

## Embeddings
![1747493365110](image/README/1747493365110.png)

* load md data into document ('document_loaders')
* doc → text from the md + metadata, (as well as any other metadata can also be added)
* single doc can be v. long (problem), i.e. not enough to load each md file into 1 doc
* also split each doc, if they're too long on their own
* those smaller chunks → could be a para, sentence, or even several pages
* so that → when we search through all of this data, each chunk is gonna be more focused & more relevant to what we-re looking for ('RecursiveCharacterTextSplitter')
* to be able to query each chunk → need to turn this into a db → chromadb → as it uses vector embeddings as the key

[embeddings 101](https://youtu.be/QdDoFfkVkcw?si=hY-VQVtDCF_1fcg3)

![1747595604028](image/README/1747595604028.png)
* texts with similar meaning will be plotted together on the chart, & dis-similar → far apart
* coordinates of these embeddings → those 2 number coordinates → they represent the embeddings
* (2D chart here for example)
* in reality we use way more than 2 dimensions (in 100s or 1000s)

## Vector Embeddings
![1747517469016](image/README/1747517469016.png)

* [OpenAI Tokenizer](https://platform.openai.com/tokenizer)
* [Algo. used by OpenAI: **Byte-Pair Encoding (BPE)** or **Diagram Coding** → which uses lookup table for the mapping](https://huggingface.co/learn/llm-course/en/chapter6/5)
* [Tokenization Algo. used for BERT (by google): **WordPiece Encoding**](https://huggingface.co/learn/llm-course/en/chapter6/6?fw=pt)

[MTEB: massive text embedding benchmark](https://huggingface.co/papers/2210.07316)