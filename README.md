# rag_ex

## Embeddings
![1747493365110](image/README/1747493365110.png)

* load md data into document ('document_loaders')
* doc -> text from the md + metadata, (as well as any other metadata can also be added)
* single doc can be v. long (problem), i.e. not enough to load each md file into 1 doc
* also split each doc, if they're too long on their own
* those smaller chunks -> could be a para, sentence, or even several pages
* so that -> when we search through all of this data, each chunk is gonna be more focused & more relevant to what we-re looking for ('RecursiveCharacterTextSplitter')
* to be able to query each chunk -> need to turn this into a db -> chromadb -> as it uses vector embeddings as the key

[embeddings 101](https://youtu.be/QdDoFfkVkcw?si=hY-VQVtDCF_1fcg3)

## Vector Embeddings
![1747517469016](image/README/1747517469016.png)