# Symmachus.jl

> We see the same stars, the sky is shared by all, the same world surrounds us. What does it matter what wisdom a person uses to seek for the truth? - Quintus Aurelius Symmachus (342 - 402/3 CE)

![Julia](https://img.shields.io/badge/-Julia-9558B2?style=for-the-badge&logo=julia&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

💡 This project is realized as a Master thesis at Nova School of Business and Economics, Lisbon under the supervision of Prof. Leid Zejnilovic. 

📄 Symmachus implements a semantic search engine for Portuguese politics. 

💾 The data is derived from [Debates Parlamentares](https://debates.parlamento.pt), for plenary and the [Assembleia da República](https://www.parlamento.pt/Cidadania/Paginas/DAatividadeDeputado.aspx) for deputies' activities (such as questions to the government, law proposals and other parliamentary initiatives). The ETL process was implemented in Python and is stored in another repository at the Data Science Knowledge Center.

⌨️ There are two main components in this repository:

1. *SymmachusCore*, which implements a custom document embedding procedure. That procedure is used to filter the data, according to specific speech acts, especially those stating explicit or implicit political intentions.

2. A search engine on the basis of `sentence-transformers`. This can be used to filter specific topics as well as *scan* the statements of an individual politician.



